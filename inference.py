import torch
import torch.nn.functional as F
import clip
import argparse
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torchattacks # <<< Import the library

# Make sure shallow_encoder.py is in the same directory
from shallow_encoder import TextEncoder, VisionEncoder

class PromptCLIPWrapper(torch.nn.Module):
    """
    A wrapper to make the fragmented prompt-based model compatible with torchattacks.
    torchattacks expects a single model that takes an image and returns logits.
    """
    def __init__(self, vision_encoder, text_features, image_prompt, clip_model):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_features = text_features
        self.image_prompt = image_prompt
        self.logit_scale = clip_model.logit_scale.exp()
        # The vision_encoder's context is already set outside
        
    def forward(self, images):
        # The input 'images' from torchattacks are already normalized if we set it up correctly.
        image_features = self.vision_encoder(images, self.image_prompt)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits = self.logit_scale * image_features @ self.text_features.t()
        return logits
# <<< END: New Wrapper Class >>>


# Helper function to get class names and the test loader (same as before)
def get_dataset_info(dataset_name, preprocess_fn, batch_size=64, data_dir='./data'):
    print(f"Loading test set for dataset: {dataset_name}")
    if dataset_name.lower() in ['cifar10', 'cifar10_pgd']:
        test_dataset = CIFAR10(data_dir, download=True, train=False, transform=preprocess_fn)
        class_names = test_dataset.classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return class_names, test_loader, len(test_dataset)

# Your custom PGD attack function (same as before)
def run_pgd_attack_batch(model_wrapper, image_batch_orig, label_batch, config, device, dtype):
    epsilon, alpha, num_iter = config['epsilon'], config['alpha'], config['num_iter']
    delta = torch.zeros_like(image_batch_orig, requires_grad=True)
    # The dataloader already provides normalized images, so we attack in that space.
    for _ in range(num_iter):
        loss = F.cross_entropy(model_wrapper(image_batch_orig + delta), label_batch)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return (image_batch_orig + delta.detach()).to(dtype)

def run_fgsm_attack_batch(model_wrapper, image_batch_orig, label_batch, config, device, dtype):
    """ A simple FGSM attack - essentially a single-step PGD. """
    epsilon = config['epsilon']
    delta = torch.zeros_like(image_batch_orig, requires_grad=True)
    loss = F.cross_entropy(model_wrapper(image_batch_orig + delta), label_batch)
    loss.backward()
    return (image_batch_orig + epsilon*delta.grad.detach().sign()).to(dtype)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config_used']
    best_prompt_text = checkpoint['best_prompt_text'].to(device)
    best_prompt_image = checkpoint['best_prompt_image'].to(device)

    model, preprocess = clip.load(config['backbone'], device=device)
    model.eval()
    text_encoder = TextEncoder(model)
    vision_encoder = VisionEncoder(model)
    
    class_names, test_loader, n_test_samples = get_dataset_info(args.task_name, preprocess, args.batch_size)
    
    text_context = {"n_cls": len(class_names), "n_prompt_tokens_L": config['n_prompt_tokens_L'], "init_pattern_embedding": model.token_embedding(torch.cat([clip.tokenize(p.replace("_", " ")) for p in class_names])).type(model.dtype), "tokenized_pattern_prompts": torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * config['n_prompt_tokens_L']) + " " + c + "." for c in class_names]]).to(device), "ctx_start_idx": 1, "batch_size": args.batch_size, "pop_size": 1, "parallel": False}
    vision_context = {"n_prompt_tokens_V": config['n_prompt_tokens_V'], "batch_size": args.batch_size, "pop_size": 1, "parallel": False}
    
    text_encoder.set_context(text_context)
    vision_encoder.set_context(vision_context)

    with torch.no_grad():
        text_features = text_encoder(best_prompt_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    print(f"Preparing for evaluation with attack: {args.attack_type}")
    model_wrapper = PromptCLIPWrapper(vision_encoder, text_features, best_prompt_image, model)
    attack = None
    if args.attack_type != 'none':
        pgd_config = {'epsilon': args.epsilon, 'alpha': args.alpha, 'num_iter': args.pgd_num_iter}
        if args.attack_type == 'pgd':
            # Using our custom PGD
            attack = lambda images, labels: run_pgd_attack_batch(model_wrapper, images, labels, pgd_config, device, model.dtype)
        elif args.attack_type == 'fgsm':
            # Using our custom FGSM
            attack = lambda images, labels: run_fgsm_attack_batch(model_wrapper, images, labels, pgd_config, device, model.dtype)
        else:
            if args.attack_type == 'autoattack':
                attack = torchattacks.AutoAttack(model_wrapper, norm='Linf', eps=args.epsilon, version='standard')
            elif args.attack_type == 'cw':
                # C&W is slow, usually run with a smaller number of steps for evaluation
                attack = torchattacks.CW(model_wrapper, c=1, steps=100)
            else:
                raise ValueError(f"Unknown attack type: {args.attack_type}")
            
            mean = preprocess.transforms[-1].mean
            std = preprocess.transforms[-1].std
            attack.set_normalization_used(mean=mean, std=std)

    correct_clean = 0
    correct_adv = 0
    
    for images, labels in tqdm(test_loader, desc=f"Evaluating (Attack: {args.attack_type})"):
        images, labels = images.to(device).to(model.dtype), labels.to(device)
        
        with torch.no_grad():
            outputs_clean = model_wrapper(images)
            predictions_clean = outputs_clean.argmax(dim=-1)
            correct_clean += (predictions_clean == labels).sum().item()
        
        if attack:
            adv_images = attack(images, labels)
            with torch.no_grad():
                outputs_adv = model_wrapper(adv_images)
                predictions_adv = outputs_adv.argmax(dim=-1)
                correct_adv += (predictions_adv == labels).sum().item()

    print("\n--- Evaluation Complete ---")
    clean_accuracy = (correct_clean / n_test_samples) * 100
    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({correct_clean} / {n_test_samples})")
    
    if attack:
        adv_accuracy = (correct_adv / n_test_samples) * 100
        print(f"Adversarial Accuracy ({args.attack_type.upper()}): {adv_accuracy:.2f}% ({correct_adv} / {n_test_samples})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model with various adversarial attacks.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--task_name", type=str, choices=['CIFAR10', 'CIFAR100', 'CIFAR10_PGD'], help="Dataset to evalute on.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation. Use a smaller size for C&W or AutoAttack.")
    parser.add_argument('--attack_type', type=str, default='none', choices=['none', 'pgd', 'fgsm', 'autoattack', 'cw'], help='Type of adversarial attack for evaluation.')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='Epsilon for adversarial attacks.')
    parser.add_argument('--alpha', type=float, default=2/255, help='Alpha/step size for PGD attack.')
    parser.add_argument('--pgd_num_iter', type=int, default=10, help='Number of iterations for PGD attack.')
    
    args = parser.parse_args()
    main(args)