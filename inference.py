import torch
import torch.nn.functional as F
import clip
import argparse
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100
import torchattacks # <<< Import the library

# Make sure shallow_encoder.py is in the same directory
from model.shallow_encoder import TextEncoder, VisionEncoder

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
# <<< END: Wrapper Class >>>


# Helper function to get class names and the test loader
def get_dataset_info(dataset_name, preprocess_fn, batch_size=64, data_dir='./data'):
    print(f"Loading test set for dataset: {dataset_name}")
    if dataset_name.lower() in ['cifar10', 'cifar100', 'cifar10_pgd']:
        DatasetClass = CIFAR10 if dataset_name.lower() in ['cifar10', 'cifar10_pgd'] else CIFAR100
        test_dataset = DatasetClass(data_dir, download=True, train=False, transform=preprocess_fn)
        class_names = test_dataset.classes
    else:
        # This part could be expanded to support the other datasets from the training script
        raise ValueError(f"Unknown or unsupported dataset for inference: {dataset_name}.")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return class_names, test_loader, len(test_dataset), test_dataset


# <<< START: Corrected Attack Functions >>>
def run_pgd_attack_batch(model_wrapper, image_batch_orig, label_batch, config, device, dtype):
    """
    Corrected PGD attack function using torch.autograd.grad to avoid graph issues
    and handle mixed-precision correctly.
    """
    epsilon, alpha, num_iter = config['epsilon'], config['alpha'], config['num_iter']
    norm_lower_limit = config['norm_lower_limit']
    norm_upper_limit = config['norm_upper_limit']

    # Start with a clone of the original images
    adv_images = image_batch_orig.clone().detach()

    for _ in range(num_iter):
        # We need to track gradients w.r.t. the adversarial images in each step
        adv_images.requires_grad = True
        
        # Forward pass to get logits
        logits = model_wrapper(adv_images)
        loss = F.cross_entropy(logits.float(), label_batch)

        # Calculate gradients of the loss w.r.t. the images
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            # Update adversarial images. The result will be float32 due to type promotion from the gradient.
            adv_images = adv_images.detach() + alpha * grad.sign()
            # Project perturbation back into the epsilon-ball
            eta = torch.clamp(adv_images - image_batch_orig, min=-epsilon, max=epsilon)
            # Clip final images to be within the valid normalized range
            adv_images = torch.clamp(image_batch_orig + eta, min=norm_lower_limit, max=norm_upper_limit)
            
            # *** THE FIX: Cast the updated images back to the model's expected dtype for the next iteration ***
            adv_images = adv_images.to(dtype)
            
    return adv_images


def run_fgsm_attack_batch(model_wrapper, image_batch_orig, label_batch, config, device, dtype):
    """ 
    A corrected FGSM attack using torch.autograd.grad.
    """
    epsilon = config['epsilon']
    norm_lower_limit = config['norm_lower_limit']
    norm_upper_limit = config['norm_upper_limit']
    
    adv_images = image_batch_orig.clone().detach().requires_grad_(True)

    logits = model_wrapper(adv_images)
    loss = F.cross_entropy(logits.float(), label_batch)

    grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        # The update results in a float32 tensor
        perturbed_images = adv_images + epsilon * grad.sign()
        adv_images = torch.clamp(perturbed_images, min=norm_lower_limit, max=norm_upper_limit)

    # The final cast to the correct dtype is sufficient here as there is no loop
    return adv_images.detach().to(dtype)
# <<< END: Corrected Attack Functions >>>


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config_used']
    best_prompt_text = checkpoint['best_prompt_text'].to(device)
    best_prompt_image = checkpoint['best_prompt_image'].to(device)

    task_name = args.task_name if args.task_name else config['task_name']
    print(f"Evaluating on task: {task_name}")

    model, preprocess = clip.load(config['backbone'], device=device)
    model.eval()
    text_encoder = TextEncoder(model)
    vision_encoder = VisionEncoder(model)
    
    class_names, test_loader, n_test_samples, _ = get_dataset_info(task_name, preprocess, args.batch_size)
    
    n_prompt_tokens_L = config['n_prompt_tokens_L']
    initial_prompt_text = config.get('initial_prompt_text', None)
    learned_prompt_pos = config.get('learned_prompt_pos', 'prefix') 
    
    prompt_prefix_placeholder = " ".join(["X"] * n_prompt_tokens_L)
    pattern_prompts = []
    
    print(f"Reconstructing prompts with settings: position='{learned_prompt_pos}', initial_text='{initial_prompt_text}'")
    
    for name in class_names:
        clean_name = name.replace("_", " ").replace("-", " ")
        initial_prompt = initial_prompt_text if initial_prompt_text else ""
        
        if learned_prompt_pos == "prefix":
            template = f"{prompt_prefix_placeholder} {initial_prompt} {clean_name}."
        elif learned_prompt_pos == "middle":
            template = f"{initial_prompt} {prompt_prefix_placeholder} {clean_name}."
        elif learned_prompt_pos == "suffix":
            template = f"{initial_prompt} {clean_name} {prompt_prefix_placeholder}."
        else: # Default to prefix
            template = f"{prompt_prefix_placeholder} {initial_prompt} {clean_name}."

        pattern_prompts.append(" ".join(template.split()))

    tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(device)
    x_token_id = clip.tokenize("X")[0, 1].item()
    ctx_start_idx = (tokenized_pattern_prompts == x_token_id).nonzero(as_tuple=True)[1].min().item()

    with torch.no_grad():
        init_pattern_embedding = model.token_embedding(tokenized_pattern_prompts).type(model.dtype)

    text_context = {
        "n_cls": len(class_names),
        "n_prompt_tokens_L": n_prompt_tokens_L,
        "init_pattern_embedding": init_pattern_embedding,
        "tokenized_pattern_prompts": tokenized_pattern_prompts,
        "ctx_start_idx": ctx_start_idx,
        "batch_size": args.batch_size,
        "pop_size": 1,
        "parallel": False
    }

    vision_context = {"n_prompt_tokens_V": config['n_prompt_tokens_V'], "batch_size": args.batch_size, "pop_size": 1, "parallel": False}
    
    text_encoder.set_context(text_context)
    vision_encoder.set_context(vision_context)

    with torch.no_grad():
        text_features = text_encoder(best_prompt_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    print(f"Preparing for evaluation with attack: {args.attack_type}")
    mean = preprocess.transforms[-1].mean
    std = preprocess.transforms[-1].std
    model_wrapper = PromptCLIPWrapper(vision_encoder, text_features, best_prompt_image, model)
    attack = None
    if args.attack_type != 'none':
        norm_mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        norm_std = torch.tensor(std).to(device).view(-1, 1, 1)
        norm_upper_limit = ((1 - norm_mean) / norm_std)
        norm_lower_limit = ((0 - norm_mean) / norm_std)

        # Config for our custom PGD/FGSM functions
        custom_attack_config = {
            'epsilon': args.epsilon, 
            'alpha': args.alpha, 
            'num_iter': args.pgd_num_iter, 
            'norm_lower_limit': norm_lower_limit, 
            'norm_upper_limit': norm_upper_limit
        }
        
        if args.attack_type == 'pgd':
            attack = lambda images, labels: run_pgd_attack_batch(model_wrapper, images, labels, custom_attack_config, device, model.dtype)
        elif args.attack_type == 'fgsm':
            attack = lambda images, labels: run_fgsm_attack_batch(model_wrapper, images, labels, custom_attack_config, device, model.dtype)
        elif args.attack_type == 'autoattack':
            attack = torchattacks.AutoAttack(model_wrapper, norm='Linf', eps=args.epsilon, version='standard')
            attack.set_normalization_used(mean=mean, std=std)
        elif args.attack_type == 'cw':
            attack = torchattacks.CW(model_wrapper, c=1, steps=100)
            attack.set_normalization_used(mean=mean, std=std)
        else:
            raise ValueError(f"Unknown attack type: {args.attack_type}")

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
    parser.add_argument("--task_name", type=str, default=None, choices=['CIFAR10', 'CIFAR100', 'CIFAR10_PGD'], help="Dataset to evaluate on. If not provided, it will be inferred from the checkpoint.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation. Use a smaller size for C&W or AutoAttack.")
    parser.add_argument('--attack_type', type=str, default='none', choices=['none', 'pgd', 'fgsm', 'autoattack', 'cw'], help='Type of adversarial attack for evaluation.')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='Epsilon for adversarial attacks.')
    parser.add_argument('--alpha', type=float, default=2/255, help='Alpha/step size for PGD attack.')
    parser.add_argument('--pgd_num_iter', type=int, default=10, help='Number of iterations for PGD attack.')
    
    args = parser.parse_args()
    main(args)