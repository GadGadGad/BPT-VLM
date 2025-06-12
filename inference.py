import torch
import torch.nn.functional as F
import clip
import argparse
from tqdm import tqdm
from torchvision.datasets import CIFAR10

# Make sure shallow_encoder.py is in the same directory
from model.shallow_encoder import TextEncoder, VisionEncoder

# Helper function to get class names and the test loader
def get_dataset_info(dataset_name, preprocess_fn, batch_size=64, data_dir='./data'):
    """Loads class names and the test dataloader for a given dataset."""
    print(f"Loading test set for dataset: {dataset_name}")
    if dataset_name.lower() in ['cifar10', 'cifar10_pgd']:
        test_dataset = CIFAR10(data_dir, download=True, train=False, transform=preprocess_fn)
        class_names = test_dataset.classes
    # Add other datasets from your training script as needed
    # elif dataset_name.lower() == 'cifar100':
    #     test_dataset = CIFAR100(data_dir, download=True, train=False, transform=preprocess_fn)
    #     class_names = test_dataset.classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please add it to the `get_dataset_info` function.")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return class_names, test_loader, len(test_dataset)

def run_pgd_attack_batch(model, text_encoder, vision_encoder, image_batch_orig, label_batch, text_features_for_attack, image_prompt_for_attack, preprocess, config, device, dtype):
    """
    Performs a PGD attack on a BATCH of images.
    """
    epsilon = config['epsilon']
    alpha = config['alpha']
    num_iter = config['num_iter']
    
    # Get normalization parameters
    mean = preprocess.transforms[-1].mean
    std = preprocess.transforms[-1].std
    norm_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    norm_upper_limit = ((1 - norm_mean) / norm_std)
    norm_lower_limit = ((0 - norm_mean) / norm_std)
    
    # Initialize perturbation
    delta = torch.zeros_like(image_batch_orig, requires_grad=True, device=device)
    delta.data.uniform_(-epsilon, epsilon)
    delta.data = torch.clamp(image_batch_orig + delta.data, min=norm_lower_limit, max=norm_upper_limit) - image_batch_orig
    
    for _ in range(num_iter):
        delta.requires_grad_(True)
        perturbed_image_batch = (image_batch_orig + delta).to(dtype)
        
        with torch.enable_grad():
            image_features = vision_encoder(perturbed_image_batch, image_prompt_for_attack)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = model.logit_scale.exp() * image_features @ text_features_for_attack.t()
            loss = F.cross_entropy(logits, label_batch)
            
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            
        delta.data = delta.data + alpha * grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(image_batch_orig + delta.data, min=norm_lower_limit, max=norm_upper_limit) - image_batch_orig

    final_perturbed_image_batch = (image_batch_orig + delta.detach()).clamp(min=norm_lower_limit, max=norm_upper_limit)
    return final_perturbed_image_batch.to(dtype)


def main(args):
    """
    Evaluates a trained model on a full dataset (e.g., CIFAR-10 test set).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load checkpoint and config
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config_used']
    best_prompt_text = checkpoint['best_prompt_text'].to(device)
    best_prompt_image = checkpoint['best_prompt_image'].to(device)

    # 2. Load CLIP model, encoders, and preprocessor
    model, preprocess = clip.load(config['backbone'], device=device)
    model.eval()
    text_encoder = TextEncoder(model)
    vision_encoder = VisionEncoder(model)
    
    # 3. Get dataset info and prepare text context
    class_names, test_loader, n_test_samples = get_dataset_info(args.task_name, preprocess, args.batch_size)
    n_cls = len(class_names)
    
    # Replicate text prompt creation
    prompt_prefix_placeholder = " ".join(["X"] * config['n_prompt_tokens_L'])
    initial_prompt = config.get('initial_prompt_text') if config.get('initial_prompt_text') else ""
    learned_prompt_pos = config.get('learned_prompt_pos', 'prefix')
    
    pattern_prompts = []
    for name in class_names:
        clean_name = name.replace("_", " ").replace("-", " ")
        if learned_prompt_pos == "prefix": template = f"{prompt_prefix_placeholder} {initial_prompt} {clean_name}."
        elif learned_prompt_pos == "middle": template = f"{initial_prompt} {prompt_prefix_placeholder} {clean_name}."
        else: template = f"{initial_prompt} {clean_name} {prompt_prefix_placeholder}."
        pattern_prompts.append(" ".join(template.split()))

    tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(device)
    x_token_id = clip.tokenize("X")[0, 1].item()
    ctx_start_idx = (tokenized_pattern_prompts == x_token_id).nonzero(as_tuple=True)[1].min().item()

    with torch.no_grad():
        init_pattern_embedding = model.token_embedding(tokenized_pattern_prompts).type(model.dtype)

    text_context = {"n_cls": n_cls, "n_prompt_tokens_L": config['n_prompt_tokens_L'], "init_pattern_embedding": init_pattern_embedding, "tokenized_pattern_prompts": tokenized_pattern_prompts, "ctx_start_idx": ctx_start_idx, "batch_size": args.batch_size, "pop_size": 1, "parallel": False}
    text_encoder.set_context(text_context)
    vision_context = {
        "n_prompt_tokens_V": config['n_prompt_tokens_V'],
        "batch_size": args.batch_size, 
        "pop_size": 1,
        "parallel": False
    }
    vision_encoder.set_context(vision_context)

    # Pre-compute text features once
    with torch.no_grad():
        text_features = text_encoder(best_prompt_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 4. Initialize accumulators for evaluation
    correct_clean = 0
    correct_pgd = 0
    
    # 5. Loop over the test set
    for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        images, labels = images.to(device), labels.to(device)
        
        # --- Clean Evaluation ---
        with torch.no_grad():
            image_features_clean = vision_encoder(images, best_prompt_image)
            image_features_clean /= image_features_clean.norm(dim=-1, keepdim=True)
            logits_clean = model.logit_scale.exp() * image_features_clean @ text_features.t()
            predictions_clean = logits_clean.argmax(dim=-1)
            correct_clean += (predictions_clean == labels).sum().item()
            
        if args.pgd_eval:
            pgd_config = {
                'epsilon': args.pgd_epsilon,
                'alpha': args.pgd_alpha,
                'num_iter': args.pgd_num_iter
            }
            perturbed_images = run_pgd_attack_batch(
                model, text_encoder, vision_encoder,
                images, labels, text_features, best_prompt_image,
                preprocess, pgd_config, device, model.dtype
            )
            
            with torch.no_grad():
                image_features_pgd = vision_encoder(perturbed_images, best_prompt_image)
                image_features_pgd /= image_features_pgd.norm(dim=-1, keepdim=True)
                logits_pgd = model.logit_scale.exp() * image_features_pgd @ text_features.t()
                predictions_pgd = logits_pgd.argmax(dim=-1)
                correct_pgd += (predictions_pgd == labels).sum().item()

    print("\n--- Evaluation Complete ---")
    clean_accuracy = (correct_clean / n_test_samples) * 100
    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({correct_clean} / {n_test_samples})")
    
    if args.pgd_eval:
        pgd_accuracy = (correct_pgd / n_test_samples) * 100
        print(f"PGD Robust Accuracy: {pgd_accuracy:.2f}% ({correct_pgd} / {n_test_samples})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained Shallow Prompt CLIP model on a full dataset.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--task_name", type=str, help="Task name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    
    # PGD Evaluation Arguments
    pgd_group = parser.add_argument_group('PGD Evaluation Parameters')
    pgd_group.add_argument("--pgd_eval", action='store_true', help="Enable PGD evaluation on the test set.")
    pgd_group.add_argument('--pgd_epsilon', type=float, default = 8/255, help='Epsilon for PGD attack')
    pgd_group.add_argument('--pgd_alpha', type=float, default = 2/255, help='Alpha for PGD attack') 
    pgd_group.add_argument('--pgd_num_iter', type=int, default = 10, help='Number of iterations for PGD attack')
    
    args = parser.parse_args()
    main(args)