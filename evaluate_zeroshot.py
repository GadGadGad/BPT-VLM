import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
import torchattacks
import time

from clip.custom_clip import CustomCLIP

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default="ViT-B/32", help="Backbone to use")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--task_name", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100"], help="Dataset to use")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
parser.add_argument("--prompt", type=str, default=None, help="Custom prompt (WARNING: Likely incompatible with PGD evaluation if used)")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = args.backbone
BATCH_SIZE = args.batch_size
PGD_EPS = 4/255
PGD_ALPHA = PGD_EPS/4
PGD_STEPS = 10

print("Loading CLIP model...")
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model = model.float()
model.eval()

print(f"CLIP model '{MODEL_NAME}' loaded.")
print("Input resolution:", model.visual.input_resolution)

# Load dataset
dataset_name = args.task_name.upper()
if dataset_name == "CIFAR10":
    print("Loading CIFAR-10 dataset...")
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
elif dataset_name == "CIFAR100":
    print("Loading CIFAR-100 dataset...")
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=preprocess)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = test_dataset.classes
NUM_CLASSES = len(class_names)
print(f"{dataset_name} dataset loaded. Number of classes: {NUM_CLASSES}")

def get_default_text_features(clip_model, class_names, device):
    print("Generating text features using default class name prompts...")
    text_descriptions = [f"{class_name}" for class_name in class_names]
    text_tokens = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_tokens)
        features /= features.norm(dim=-1, keepdim=True)
    print(f"Generated default text features with shape: {features.shape}")
    return features.float()

text_features = get_default_text_features(model, class_names, DEVICE)

clip_custom = CustomCLIP(model, text_features).to(DEVICE)

atk = torchattacks.PGD(clip_custom, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
print(f"PGD Attack defined: eps={PGD_EPS}, alpha={PGD_ALPHA}, steps={PGD_STEPS}")
print("Note: Attack operates on CLIP's preprocessed (normalized) images.")

clean_correct_total = 0
robust_correct_total = 0
total_images = 0

print("\nStarting evaluation loop...")
for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        logits_clean = clip_custom(images)
        predictions_clean = logits_clean.argmax(dim=-1)
        clean_correct_total += (predictions_clean == labels).sum().item()

    adv_images = atk(images, labels)
    with torch.no_grad():
        logits_adv = clip_custom(adv_images)
        predictions_adv = logits_adv.argmax(dim=-1)
        robust_correct_total += (predictions_adv == labels).sum().item()

    total_images += images.shape[0]
    
    if i % 10 == 0 or i == len(test_loader) - 1:
        print(f"Batch {i+1}/{len(test_loader)} | Clean Acc: {100 * clean_correct_total / total_images:.2f}% | Robust Acc: {100 * robust_correct_total / total_images:.2f}%")

if total_images > 0:
    print("\n--- Final Evaluation Results ---")
    print(f"Total images evaluated: {total_images}")
    print(f"Overall Clean Accuracy: {100 * clean_correct_total / total_images:.2f}%")
    print(f"Overall Robust Accuracy against PGD: {100 * robust_correct_total / total_images:.2f}%")
