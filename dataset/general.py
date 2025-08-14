# --- START OF FILE general.py ---

import torch
import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader
import argparse
from dataset.utils import Util
import pickle
from PIL import Image
from tqdm import tqdm

# --- MODIFIED: The helper function is refactored to be like cifar10.py's version ---
def _construct_general_attacked_data(clean_data_info, attack_config, image_dir, name="dataset"):
    """
    Helper function to generate attacked data for general datasets using a surrogate model.
    This function is now aligned with the logic in cifar10.py.
    """
    attacked_data = []

    # Unpack the config, same as in cifar10.py
    model_wrapper = attack_config['model_wrapper']
    surrogate_model = attack_config['surrogate_clip_model']
    surrogate_preprocess_fn = attack_config['surrogate_preprocess']
    attack_type = attack_config.get('type', 'PGD') # Default to PGD if not specified
    ratio = attack_config['ratio']
    
    num_to_attack = int(len(clean_data_info) * ratio)
    attack_indices = np.random.choice(len(clean_data_info), num_to_attack, replace=False)
    attack_indices_set = set(attack_indices)

    # Try to get model name from its attributes for logging, like in cifar10.py
    try:
        # A robust way to get a unique model identifier
        model_name = f"ViT-L-{surrogate_model.visual.patch_size}" if hasattr(surrogate_model.visual, 'patch_size') else f"ViT-{surrogate_model.visual.transformer.width}"
    except:
        model_name = "Unknown_ViT"

    print(f"Generating '{attack_type.upper()}' attacks using surrogate '{model_name}' "
          f"for {num_to_attack}/{len(clean_data_info)} images for the {name}...")

    batch_size = 64  # Internal batch size for attack generation
    for i in tqdm(range(0, len(clean_data_info), batch_size), desc=f"Generating {name} Attack"):
        batch_info = clean_data_info[i:i + batch_size]
        # We always start from the clean PIL images
        batch_pil_images = [Image.open(os.path.join(image_dir, info[0])) for info in batch_info]
        batch_labels = torch.tensor([info[1] for info in batch_info], device=model_wrapper.device)

        # Preprocess images specifically for the SURROGATE model
        batch_images_tensor = torch.stack([surrogate_preprocess_fn(img) for img in batch_pil_images]).to(model_wrapper.device)

        indices_in_batch_to_attack = [j for j, k in enumerate(range(i, i + len(batch_labels))) if k in attack_indices_set]

        if len(indices_in_batch_to_attack) > 0:
            images_to_attack = batch_images_tensor[indices_in_batch_to_attack]
            labels_of_attacked = batch_labels[indices_in_batch_to_attack]
            
            # --- MODIFIED: Use the model_wrapper to perform the attack, just like in cifar10.py ---
            # This decouples the dataset from the attack implementation.
            attacked_images_tensor = model_wrapper.perform_attack_on_batch(
                images_to_attack, labels_of_attacked, attack_config
            )
            batch_images_tensor[indices_in_batch_to_attack] = attacked_images_tensor

        # The final dataset is composed of tensors (attacked or clean) and labels.
        for img_tensor, label_tensor in zip(batch_images_tensor, batch_labels):
            attacked_data.append([img_tensor.cpu(), label_tensor.cpu().item()])

    return attacked_data

class FewshotDataset(Dataset):
    def __init__(self, args, attack_config=None):
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.seed = args["seed"]
        self.shots = args["shots"]
        self.preprocess = args["preprocess"] # This is for the TARGET model

        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        Util.mkdir_if_missing(self.split_fewshot_dir)

        self.all_data = Util.read_split(self.split_path)
        self.all_train = self.all_data["train"]

        # --- MODIFIED: Unified Caching Logic, now including surrogate model name ---
        fname_suffix = ""
        if attack_config:
            ratio = attack_config['ratio']
            eps = attack_config['eps']
            surrogate_model = attack_config['surrogate_clip_model']
            # Get a unique name for the surrogate model for caching
            surrogate_name = surrogate_model.visual.transformer.width
            fname_suffix = f"_attack_r{ratio:.2f}_e{eps}_surrogate_{surrogate_name}"
        
        preprocessed_path = os.path.join(self.split_fewshot_dir, f"shot_{self.shots}_seed_{self.seed}{fname_suffix}.pkl")

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed few-shot data from {preprocessed_path}")
            with open(preprocessed_path, "rb") as file:
                content = pickle.load(file)
                self.new_train_data = content["new_train_data"]
                self.classes = content["classes"]
        else:
            clean_data_info, self.classes = self.construct_few_shot_data()

            if attack_config is None:
                print(f"Generating and caching clean data for {self.dataset_dir}")
                self.new_train_data = []
                # For clean data, we use the target model's preprocess function
                for (img_path, label, _) in clean_data_info:
                    image = self.preprocess(Image.open(os.path.join(self.image_dir, img_path)))
                    self.new_train_data.append([image, label])
            else:
                print(f"Generating and caching attacked data for {self.dataset_dir}")
                # --- MODIFIED: Call the new refactored helper function ---
                # It no longer needs `self.preprocess` as it uses the surrogate's preprocess from the config
                self.new_train_data = _construct_general_attacked_data(clean_data_info, attack_config, self.image_dir, "train set")

            print(f"Saving data to {preprocessed_path}")
            content = {"new_train_data": self.new_train_data, "classes": self.classes}
            with open(preprocessed_path, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):
        # Data is already pre-processed and is a tensor
        return {"image": self.new_train_data[idx][0], "label": self.new_train_data[idx][1]}

    def construct_few_shot_data(self):
        new_train_data_info = []
        train_shot_count = {}
        classes_dict = {}
        all_indices = list(range(len(self.all_train)))
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)

        for index in all_indices:
            path, label, classname = self.all_train[index]

            if label not in train_shot_count:
                train_shot_count[label] = 0
                classes_dict[label] = classname
            if train_shot_count[label] < self.shots:
                new_train_data_info.append((path, label, classname))
                train_shot_count[label] += 1
        classes = [classes_dict[i] for i in range(len(classes_dict))]
        return new_train_data_info, classes


def load_train(batch_size=1, seed=42, shots=16, preprocess=None, root=None, dataset_dir=None, attack_config=None):
    args = {"shots": shots, "preprocess": preprocess, "root": root, "dataset_dir": dataset_dir, "seed": seed}
    train_data = FewshotDataset(args, attack_config)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_data, train_loader


class TestDataset(Dataset):
    def __init__(self, args, attack_config=None):
        self.preprocess = args["preprocess"] # This is for the TARGET model
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.all_data = Util.read_split(self.split_path)
        self.all_test_info = self.all_data["test"]
        
        self.test_data_dir = os.path.join(self.dataset_dir, "test_data")
        Util.mkdir_if_missing(self.test_data_dir)
        
        # --- MODIFIED: Caching logic updated to include surrogate model name ---
        fname_suffix = ""
        if attack_config:
            ratio = attack_config['ratio']
            eps = attack_config['eps']
            surrogate_model = attack_config['surrogate_clip_model']
            surrogate_name = surrogate_model.visual.transformer.width
            fname_suffix = f"_attack_r{ratio:.2f}_e{eps}_surrogate_{surrogate_name}"

        preprocessed_path = os.path.join(self.test_data_dir, f"test{fname_suffix}.pkl")

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed test data from {preprocessed_path}")
            with open(preprocessed_path, "rb") as file:
                self.all_test = pickle.load(file)
        else:
            if attack_config is None:
                print(f"Generating and caching clean test data for {self.dataset_dir}")
                self.all_test = []
                # For clean data, use the target model's preprocess function
                for (img_path, label, _) in self.all_test_info:
                    image = self.preprocess(Image.open(os.path.join(self.image_dir, img_path)))
                    self.all_test.append([image, label])
            else:
                print(f"Generating and caching attacked test data for {self.dataset_dir}")
                # --- MODIFIED: Call the new refactored helper function ---
                self.all_test = _construct_general_attacked_data(self.all_test_info, attack_config, self.image_dir, "test set")
            
            print(f"Saving test data to {preprocessed_path}")
            with open(preprocessed_path, "wb") as file:
                pickle.dump(self.all_test, file, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        # Data is already pre-processed and is a tensor
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}


def load_test(batch_size=1, preprocess=None, root=None, dataset_dir=None, attack_config=None):
    args = {"preprocess": preprocess, "root": root, "dataset_dir": dataset_dir}
    test_data = TestDataset(args, attack_config)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4) # Set shuffle=False for test loader
    return test_data, test_loader