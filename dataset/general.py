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

def _construct_general_attacked_data(clean_data_info, attack_config, preprocess_fn, image_dir, name="dataset"):
    """Helper function to generate PGD attacked data for general datasets."""
    attacked_data = []
    model = attack_config['model']
    ratio = attack_config['ratio']
    eps = attack_config['eps']
    alpha = attack_config['alpha']
    steps = attack_config['steps']

    num_to_attack = int(len(clean_data_info) * ratio)
    attack_indices = np.random.choice(len(clean_data_info), num_to_attack, replace=False)
    attack_indices_set = set(attack_indices)

    print(f"Attacking {num_to_attack}/{len(clean_data_info)} images for the {name}...")

    batch_size = 64 # Internal batch size
    for i in tqdm(range(0, len(clean_data_info), batch_size), desc=f"Generating {name} Attack"):
        batch_info = clean_data_info[i:i+batch_size]
        batch_pil_images = [Image.open(os.path.join(image_dir, info[0])) for info in batch_info]
        batch_labels = torch.tensor([info[1] for info in batch_info], device=model.device)

        batch_images_tensor = torch.stack([preprocess_fn(img) for img in batch_pil_images]).to(model.device)

        indices_in_batch_to_attack = [j for j, k in enumerate(range(i, i+len(batch_labels))) if k in attack_indices_set]

        if len(indices_in_batch_to_attack) > 0:
            images_to_attack = batch_images_tensor[indices_in_batch_to_attack]
            labels_of_attacked = batch_labels[indices_in_batch_to_attack]
            
            attacked_images_tensor = model._perform_pgd_attack(images_to_attack, labels_of_attacked, eps, alpha, steps)
            batch_images_tensor[indices_in_batch_to_attack] = attacked_images_tensor

        for img_tensor, label_tensor in zip(batch_images_tensor, batch_labels):
            attacked_data.append([img_tensor.cpu(), label_tensor.cpu().item()])

    return attacked_data

class FewshotDataset(Dataset):
    def __init__(self, args, attack_config=None):
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.seed = args["seed"]
        self.shots = args["shots"]
        self.preprocess = args["preprocess"]

        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        Util.mkdir_if_missing(self.split_fewshot_dir)

        self.all_data = Util.read_split(self.split_path)
        self.all_train = self.all_data["train"]

        # --- MODIFIED: Unified Caching Logic ---
        fname_suffix = ""
        if attack_config:
            ratio = attack_config['ratio']
            eps = attack_config['eps']
            fname_suffix = f"_attack_r{ratio:.2f}_e{eps}"
        
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
                for (img_path, label, _) in clean_data_info:
                    image = self.preprocess(Image.open(os.path.join(self.image_dir, img_path)))
                    self.new_train_data.append([image, label])
            else:
                print(f"Generating and caching attacked data for {self.dataset_dir}")
                self.new_train_data = _construct_general_attacked_data(clean_data_info, attack_config, self.preprocess, self.image_dir, "train set")

            print(f"Saving data to {preprocessed_path}")
            content = {"new_train_data": self.new_train_data, "classes": self.classes}
            with open(preprocessed_path, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):
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
        self.preprocess = args["preprocess"]
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]

        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.all_data = Util.read_split(self.split_path)
        self.all_test_info = self.all_data["test"]
        
        self.test_data_dir = os.path.join(self.dataset_dir, "test_data")
        Util.mkdir_if_missing(self.test_data_dir)
        fname_suffix = ""
        if attack_config:
            ratio = attack_config['ratio']
            eps = attack_config['eps']
            fname_suffix = f"_attack_r{ratio:.2f}_e{eps}"

        preprocessed_path = os.path.join(self.test_data_dir, f"test{fname_suffix}.pkl")

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed test data from {preprocessed_path}")
            with open(preprocessed_path, "rb") as file:
                self.all_test = pickle.load(file)
        else:
            if attack_config is None:
                print(f"Generating and caching clean test data for {self.dataset_dir}")
                self.all_test = []
                for (img_path, label, _) in self.all_test_info:
                    image = self.preprocess(Image.open(os.path.join(self.image_dir, img_path)))
                    self.all_test.append([image, label])
            else:
                print(f"Generating and caching attacked test data for {self.dataset_dir}")
                self.all_test = _construct_general_attacked_data(self.all_test_info, attack_config, self.preprocess, self.image_dir, "test set")
            
            print(f"Saving test data to {preprocessed_path}")
            with open(preprocessed_path, "wb") as file:
                pickle.dump(self.all_test, file, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}


def load_test(batch_size=1, preprocess=None, root=None, dataset_dir=None, attack_config=None):
    args = {"preprocess": preprocess, "root": root, "dataset_dir": dataset_dir}
    test_data = TestDataset(args, attack_config)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_data, test_loader