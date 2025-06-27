import torch
import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
#---------------------------------------------- device:cpu dtype:float32-----------------------------------------------

def _construct_attacked_data(clean_data, attack_config, preprocess_fn, name="dataset"):
    """Helper function to generate PGD attacked data."""
    attacked_data = []
    model = attack_config['model']
    ratio = attack_config['ratio']
    eps = attack_config['eps']
    alpha = attack_config['alpha']
    steps = attack_config['steps']
    
    num_to_attack = int(len(clean_data) * ratio)
    attack_indices = np.random.choice(len(clean_data), num_to_attack, replace=False)
    attack_indices_set = set(attack_indices)

    print(f"Attacking {num_to_attack}/{len(clean_data)} images for the {name}...")
    
    batch_size = 64 # Internal batch size for attack generation
    for i in tqdm(range(0, len(clean_data), batch_size), desc=f"Generating {name} Attack"):
        batch_pil_images = [item[0] for item in clean_data[i:i+batch_size]]
        batch_labels = torch.tensor([item[1] for item in clean_data[i:i+batch_size]], device=model.device)
        
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

class Cifar_FewshotDataset(Dataset):
    def __init__(self, args, attack_config=None):
        self.shots = args["shots"]
        self.preprocess = args["preprocess"]
        self.seed = args["seed"]
        
        self.all_train_base = CIFAR10(os.path.expanduser("../dataset"), transform=None, download=True, train=True)
        clean_data = self.construct_few_shot_data()

        if attack_config is None:
            print("Using clean training data for CIFAR10.")
            self.new_train_data = [[self.preprocess(item[0]), item[1]] for item in clean_data]
        else:
            cache_dir = os.path.join(os.path.expanduser("../dataset"), "cifar10_cache")
            os.makedirs(cache_dir, exist_ok=True)
            ratio = attack_config['ratio']
            eps = attack_config['eps']
            cache_fname = f"train_shot_{self.shots}_seed_{self.seed}_ratio_{ratio:.2f}_eps_{eps}.pkl"
            cache_path = os.path.join(cache_dir, cache_fname)
            
            if os.path.exists(cache_path):
                print(f"Loading cached attacked training data from {cache_path}")
                with open(cache_path, "rb") as f:
                    self.new_train_data = pickle.load(f)
            else:
                print(f"Generating attacked training data, will cache to {cache_path}")
                self.new_train_data = _construct_attacked_data(clean_data, attack_config, self.preprocess, "train set")
                with open(cache_path, "wb") as f:
                    pickle.dump(self.new_train_data, f)

    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):
        return {"image": self.new_train_data[idx][0], "label": self.new_train_data[idx][1]}

    def construct_few_shot_data(self):
        new_train_data = []
        train_shot_count={}
        all_indices = [_ for _ in range(len(self.all_train_base))]
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)

        for index in all_indices:
            label = self.all_train_base[index][1]
            if label not in train_shot_count:
                train_shot_count[label]=0

            if train_shot_count[label]<self.shots:
                tmp = self.all_train_base[index]
                new_train_data.append(tmp)
                train_shot_count[label] += 1
        return new_train_data

def load_train_cifar10(batch_size=1,shots=16,preprocess=None,seed=42, attack_config=None):
    args = {"shots":shots,"preprocess":preprocess, "seed": seed}
    train_data = Cifar_FewshotDataset(args, attack_config=attack_config)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    return train_data,train_loader

class Cifar_TestDataset(Dataset):
    def __init__(self, args, attack_config=None):
        self.preprocess = args["preprocess"]
        all_test_base = CIFAR10(os.path.expanduser("../dataset"), transform=None, download=True, train=False)
        clean_data = list(all_test_base)

        if attack_config is None:
            print("Using clean test data for CIFAR10.")
            self.all_test = [[self.preprocess(item[0]), item[1]] for item in clean_data]
        else:
            cache_dir = os.path.join(os.path.expanduser("../dataset"), "cifar10_cache")
            os.makedirs(cache_dir, exist_ok=True)
            ratio = attack_config['ratio']
            eps = attack_config['eps']
            cache_fname = f"test_ratio_{ratio:.2f}_eps_{eps}.pkl"
            cache_path = os.path.join(cache_dir, cache_fname)

            if os.path.exists(cache_path):
                print(f"Loading cached attacked test data from {cache_path}")
                with open(cache_path, "rb") as f:
                    self.all_test = pickle.load(f)
            else:
                print(f"Generating attacked test data, will cache to {cache_path}")
                self.all_test = _construct_attacked_data(clean_data, attack_config, self.preprocess, "test set")
                with open(cache_path, "wb") as f:
                    pickle.dump(self.all_test, f)

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}


def load_test_cifar10(batch_size=1,preprocess=None, attack_config=None):
    args = {"preprocess":preprocess}
    test_data = Cifar_TestDataset(args, attack_config=attack_config)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=4)
    return test_data,test_loader

print(os.path.expanduser("../dataset"))