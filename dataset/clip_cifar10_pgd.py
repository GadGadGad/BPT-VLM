import os
import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import *
import gdown
import warnings
import shutil
import time 

GOOGLE_DRIVE_FILES = {
    "train": {
        1: "1rpyHxJ-OvyulMtZLd97bx0Z0tOZ0-5cd",
        2: "19Gyl9y64YpDT3zLLQZilacV9h7k-z4gN",
        3: "1rP-xz4l9k0rQPUECL6kGhHdMvFEQ0l08",
        4: "1Ib_oqLMtglszWckXD9vnwGBjyArQpAz5",
        5: "1FABXrdF8zCYpdtUvaKBbyNHGtLlKbaGQ",
        6: "1RUU4h0-guHuoy7eOLz2iIkg2ulwTV1_K",
        7: "18h7vSwHDto57Gm7MkNIG1G-HM2i9jRUs",
        8: "1w4DZohW0BSExjF8t9ySK-UaZ7DUhpvun",
        9: "1u92IrSB_7fYdA587fuYR7-dhkaXa28Y_",
        10: "1H4aGOKeLXdwv-l3Wl6k92PYDwtTeSPon",
    },
    "test": {
        1: "12BVk6eQF8ng3U3ePQ1DxNXtwY2Y3kmhc",
        2: "1ybikWylIhULNOXWYVPWADOW2F-vvxywY",
    }
}

class PGDAttackedCIFAR10(Dataset):
    def __init__(self,
                 split: str = "train",
                 num_batches: Optional[int] = None,
                 zip_dir: str = "./zip_files",
                 dataset_dir: str = "./extracted_files",
                 download: bool = False,
                 device: str="cuda"):

        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self.filename_prefix = "trainset" if self.split == "train" else "testset"
        self.num_classes = 10
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.device = device
        max_available = len(GOOGLE_DRIVE_FILES.get(self.split, {}))
        if num_batches is None:
            self.num_batches = max_available
        else:
             if num_batches > max_available:
                 warnings.warn(f"Requested num_batches ({num_batches}) for '{self.split}' exceeds "
                               f"the number of defined file IDs ({max_available}). "
                               f"Using {max_available} batches instead.", UserWarning)
                 self.num_batches = max_available
             else:
                self.num_batches = num_batches

        self.zip_dir = zip_dir
        self.dataset_dir = dataset_dir
        self.download = download
        self.batch_map = []

        os.makedirs(self.zip_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

        for batch_idx in range(1, self.num_batches + 1):
            zip_filename = f'{self.filename_prefix}_{batch_idx}.zip'
            zip_path = os.path.join(self.zip_dir, zip_filename)
            self.batch_map.append((zip_path, batch_idx))

        self.current_batch_data = None
        self.current_batch_idx = None
        self._calculated_length = None
        self._batch_lengths = {}

        if self.download and self.split not in GOOGLE_DRIVE_FILES:
             raise ValueError(f"Google Drive File IDs not defined for split '{self.split}' in GOOGLE_DRIVE_FILES. Cannot enable download.")
         
    def _download_single_batch(self, batch_idx: int) -> bool:
        zip_filename = f'{self.filename_prefix}_{batch_idx}.zip'
        target_path = os.path.join(self.zip_dir, zip_filename)

        if os.path.exists(target_path):
            return True

        if not self.download:
            return False

        dataset_files = GOOGLE_DRIVE_FILES.get(self.split)
        if not dataset_files or batch_idx not in dataset_files:
             warnings.warn(f"No Google Drive ID defined for {self.split} batch {batch_idx}. Cannot download.", UserWarning)
             return False

        file_id_or_url = dataset_files[batch_idx]
        if not file_id_or_url or file_id_or_url.startswith("REPLACE_"):
             warnings.warn(f"Placeholder ID found for {self.split} batch {batch_idx}. Skipping download.", UserWarning)
             return False

        try:
            gdown.download(id=file_id_or_url, output=target_path, quiet=True, fuzzy=True)
            if not os.path.exists(target_path):
                 raise IOError(f"gdown download completed but the output file '{target_path}' was not found.")
            return True
        except Exception as e:
             if os.path.exists(target_path):
                 try:
                     os.remove(target_path)
                     print(f"Removed potentially incomplete download artifact: {target_path}")
                 except OSError as remove_err:
                     print(f"Warning: Could not remove potentially incomplete file {target_path}: {remove_err}")
             print(f"ERROR: Failed to download batch {batch_idx} ({zip_filename}). Reason: {e}.")
             raise ConnectionError(f"Download failed for critical batch {batch_idx}.") from e
        return False


    # --- Modified Cleanup Logic ---
    def _cleanup_batch_pt_file(self, batch_idx: int):
        """Safely attempts to delete ONLY the .pt file for a given batch index."""
        if batch_idx is None:
            return

        pt_filename = f'{self.filename_prefix}_{batch_idx}.pt'
        pt_path = os.path.join(self.dataset_dir, pt_filename)
        if os.path.exists(pt_path):
            try:
                os.remove(pt_path)
                # print(f"Debug: Cleaned up {pt_path}")
            except OSError as e:
                print(f"Warning: Could not remove batch PT file {pt_path}: {e}")

    def _cleanup_batch_zip_file(self, batch_idx: int):
        """Safely attempts to delete ONLY the .zip file for a given batch index."""
        if batch_idx is None:
            return

        zip_filename = f'{self.filename_prefix}_{batch_idx}.zip'
        zip_path = os.path.join(self.zip_dir, zip_filename)
        if os.path.exists(zip_path):
             try:
                 os.remove(zip_path)
                 # print(f"Debug: Cleaned up {zip_path}")
             except OSError as e:
                 print(f"Warning: Could not remove batch zip file {zip_path}: {e}")


    def __len__(self):
        if self._calculated_length is None:
            print(f"Calculating dataset length for {self.split} set (first call, may involve downloads)...")
            total_len = 0
            self._batch_lengths = {}
            pbar = tqdm(range(1, self.num_batches + 1), desc=f"Calculating length ({self.split} set)")
            for batch_idx in pbar:
                 zip_path = None
                 for zp, bi in self.batch_map:
                     if bi == batch_idx:
                         zip_path = zp
                         break
                 if zip_path is None:
                     # Should not happen if batch_map is correct
                     raise RuntimeError(f"Logic Error: Cannot find zip path for batch {batch_idx} in batch_map.")

                 pbar.set_postfix({"batch": batch_idx})

                 if not os.path.exists(zip_path):
                     if self.download:
                         if not self._download_single_batch(batch_idx):
                              raise FileNotFoundError(f"Required zip file {zip_path} not found and could not be downloaded.")
                     else:
                          raise FileNotFoundError(f"Zip file not found: {zip_path}. Set download=True.")

                 try:
                     # This extracts, reads length, DELETES PT, keeps ZIP
                     batch_len = self._get_batch_length_from_pt(zip_path, batch_idx)
                     self._batch_lengths[batch_idx] = batch_len
                     total_len += batch_len
                 except Exception as e:
                     print(f"\nError processing batch {batch_idx} during length calculation: {e}")
                     raise RuntimeError(f"Failed to determine length of batch {batch_idx}.") from e

            self._calculated_length = total_len
            print(f"Dataset length calculated: {self._calculated_length}")
        return self._calculated_length

    def _get_batch_length_from_pt(self, zip_path, batch_idx):
        pt_filename = f'{self.filename_prefix}_{batch_idx}.pt'
        pt_path = os.path.join(self.dataset_dir, pt_filename)
        extracted = False

        if not os.path.exists(pt_path):
            if not os.path.exists(zip_path):
                 raise FileNotFoundError(f"Source Zip file not found for length check: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    if pt_filename not in zip_ref.namelist():
                         raise FileNotFoundError(f"File '{pt_filename}' not found inside zip '{zip_path}'")
                    zip_ref.extract(pt_filename, self.dataset_dir)
                extracted = True
            except zipfile.BadZipFile:
                 raise zipfile.BadZipFile(f"Error: Bad zip file: {zip_path}.")
            except Exception as e:
                 raise IOError(f"Failed to extract {pt_filename} from {zip_path}: {e}")

        if not os.path.exists(pt_path):
             raise FileNotFoundError(f"Extracted file {pt_path} not found after extraction attempt.")

        length = 0
        try:
            data = torch.load(pt_path, map_location=self.device)
            if 'images' not in data:
                 raise KeyError(f"Missing 'images' key in {pt_path}.")
            length = len(data['images'])
            del data
        except Exception as e:
            if os.path.exists(pt_path):
                try: os.remove(pt_path)
                except OSError: pass # Ignore cleanup error here, raise original
            raise IOError(f"Failed to load or process {pt_path} for length check: {e}")

        # Cleanup extracted PT file ALWAYS
        if os.path.exists(pt_path):
            try:
                os.remove(pt_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {pt_path} after length check: {e}")

        return length

    def __getitem__(self, idx):
        if self._calculated_length is None:
            _ = len(self) 

        if idx < 0 or idx >= self._calculated_length:
             raise IndexError(f"Index {idx} out of range for dataset size {self._calculated_length}")

        running_total = 0
        target_batch_idx = -1
        local_idx = -1
        target_zip_path = None

        if not self._batch_lengths:
             raise RuntimeError("Batch lengths were not cached correctly during len() calculation.")

        sorted_batch_indices = sorted(self._batch_lengths.keys())

        for b_idx in sorted_batch_indices:
            batch_len = self._batch_lengths[b_idx]
            if idx < running_total + batch_len:
                target_batch_idx = b_idx
                local_idx = idx - running_total
                for zp, bi in self.batch_map:
                    if bi == target_batch_idx:
                        target_zip_path = zp
                        break
                if target_zip_path is None:
                    raise RuntimeError(f"Could not find zip path for target batch index {target_batch_idx}")
                break
            running_total += batch_len

        if target_batch_idx == -1:
             raise IndexError(f"Could not map index {idx} to any batch.")
        return self._load_batch_and_get_item(target_zip_path, target_batch_idx, local_idx)


    def _load_batch_and_get_item(self, zip_path, batch_idx, local_idx):
        """
        Loads the necessary batch, cleaning up ONLY the .pt file of the PREVIOUS batch.
        Keeps zip files until explicit cleanup.
        """
        pt_filename = f'{self.filename_prefix}_{batch_idx}.pt'
        pt_path = os.path.join(self.dataset_dir, pt_filename)

        if self.current_batch_idx != batch_idx:
            if self.current_batch_idx is not None:
                self._cleanup_batch_pt_file(self.current_batch_idx)

            # Clear previous data from memory
            if self.current_batch_data is not None:
                del self.current_batch_data
                self.current_batch_data = None

            if not os.path.exists(zip_path):
                if self.download:
                    if not self._download_single_batch(batch_idx):
                         raise FileNotFoundError(f"Required zip file {zip_path} not found / could not be downloaded.")
                else:
                     raise FileNotFoundError(f"Required zip file {zip_path} not found. Set download=True.")

            if not os.path.exists(pt_path):
                try:
                    # print(f"Debug: Extracting {pt_filename} for access")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                         if pt_filename not in zip_ref.namelist():
                            raise FileNotFoundError(f"File '{pt_filename}' not found inside zip '{zip_path}'")
                         zip_ref.extract(pt_filename, self.dataset_dir)
                except zipfile.BadZipFile:
                    print(f"Error: Bad zip file detected: {zip_path}. Attempting to remove.")
                    try: os.remove(zip_path)
                    except OSError: pass
                    raise zipfile.BadZipFile(f"Error: Bad zip file: {zip_path}.")
                except Exception as e:
                    raise IOError(f"Failed to extract {pt_filename} from {zip_path}: {e}")

            if not os.path.exists(pt_path):
                 raise FileNotFoundError(f"Extracted file {pt_path} not found after extraction attempt.")

            try:
                # print(f"Debug: Loading {pt_path} into memory")
                self.current_batch_data = torch.load(pt_path, map_location=self.device)
                self.current_batch_idx = batch_idx 

                if not isinstance(self.current_batch_data, dict) or \
                   'images' not in self.current_batch_data or \
                   'labels' not in self.current_batch_data:
                    self._cleanup_batch_pt_file(batch_idx)
                    raise TypeError(f"Loaded data from {pt_path} is not valid dict with 'images'/'labels'.")
                if len(self.current_batch_data['images']) != len(self.current_batch_data['labels']):
                     self._cleanup_batch_pt_file(batch_idx)
                     raise ValueError(f"Mismatch images/labels count in {pt_path}")
                if self._batch_lengths.get(batch_idx) != len(self.current_batch_data['images']):
                    warnings.warn(f"Length mismatch batch {batch_idx}: Cached={self._batch_lengths.get(batch_idx)}, Actual={len(self.current_batch_data['images'])}.", UserWarning)


            except Exception as e:
                 self._cleanup_batch_pt_file(batch_idx) 
                 self.current_batch_data = None
                 raise IOError(f"Failed to load data from {pt_path}: {e}")

        if self.current_batch_data is None or self.current_batch_idx != batch_idx:
             raise RuntimeError(f"Error: Data for batch {batch_idx} not loaded correctly.")

        try:
            image = self.current_batch_data['images'][local_idx]
            label = self.current_batch_data['labels'][local_idx]
            return image, label
        except KeyError as e:
            raise KeyError(f"Missing key '{e}' in loaded batch {self.current_batch_idx}.")
        except IndexError:
            actual_len = len(self.current_batch_data.get('images', []))
            raise IndexError(f"Local index {local_idx} out of range for batch {self.current_batch_idx} (len: {actual_len}). Global idx: {idx}.")
        except Exception as e:
             raise RuntimeError(f"Unexpected error retrieving item {local_idx} from batch {self.current_batch_idx}: {e}")

    def cleanup(self, delete_zips=True):
        """
        Explicitly cleans up the currently loaded batch's PT file and optionally
        ALL zip files associated with this dataset instance.
        """
        print("\nRunning cleanup...")
        if self.current_batch_idx is not None:
            print(f"Cleaning up PT file for last loaded batch: {self.current_batch_idx}")
            self._cleanup_batch_pt_file(self.current_batch_idx)
            self.current_batch_data = None
            self.current_batch_idx = None
        else:
            print("No batch PT file currently loaded.")

        if delete_zips:
            print("Cleaning up ALL associated ZIP files...")
            cleaned_zips = 0
            for zip_path, batch_idx in self.batch_map:
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                        cleaned_zips += 1
                    except OSError as e:
                        print(f"Warning: Could not remove zip file {zip_path}: {e}")
            print(f"Removed {cleaned_zips} ZIP files.")
        else:
            print("Skipping ZIP file cleanup as requested.")

class Cifar_FewshotDataset(Dataset):
    def __init__(self, args):
        self.shots = args["shots"]
        self.all_train = PGDAttackedCIFAR10(split="train", download=True, num_batches=10)
        self.all_train.cleanup(delete_zips=True)
        self.new_train_data = self.construct_few_shot_data()
        pass

    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):
        return {"image": self.new_train_data[idx][0], "label": self.new_train_data[idx][1]}

      
    def construct_few_shot_data(self):
        new_train_data = []
        train_shot_count = {i: 0 for i in range(self.all_train.num_classes)}
        needed_shots = self.shots * self.all_train.num_classes
        collected_shots = 0

        try:
            _ = len(self.all_train) 
        except Exception as e:
            print(f"Error calculating initial length: {e}")
            pass

        batch_indices = list(self.all_train._batch_lengths.keys())
        np.random.shuffle(batch_indices) # Shuffle batch order

        print(f"Constructing few-shot data ({self.shots} shots/class)...")
        pbar = tqdm(batch_indices, desc="Processing batches for few-shot")
        for batch_idx in pbar:
            if collected_shots >= needed_shots:
                break # Stop if we have enough samples

            batch_data = None
            try:
                # Find zip path for this batch_idx
                zip_path = None
                for zp, bi in self.all_train.batch_map:
                    if bi == batch_idx:
                        zip_path = zp
                        break
                if zip_path is None: continue # Should not happen

                # --- Simplified Load Logic (Adapt from _load_batch_and_get_item) ---
                pt_filename = f'{self.all_train.filename_prefix}_{batch_idx}.pt'
                pt_path = os.path.join(self.all_train.dataset_dir, pt_filename)

                if not os.path.exists(zip_path):
                    if self.all_train.download:
                        if not self.all_train._download_single_batch(batch_idx):
                            print(f"Warning: Could not download zip for batch {batch_idx}")
                            continue
                    else:
                        print(f"Warning: Zip file not found for batch {batch_idx}")
                        continue
                # Extract
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extract(pt_filename, self.all_train.dataset_dir)
                except Exception as e:
                    print(f"Warning: Failed to extract {pt_filename}: {e}")
                    continue # Skip this batch

                batch_data = torch.load(pt_path, map_location='cpu') # Force CPU load

                images = batch_data['images']
                labels = batch_data['labels']

                indices_in_batch = list(range(len(images)))
                np.random.shuffle(indices_in_batch) 

                for local_idx in indices_in_batch:
                    label = labels[local_idx].item()
                    if train_shot_count[label] < self.shots:
                        img = images[local_idx]
                        new_train_data.append([img, torch.tensor(label)]) 
                        train_shot_count[label] += 1
                        collected_shots += 1
                        pbar.set_postfix({"Collected": collected_shots})
                        if collected_shots >= needed_shots:
                            break

            except Exception as e:
                print(f"\nError processing batch {batch_idx} during few-shot construction: {e}")
            finally:
                if batch_data is not None:
                    del batch_data
                    del images 
                    del labels
                    torch.cuda.empty_cache() 

                self.all_train._cleanup_batch_pt_file(batch_idx)

            if collected_shots >= needed_shots:
                break # Stop outer loop

        del self.all_train

        if collected_shots < needed_shots:
            warnings.warn(f"Could only collect {collected_shots}/{needed_shots} samples.", UserWarning)

        return new_train_data # List of [image_tensor, label_tensor_or_int]


def load_train_cifar10_pgd(batch_size=1,shots=16):
    args = {"shots":shots}
    train_data = Cifar_FewshotDataset(args)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True)
    return train_data,train_loader

class Cifar_TestDataset(Dataset):
    def __init__(self):
        self.all_test = PGDAttackedCIFAR10(split="test", download=True, num_batches=2)
        self.all_test.cleanup(delete_zips=True)

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}


def load_test_cifar10_pgd(batch_size=1):
    test_data = Cifar_TestDataset()
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False,num_workers=4, pin_memory=True)
    return test_data, test_loader

print(os.path.expanduser("../dataset"))
