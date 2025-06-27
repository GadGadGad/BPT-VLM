import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from dataset.cifar10 import load_train_cifar10, load_test_cifar10
from model.shallow_encoder import TextEncoder,VisionEncoder
from model.analysis_utils import Analysis_Util
from dataset.general import load_train,load_test
from tqdm import tqdm
import logging
logger= logging.getLogger(__name__)

class CustomDictTensorDataset(TensorDataset):
    """A custom dataset that wraps Tensors and returns them as a dictionary."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return {"image": image, "label": label}

class PromptCLIP_Shallow:
    def __init__(self,task_name,cfg):
        self.task_name = task_name
        self.opt_name = cfg["opt_name"]
        self.data_dir = cfg["data_dir"]
        self.output_dir = cfg["output_dir"]
        self.backbone = cfg["backbone"]
        self.popsize = cfg["popsize"]
        self.parallel = cfg["parallel"]
        self.batch_size = cfg["batch_size"]
        self.k_shot = cfg["k_shot"]
        self.seed = cfg["seed"]
        self.initial_prompt_text = cfg.get("initial_prompt_text", None)
        self.learned_prompt_pos = cfg.get("learned_prompt_pos", "prefix")
        self.test_every_gens = cfg.get("test_every_n_gens", None)
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.backbone,device=self.device)
        
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        
        self.loss = []
        self.acc = []
        self.acc_attack = []
        self.train_acc = []
        self._training_dataset_snapshot = None

        self.use_pgd_pre_attack = cfg.get("use_pgd_pre_attack", False)
        if self.use_pgd_pre_attack:
            self.pgd_epsilon = cfg["pgd_epsilon"]
            self.pgd_alpha = cfg["pgd_alpha"]
            self.pgd_steps = cfg["pgd_steps"]
            self.pgd_train_ratio = cfg.get("pgd_train_ratio", 1.0)
            self.pgd_test_ratio = cfg.get("pgd_test_ratio", 1.0)

        self.load_dataset()
        self._capture_training_dataset()

        self.maximize_loss = cfg.get("maximize_loss", False)
        self.best_objective_loss_value = -float('inf') if self.maximize_loss else float('inf')
        logger.info(f"--- Prompt Optimization Mode: {'MAXIMIZE' if self.maximize_loss else 'MINIMIZE'} Loss ---")
        
        self.n_prompt_tokens_L = cfg["n_prompt_tokens_L"]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.ctx_dim_L = self.model.ln_final.weight.shape[0]
        self.text_encoder = TextEncoder(self.model)

        self.n_prompt_tokens_V = cfg["n_prompt_tokens_V"]
        self.ctx_dim_V = self.model.visual.width
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.image_encoder = VisionEncoder(self.model)
        self.image_encoder.n_prompt_tokens_V = self.n_prompt_tokens_V

        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0.0
        self.best_train_accuracy = 0.0
        self.sigma = cfg["sigma"]
        self.linear_L = torch.nn.Linear(self.intrinsic_dim_L, self.n_prompt_tokens_L * self.ctx_dim_L,
                                      bias=False,device=self.device,dtype=self.dtype)
        embedding = self.model.token_embedding.weight.cpu()
        mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        mu = 0.0
        std = std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)
        for p in self.linear_L.parameters():
            torch.nn.init.normal_(p, mu, std)
        self.linear_V = torch.nn.Linear(self.intrinsic_dim_V, self.n_prompt_tokens_V * self.ctx_dim_V,
                                        bias=False, device=self.device, dtype=self.dtype)
        conv = self.model.visual.conv1.weight.cpu()
        mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(conv.reshape(-1).detach().cpu().numpy())
        mu = mu_hat*3072/self.intrinsic_dim_V
        std = std_hat * np.sqrt(3072/self.intrinsic_dim_V) * self.sigma
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)

    def _capture_training_dataset(self):
        logger.info("Capturing the training dataset for saving.")
        all_images, all_labels = [], []
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            logger.warning("train_loader not available. Cannot capture dataset snapshot.")
            return

        for batch in self.train_loader:
            all_images.append(batch["image"].cpu())
            all_labels.append(batch["label"].cpu())

        if not all_images:
            self._training_dataset_snapshot = None; return

        try:
            images_tensor = torch.cat(all_images, dim=0)
            labels_tensor = torch.cat(all_labels, dim=0)
            self._training_dataset_snapshot = {'images': images_tensor, 'labels': labels_tensor}
            logger.info(f"Captured training dataset snapshot with {images_tensor.shape[0]} samples.")
        except Exception as e:
            logger.error(f"Failed to create training dataset snapshot: {e}")
            self._training_dataset_snapshot = None


# In class PromptCLIP_Shallow:

 # In class PromptCLIP_Shallow:

 # In class PromptCLIP_Shallow:

    def _generate_pgd_attacked_set(self, data_loader, set_name):
        logger.info(f"--- Generating PGD attacked dataset for '{set_name}' ---")
        # Text features are needed for the loss. They are float16 as expected by the model.
        fixed_text_features = self.get_original_text_features().detach()
        attacked_images_list, labels_list = [], []

        for batch in tqdm(data_loader, desc=f"Attacking {set_name} set"):
            # --- Setup: Work in float32 for stability ---
            original_parallel_flag = self.parallel
            self.parallel = False
            # Get images (which are float32 from parse_batch) and labels
            images_orig_f32, labels = self.parse_batch(batch)
            images_orig_f32 = images_orig_f32.to(self.device)
            self.parallel = original_parallel_flag

            # Initialize delta perturbation in float32
            delta_img_f32 = torch.zeros_like(images_orig_f32, requires_grad=True)
            # Add random start for PGD
            delta_img_f32.data.uniform_(-self.pgd_epsilon, self.pgd_epsilon)
            # Clamp initial random perturbation to valid image range
            delta_img_f32.data = torch.clamp(images_orig_f32 + delta_img_f32.data, min=0, max=1) - images_orig_f32

            # --- PGD Attack Loop ---
            for step in range(self.pgd_steps):
                delta_img_f32.requires_grad_(True)
                
                # Create perturbed image. It's still float32 here.
                perturbed_image_f32 = images_orig_f32 + delta_img_f32
                
                # --- Forward Pass ---
                # Cast to model's expected dtype (float16) ONLY for the forward pass.
                image_for_model = perturbed_image_f32.to(self.dtype)
                
                image_features = self.model.encode_image(image_for_model)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logits = self.model.logit_scale.exp() * image_features @ fixed_text_features.t()
                loss = F.cross_entropy(logits, labels)

                # --- Backward Pass ---
                # Gradient is computed w.r.t. the float32 delta, resulting in a stable float32 grad.
                delta_img_grad = torch.autograd.grad(loss, delta_img_f32, only_inputs=True)[0]
                
                grad_sign = delta_img_grad.sign()
                
                # --- The Correct Update Step (from your working example) ---
                # The update is performed entirely in float32. By casting the grad_sign
                # to the delta's dtype, we guarantee they match and prevent dtype pollution.
                delta_img_f32.data = delta_img_f32.data + self.pgd_alpha * grad_sign.to(delta_img_f32.dtype)
                
                # Clamp the delta to the epsilon-ball (still in float32)
                delta_img_f32.data = torch.clamp(delta_img_f32.data, -self.pgd_epsilon, self.pgd_epsilon)
                # Clamp the resulting perturbed image to the valid [0, 1] range
                delta_img_f32.data = torch.clamp(images_orig_f32 + delta_img_f32.data, min=0, max=1) - images_orig_f32
            
            # --- Finalize ---
            # Final perturbed image, detached from graph. It remains float32 for the dataset.
            final_perturbed_image = (images_orig_f32 + delta_img_f32.detach())
            
            attacked_images_list.append(final_perturbed_image.cpu())
            labels_list.append(labels.cpu())
            
        all_attacked_images = torch.cat(attacked_images_list)
        all_labels = torch.cat(labels_list)
        attacked_dataset = CustomDictTensorDataset(all_attacked_images, all_labels)
        attacked_loader = DataLoader(attacked_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        logger.info(f"--- PGD attack for '{set_name}' complete. {len(attacked_dataset)} samples generated. ---")
        return attacked_dataset, attacked_loader
    # ... The rest of the file is unchanged ...
    def _combine_clean_and_attacked_sets(self, clean_loader, attacked_loader, ratio, set_name, shuffle_final=True):
        if ratio <= 0.0:
            logger.info(f"Using 100% clean data for '{set_name}' set.")
            return clean_loader.dataset, clean_loader
        if ratio >= 1.0:
            logger.info(f"Using 100% attacked data for '{set_name}' set.")
            return attacked_loader.dataset, attacked_loader
        
        logger.info(f"Combining clean and attacked data for '{set_name}' set with a {ratio:.2f} attacked ratio.")

        clean_images = torch.cat([b['image'] for b in clean_loader], dim=0)
        clean_labels = torch.cat([b['label'] for b in clean_loader], dim=0)
        attacked_images = torch.cat([b['image'] for b in attacked_loader], dim=0)
        attacked_labels = torch.cat([b['label'] for b in attacked_loader], dim=0)

        total_size = len(clean_images)
        num_attacked = int(total_size * ratio)
        
        indices = torch.randperm(total_size)
        attacked_indices = indices[:num_attacked]
        
        final_images = clean_images.clone()
        final_labels = clean_labels.clone()
        
        if num_attacked > 0:
            final_images[attacked_indices] = attacked_images[attacked_indices]
            final_labels[attacked_indices] = attacked_labels[attacked_indices]
        
        mixed_dataset = CustomDictTensorDataset(final_images, final_labels)
        mixed_loader = DataLoader(mixed_dataset, batch_size=self.batch_size, shuffle=shuffle_final, num_workers=2, pin_memory=True)

        logger.info(f"Created mixed '{set_name}' set: {num_attacked} attacked samples, {total_size - num_attacked} clean samples.")
        return mixed_dataset, mixed_loader

    def load_dataset(self):
        base_task_name = self.task_name.replace("_PGD", "")
        
        if base_task_name in ['CIFAR100', 'CIFAR10']:
            if base_task_name == 'CIFAR100':
                self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
                self.train_data, self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed)
                self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)
            elif base_task_name == 'CIFAR10':
                self.dataset = CIFAR10(self.data_dir, transform=self.preprocess, download=True)
                self.train_data, self.train_loader = load_train_cifar10(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed)
                self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
        else:
            try:
                dataset_dir_name = base_task_name + "_Gen"
                self.train_data, self.train_loader = load_train(batch_size=self.batch_size, shots=self.k_shot, preprocess=self.preprocess, root=self.data_dir, dataset_dir=dataset_dir_name, seed=self.seed)
                self.test_data, self.test_loader = load_test(batch_size=self.batch_size, preprocess=self.preprocess, root=self.data_dir, dataset_dir=dataset_dir_name)
                self.classes = self.train_data.classes
                self.n_cls = len(self.classes)
            except FileNotFoundError:
                 logger.error(f"Generic dataset directory not found for task: {dataset_dir_name}")
                 raise

        if self.use_pgd_pre_attack:
            if self.pgd_train_ratio > 0 or self.pgd_test_ratio > 0:
                logger.info("PGD pre-attack is enabled. Generating and combining datasets...")
                if "_PGD" not in self.task_name:
                    self.task_name = f"{self.task_name}_PGD"
                logger.info(f"Task name updated to: {self.task_name}")

                if self.pgd_train_ratio > 0:
                    attacked_train_data, attacked_train_loader = self._generate_pgd_attacked_set(self.train_loader, "train")
                    self.train_data, self.train_loader = self._combine_clean_and_attacked_sets(self.train_loader, attacked_train_loader, self.pgd_train_ratio, "train", shuffle_final=True)

                if self.pgd_test_ratio > 0:
                    clean_test_loader_no_shuffle = DataLoader(self.test_loader.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
                    attacked_test_data, attacked_test_loader = self._generate_pgd_attacked_set(clean_test_loader_no_shuffle, "test")
                    self.test_data, self.test_loader = self._combine_clean_and_attacked_sets(clean_test_loader_no_shuffle, attacked_test_loader, self.pgd_test_ratio, "test", shuffle_final=False)
            else:
                logger.info("PGD pre-attack enabled, but both train and test ratios are 0. Using clean data.")
    
    def get_text_information(self,caption=None):
        prompt_prefix_placeholder = " ".join(["X"] * self.n_prompt_tokens_L)

        if caption is None:
            classnames = [name.replace("_", " ").replace("-", " ") for name in self.classes]
            pattern_prompts = []

            for name in classnames:
                initial_prompt = self.initial_prompt_text if self.initial_prompt_text else ""
                
                if self.learned_prompt_pos == "prefix":
                    template = f"{prompt_prefix_placeholder} {initial_prompt} {name}."
                elif self.learned_prompt_pos == "middle":
                    template = f"{initial_prompt} {prompt_prefix_placeholder} {name}."
                elif self.learned_prompt_pos == "suffix":
                    template = f"{initial_prompt} {name} {prompt_prefix_placeholder}."
                else:
                    template = f"{prompt_prefix_placeholder} {initial_prompt} {name}."
                pattern_prompts.append(" ".join(template.split()))

            tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            x_token_id = clip.tokenize("X")[0, 1].item()
            ctx_start_idx = (tokenized_pattern_prompts == x_token_id).nonzero(as_tuple=True)[1].min().item()

            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            
            context = {
                "n_cls": self.n_cls, "n_prompt_tokens_L": self.n_prompt_tokens_L,
                "init_pattern_embedding": init_pattern_embedding, "tokenized_pattern_prompts": tokenized_pattern_prompts,
                "ctx_start_idx": ctx_start_idx, "batch_size": self.batch_size,
                "pop_size": self.popsize, "parallel": self.parallel
            }
        else:
            pattern_prompt = prompt_prefix_placeholder + " " + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            ctx_start_idx = 1
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {
                "n_cls": 1, "n_prompt_tokens_L": self.n_prompt_tokens_L,
                "init_pattern_embedding": init_pattern_embedding, "tokenized_pattern_prompts": tokenized_pattern_prompts,
                "ctx_start_idx": ctx_start_idx, "batch_size": self.batch_size,
                "pop_size": self.popsize, "parallel": self.parallel
            }
        return context

    @torch.no_grad()
    def get_original_text_features(self, specific_prompt_text=None):
        if specific_prompt_text is not None:
            text_features = self.text_encoder(specific_prompt_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

        classnames = [name.replace("_", " ").replace("-"," ") for name in self.classes]
        prompts = [f"a photo of a {name}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        text_features = self.model.encode_text(tokenized_prompts).type(self.dtype)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_information(self):
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V, "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context

    def generate_text_prompts(self,intrinsic_vectors):
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, self.ctx_dim_L)
            if self.init_prompt is not None: z = z + self.init_prompt
            prompt_list.append(z)
        return prompt_list

    def generate_visual_prompts(self,intrinsic_vectors):
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V, self.ctx_dim_V)
            visual_prompt_list.append(z)
        return visual_prompt_list

    def metric(self,logits,label):
        ce_loss = F.cross_entropy(logits, label, reduction='none')
        if self.loss_type == "ce":
            return torch.sum(ce_loss)
        elif self.loss_type == "focal":
            gamma = 2
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            return torch.sum(focal_loss)

    @torch.no_grad()
    def eval(self, prompt_zip):
        prompt_text, prompt_image = prompt_zip[0], prompt_zip[1]
        self.num_call += 1
        loss_accumulator = 0
        logit_scale = self.logit_scale.exp()

        if self.parallel:
            loss_accumulator = [0.0] * self.popsize
            all_pop_text_features = []
            for p_text in prompt_text:
                features = self.text_encoder(p_text)
                features = features / features.norm(dim=-1, keepdim=True)
                all_pop_text_features.append(features)
            pop_txt_features = torch.stack(all_pop_text_features)
        else:
            text_features = self.text_encoder(prompt_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for batch_idx, batch in enumerate(self.train_loader):
            images, labels = self.parse_batch(batch)
            images = images.to(self.dtype)

            if self.parallel:
                B_actual = labels.shape[0]
                pop_image_batch = images.view(self.popsize, B_actual, *images.shape[1:])
                for i in range(self.popsize):
                    img_feat_i = self.image_encoder(pop_image_batch[i], prompt_image[i])
                    img_feat_i = img_feat_i / img_feat_i.norm(dim=-1, keepdim=True)
                    logits_i = logit_scale * img_feat_i @ pop_txt_features[i].t()
                    loss_accumulator[i] += self.metric(logits_i, labels).item()
            else:
                img_feat = self.image_encoder(images, prompt_image)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = logit_scale * img_feat @ text_features.t()
                loss_accumulator += self.metric(logits, labels).item()

        loss_values = [x / len(self.train_data) for x in loss_accumulator] if self.parallel else loss_accumulator / len(self.train_data)
        self.loss.append(loss_values if self.parallel else [loss_values])
        
        epoch_best_loss_in_pop = max(loss_values) if self.maximize_loss else min(loss_values)
        if (self.maximize_loss and epoch_best_loss_in_pop > self.best_objective_loss_value) or \
           (not self.maximize_loss and epoch_best_loss_in_pop < self.best_objective_loss_value):
            self.best_objective_loss_value = epoch_best_loss_in_pop
            best_idx = loss_values.index(epoch_best_loss_in_pop) if self.parallel else 0
            self.best_prompt_text = (prompt_text[best_idx] if self.parallel else prompt_text).detach().clone()
            self.best_prompt_image = (prompt_image[best_idx] if self.parallel else prompt_image).detach().clone()
            logger.info(f"*** New best {'maximized' if self.maximize_loss else 'minimized'} loss: {self.best_objective_loss_value:.4f} (call {self.num_call}) ***")

        if self.test_every_gens is not None and self.test_every_gens > 0:
            trigger_interval = self.test_every_gens if self.parallel else self.test_every_gens * self.popsize
            if self.num_call > 0 and (self.num_call % trigger_interval == 0):
                self._run_intermediate_test()

        return [l * -1 if self.maximize_loss else l for l in loss_values] if self.parallel else (loss_values * -1 if self.maximize_loss else loss_values)

    def _run_intermediate_test(self):
        current_generation = self.num_call // self.popsize if self.parallel else self.num_call
        logger.info(f"\n--- Intermediate Test at Generation ~{current_generation} ---")
        acc_train = self.test_on_train_set().item()
        self.train_acc.append(acc_train)
        self.best_train_accuracy = max(acc_train, self.best_train_accuracy)
        acc_test = self.test().item()
        self.acc.append(acc_test)
        self.best_accuracy = max(acc_test, self.best_accuracy)
        logger.info(f"Train Accuracy: {acc_train:.4f} | Test Accuracy: {acc_test:.4f} (Best Test: {self.best_accuracy:.4f})")

    @torch.no_grad()
    def test_on_train_set(self):
        if self.best_prompt_text is None or self.best_prompt_image is None: return torch.tensor(0.0)
        return self._perform_test(self.train_loader)

    @torch.no_grad()
    def test(self):
        if self.best_prompt_text is None or self.best_prompt_image is None: return torch.tensor(0.0)
        return self._perform_test(self.test_loader)

    def _perform_test(self, data_loader):
        correct, total = 0., 0.
        original_parallel_text = self.text_encoder.parallel
        original_parallel_image = self.image_encoder.parallel
        self.text_encoder.parallel = self.image_encoder.parallel = False

        text_features = self.text_encoder(self.best_prompt_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        for batch in data_loader:
            image, label = self.parse_batch(batch)
            total += image.size(0)
            image_features = self.image_encoder(image.to(self.dtype), self.best_prompt_image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ text_features.t()
            correct += (logits.argmax(dim=-1) == label).float().sum()
        
        self.text_encoder.parallel = original_parallel_text
        self.image_encoder.parallel = original_parallel_image
        return correct / total

    def parse_batch(self,batch):
        image, label = batch["image"], batch["label"]
        image = image.to(device=self.device)
        if image.dtype == torch.uint8: image = image.float() / 255.0
        
        label = label.to(device=self.device)
        
        if self.parallel: image = image.repeat(self.popsize, 1, 1, 1)
        return image, label