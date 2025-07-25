import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchvision.datasets import CIFAR100, CIFAR10
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from dataset.cifar10 import load_train_cifar10, load_test_cifar10
from model.shallow_encoder import TextEncoder,VisionEncoder
from model.analysis_utils import Analysis_Util
from dataset.general import load_train,load_test
from tqdm import tqdm
import logging
logger= logging.getLogger(__name__)

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
        self.cfg = cfg # Store cfg for PGD params
        self.initial_prompt_text = cfg.get("initial_prompt_text", None)
        self.learned_prompt_pos = cfg.get("learned_prompt_pos", "prefix")
        self.test_every_gens = cfg.get("test_every_n_gens", None) 
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.backbone,device=self.device)
        self.loss = []
        self.acc = []
        self.acc_clean_during_attack_run = []
        self.acc_attack = [] # Kept for list structure consistency
        self.train_acc = []
        self._training_dataset_snapshot = None # Holder for the dataset
        self.test_loader_clean = None # Holder for the clean test loader

        # --- Noise injection parameters ---
        self.noise_type_text = cfg.get("noise_type_text", "none")
        self.noise_type_visual = cfg.get("noise_type_visual", "none")
        self.noise_level = cfg.get("noise_level", 0.1)
        
        # --- REORDERED: Initialize all model-dependent components BEFORE dataset loading ---
        
        # Text Encoder
        self.n_prompt_tokens_L = cfg["n_prompt_tokens_L"]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.ctx_dim_L = self.model.ln_final.weight.shape[0]
        self.text_encoder = TextEncoder(self.model)

        # Image Encoder
        self.n_prompt_tokens_V = cfg["n_prompt_tokens_V"]
        self.ctx_dim_V = self.model.visual.width
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.image_encoder = VisionEncoder(self.model)
        self.image_encoder.n_prompt_tokens_V = self.n_prompt_tokens_V

        # Other Model-related attributes
        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype 
        self.sigma = cfg["sigma"]
        
        # Language Linear Layer
        self.linear_L = None
        if self.n_prompt_tokens_L > 0 and self.intrinsic_dim_L > 0:
            self.linear_L = torch.nn.Linear(self.intrinsic_dim_L, self.n_prompt_tokens_L * self.ctx_dim_L,
                                          bias=False,device=self.device,dtype=self.dtype)
            embedding = self.model.token_embedding.weight.cpu()
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)
            logger.info('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear_L.parameters():
                torch.nn.init.normal_(p, mu, std)
        else:
            logger.info("Text prompt tuning disabled (n_prompt_tokens_L or intrinsic_dim_L is 0).")

        self.linear_V = None
        if self.n_prompt_tokens_V > 0 and self.intrinsic_dim_V > 0:
            self.linear_V = torch.nn.Linear(self.intrinsic_dim_V, self.n_prompt_tokens_V * self.ctx_dim_V,
                                            bias=False, device=self.device, dtype=self.dtype)
            conv = self.model.visual.conv1.weight.cpu()
            mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(conv.reshape(-1).detach().cpu().numpy())
            mu = mu_hat*3072/self.intrinsic_dim_V
            std = std_hat * np.sqrt(3072/self.intrinsic_dim_V) * self.sigma
            logger.info('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear_V.parameters():
                torch.nn.init.normal_(p, mu, std)
        else:
            logger.info("Visual prompt tuning disabled (n_prompt_tokens_V or intrinsic_dim_V is 0).")
        
        # Now, it is safe to load the dataset, as it has access to all required model components.
        self.load_dataset()
        self._capture_training_dataset() # Capture the dataset for saving

        # Final setup after model and data are ready
        self.maximize_loss = cfg.get("maximize_loss", False)
        self.best_objective_loss_value = None
        if self.maximize_loss:
            self.best_objective_loss_value = -float('inf')
            logger.info(f"--- Prompt Optimization Mode: MAXIMIZE Loss (Targeting value: {self.best_objective_loss_value}) ---")
        else:
            self.best_objective_loss_value = float('inf')
            logger.info(f"--- Prompt Optimization Mode: MINIMIZE Loss (Targeting value: {self.best_objective_loss_value}) ---")

        logger.info("--- Standard (Clean) Prompt Optimization ---")
        
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0.0
        self.best_accuracy_attack = 0.0 
        self.best_train_accuracy = 0.0

    # --- NEW: Attack Dispatcher ---
    def perform_attack_on_batch(self, images, labels, attack_params):
        """Public-facing dispatcher to run the configured attack."""
        attack_type = attack_params.get("type", "pgd")
        if attack_type == "pgd":
            return self._perform_pgd_attack(images, labels, 
                                            eps=attack_params["eps"], 
                                            alpha=attack_params["alpha"], 
                                            steps=attack_params["steps"])
        elif attack_type == "fgsm":
            return self._perform_fgsm_attack(images, labels, 
                                             eps=attack_params["eps"])
        elif attack_type == "cw":
            return self._perform_cw_attack(images, labels, 
                                           c=attack_params["c"], 
                                           lr=attack_params["lr"], 
                                           steps=attack_params["steps"])
        else:
            logger.warning(f"Unknown attack type '{attack_type}'. Returning original images.")
            return images

    @torch.enable_grad()
    def _perform_fgsm_attack(self, images, labels, eps):
        """ Performs FGSM attack on a batch of images. """
        images_orig = images.clone().detach()
        images_adv = images.clone().detach().requires_grad_(True)
        
        vanilla_text_features = self.get_original_text_features()

        # Single gradient step
        original_parallel = self.image_encoder.parallel
        self.image_encoder.parallel = False
        image_features = self.image_encoder(images_adv.to(self.dtype), prompt=None)
        self.image_encoder.parallel = original_parallel
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ vanilla_text_features.t()
        loss = F.cross_entropy(logits, labels)
        
        self.model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = images_adv.grad.sign()
            images_adv.data = images_adv.data + eps * grad
            # Project back to eps-ball (equivalent to clamp for single step)
            delta = torch.clamp(images_adv.data - images_orig, min=-eps, max=eps)
            images_adv.data = torch.clamp(images_orig + delta, min=0, max=1)
        
        return images_adv.detach()

    @torch.enable_grad()
    def _perform_cw_attack(self, images, labels, c, lr, steps):
        """ Performs Carlini & Wagner L2 attack on a batch of images. """
        images_orig = images.clone().detach()
        
        # Transform images to arctanh space for unconstrained optimization
        w = torch.atanh((images_orig * 2) - 1).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)
        vanilla_text_features = self.get_original_text_features()

        for _ in range(steps):
            # Transform back to image space [0, 1]
            images_adv = 0.5 * (torch.tanh(w) + 1)
            
            # Forward pass
            original_parallel = self.image_encoder.parallel
            self.image_encoder.parallel = False
            image_features = self.image_encoder(images_adv.to(self.dtype), prompt=None)
            self.image_encoder.parallel = original_parallel

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ vanilla_text_features.t()

            # CW loss calculation (untargeted)
            true_class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            
            # Find max logit of other classes
            other_logits = logits.clone()
            other_logits.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_other_logits = other_logits.max(1)[0]
            
            # Loss to encourage misclassification
            # We want to minimize (true_class_logit - max_other_logit)
            # The clamp ensures we only penalize when the prediction is correct or not confidently incorrect
            class_loss = torch.clamp(true_class_logits - max_other_logits, min=-c).sum()
            
            # L2 distortion loss
            dist_loss = torch.sum((images_adv - images_orig) ** 2)
            
            # Total loss
            loss = dist_loss + c * class_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final adversarial image
        images_adv_final = (0.5 * (torch.tanh(w) + 1)).detach()
        return torch.clamp(images_adv_final, min=0, max=1)


    @torch.enable_grad()
    def _perform_pgd_attack(self, images, labels, eps, alpha, steps):
        """ Performs PGD attack on a batch of images. """
        images_orig = images.clone().detach()
        images_adv = images.clone().detach().requires_grad_(True)
        
        vanilla_text_features = self.get_original_text_features()

        for _ in range(steps):
            original_parallel = self.image_encoder.parallel
            self.image_encoder.parallel = False
            image_features = self.image_encoder(images_adv.to(self.dtype), prompt=None)
            self.image_encoder.parallel = original_parallel
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ vanilla_text_features.t()
            loss = F.cross_entropy(logits, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad = images_adv.grad.sign()
                images_adv.data = images_adv.data + alpha * grad
                delta = torch.clamp(images_adv.data - images_orig, min=-eps, max=eps)
                images_adv.data = torch.clamp(images_orig + delta, min=0, max=1)
            
            images_adv.grad.zero_()
        
        return images_adv.detach()
    
    # ... (rest of the class methods: _capture_training_dataset, get_text_information, etc. remain the same) ...
    # ... (omitting for brevity, no changes needed in eval, test, etc.) ...

    def _capture_training_dataset(self):
        """
        Iterates through the training dataloader to capture all images and labels.
        This allows the exact k-shot dataset to be saved for full reproducibility.
        """
        logger.info("Capturing the k-shot training dataset for saving.")
        all_images = []
        all_labels = []
        
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            logger.warning("train_loader not available. Cannot capture dataset snapshot.")
            return

        # Temporarily set parallel to False to avoid data duplication from parse_batch logic
        original_parallel_flag = self.parallel
        self.parallel = False
        
        for batch in self.train_loader:
            # We don't use self.parse_batch here to get the raw, un-repeated data
            images = batch["image"] # These should be preprocessed tensors
            labels = batch["label"]
            all_images.append(images.cpu()) # Move to CPU for storage
            all_labels.append(labels.cpu())

        self.parallel = original_parallel_flag # Restore original flag

        if not all_images:
            logger.warning("Training dataset snapshot is empty.")
            self._training_dataset_snapshot = None
            return

        try:
            # Concatenate all batches into single tensors
            images_tensor = torch.cat(all_images, dim=0)
            labels_tensor = torch.cat(all_labels, dim=0)
            self._training_dataset_snapshot = {
                'images': images_tensor,
                'labels': labels_tensor
            }
            logger.info(f"Captured training dataset snapshot with {images_tensor.shape[0]} samples.")
        except Exception as e:
            logger.error(f"Failed to create training dataset snapshot: {e}")
            self._training_dataset_snapshot = None


    def get_text_information(self,caption=None):
        prompt_prefix_placeholder = " ".join(["X"] * self.n_prompt_tokens_L)

        if caption is None:
            classnames = [name.replace("_", " ").replace("-", " ") for name in self.classes]
            pattern_prompts = []

            for name in classnames:
                initial_prompt = self.initial_prompt_text if self.initial_prompt_text else ""
                
                # Build the prompt string based on the learned prompt's position
                if self.learned_prompt_pos == "prefix":
                    # [Learned] [Initial] [Class]
                    template = f"{prompt_prefix_placeholder} {initial_prompt} {name}."
                elif self.learned_prompt_pos == "middle":
                    # [Initial] [Learned] [Class]
                    template = f"{initial_prompt} {prompt_prefix_placeholder} {name}."
                elif self.learned_prompt_pos == "suffix":
                    # [Initial] [Class] [Learned]
                    template = f"{initial_prompt} {name} {prompt_prefix_placeholder}."
                else: # Default to prefix
                    template = f"{prompt_prefix_placeholder} {initial_prompt} {name}."

                # Clean up extra spaces that might result from an empty initial_prompt
                pattern_prompts.append(" ".join(template.split()))

            tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            
            # Find the start index of the context tokens (the "X"s)
            # This is crucial for the generalized `incorporate_prompt`
            x_token_id = clip.tokenize("X")[0, 1].item() # The token id for a single "X"
            # Find the first column where an "X" token appears. This is our start index.
            ctx_start_idx = (tokenized_pattern_prompts == x_token_id).nonzero(as_tuple=True)[1].min().item()

            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            
            context = {
                "n_cls": self.n_cls, 
                "n_prompt_tokens_L": self.n_prompt_tokens_L,
                "init_pattern_embedding": init_pattern_embedding, 
                "tokenized_pattern_prompts": tokenized_pattern_prompts,
                "ctx_start_idx": ctx_start_idx,  # Pass the start index to the encoder
                "batch_size": self.batch_size,
                "pop_size": self.popsize,
                "parallel": self.parallel
            }
        else: # Logic for a single caption (e.g., for other tasks), kept simpler
            pattern_prompt = prompt_prefix_placeholder + " " + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            ctx_start_idx = 1 # Assuming it's always at the start for this simple case
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {
                "n_cls": 1,
                "n_prompt_tokens_L": self.n_prompt_tokens_L,
                "init_pattern_embedding": init_pattern_embedding, 
                "tokenized_pattern_prompts": tokenized_pattern_prompts,
                "ctx_start_idx": ctx_start_idx,
                "batch_size": self.batch_size,
                "pop_size": self.popsize,
                "parallel": self.parallel
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
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context

    def _add_noise(self, tensor, noise_type):
        if noise_type == "none" or self.noise_level == 0:
            return tensor
        
        if noise_type == "gaussian":
            noise = torch.randn_like(tensor) * self.noise_level
        elif noise_type == "uniform":
            # Creates noise in [-noise_level, +noise_level]
            noise = (torch.rand_like(tensor) * 2 - 1) * self.noise_level
        elif noise_type == "binomial":
            # Bernoulli {-1, 1} noise, scaled by level
            noise = (torch.bernoulli(torch.full_like(tensor, 0.5)) * 2 - 1) * self.noise_level
        else:
            return tensor
            
        return tensor + noise.to(self.device)

    def generate_text_prompts(self,intrinsic_vectors):
        prompt_list = []
        if self.linear_L is None:
            return [None] * len(intrinsic_vectors)
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, self.ctx_dim_L)
            
            # --- NEW: Add noise ---
            z = self._add_noise(z, self.noise_type_text)

            if self.init_prompt is not None:
                z = z + self.init_prompt

            prompt_list.append(z)
        return prompt_list

    def generate_visual_prompts(self,intrinsic_vectors):
        visual_prompt_list = []
        if self.linear_V is None:
            return [None] * len(intrinsic_vectors)
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V, self.ctx_dim_V)

            # --- NEW: Add noise ---
            z = self._add_noise(z, self.noise_type_visual)

            visual_prompt_list.append(z)
        return visual_prompt_list

    def metric(self,logits,label):
        ce_loss = F.cross_entropy(logits, label, reduction='none')
        final_loss = 0
        if self.loss_type == "ce":
            final_loss = torch.sum(ce_loss)
        elif self.loss_type == "focal":
            gamma = 2
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            final_loss = torch.sum(focal_loss)
        return final_loss

    @torch.no_grad()
    def eval(self, prompt_zip):
        prompt_text_list_or_tensor, prompt_image_list_or_tensor = prompt_zip[0], prompt_zip[1]
        self.num_call += 1 # num_call increments per fitness evaluation
        
        loss_accumulator = 0
        logit_scale = self.logit_scale.exp()

        if self.parallel: # optimizer is evaluating a whole population
            loss_accumulator = [0.0] * self.popsize
            all_pop_text_features = []
            # prompt_text_list_or_tensor is a list of tensors or a list of Nones
            if prompt_text_list_or_tensor[0] is not None:
                for p_text in prompt_text_list_or_tensor:
                    features = self.text_encoder(p_text)
                    features = features / features.norm(dim=-1, keepdim=True)
                    all_pop_text_features.append(features)
                pop_txt_features = torch.stack(all_pop_text_features) # [pop_size, n_cls, D]
            else:
                # If text prompts are disabled, get base text features once and repeat
                base_text_features = self.text_encoder(None)
                base_text_features = base_text_features / base_text_features.norm(dim=-1, keepdim=True)
                pop_txt_features = base_text_features.unsqueeze(0).repeat(self.popsize, 1, 1)

        else: # optimizer is evaluating a single candidate
            # Pass either the tensor or None to the encoder
            text_features = self.text_encoder(prompt_text_list_or_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        for batch_idx, batch in enumerate(self.train_loader):
            # `clean_image_orig` is [B_orig, C, H, W], `label_orig` is [B_orig]
            clean_image, label = self.parse_batch(batch)
            clean_image = clean_image.to(self.dtype)

            if self.parallel: # Population evaluation
                B_actual_batch = label.shape[0] # This is B_orig
                
                pop_clean_image_batch = clean_image.view(self.popsize, B_actual_batch, *clean_image.shape[1:])

                for i in range(self.popsize):
                    image_features_i = self.image_encoder(pop_clean_image_batch[i], prompt_image_list_or_tensor[i])
                    image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
                    tmp_logits = logit_scale * image_features_i @ pop_txt_features[i].t()
                    loss_for_member_batch_i = self.metric(tmp_logits, label)
                    loss_accumulator[i] += loss_for_member_batch_i.item()

            else: # Not parallel (single candidate evaluation)
                image_features = self.image_encoder(clean_image, prompt_image_list_or_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()
                loss_for_candidate_batch = self.metric(logits, label)
                loss_accumulator += loss_for_candidate_batch.item()


        if self.parallel:
            loss_values_final = [x / len(self.train_data) for x in loss_accumulator]
        else:
            loss_values_final = loss_accumulator / len(self.train_data)

        epoch_best_loss_in_eval = None
        prompt_candidate_text = None
        prompt_candidate_image = None

        if self.parallel:
            if self.maximize_loss:
                epoch_best_loss_in_eval = max(loss_values_final)
                best_idx_in_eval = loss_values_final.index(epoch_best_loss_in_eval)
            else:
                epoch_best_loss_in_eval = min(loss_values_final)
                best_idx_in_eval = loss_values_final.index(epoch_best_loss_in_eval)

            prompt_candidate_text = prompt_text_list_or_tensor[best_idx_in_eval]
            prompt_candidate_image = prompt_image_list_or_tensor[best_idx_in_eval]
        else:
            epoch_best_loss_in_eval = loss_values_final
            prompt_candidate_text = prompt_text_list_or_tensor
            prompt_candidate_image = prompt_image_list_or_tensor

        if self.parallel:
            self.loss.append([l for l in loss_values_final])
        else:
            self.loss.append(loss_values_final)

        update_overall_best = False
        if self.maximize_loss:
            if epoch_best_loss_in_eval > self.best_objective_loss_value:
                update_overall_best = True
        else:
            if epoch_best_loss_in_eval < self.best_objective_loss_value:
                update_overall_best = True

        if update_overall_best:
            self.best_objective_loss_value = epoch_best_loss_in_eval
            if prompt_candidate_text is not None and prompt_candidate_image is not None:
                self.best_prompt_text = prompt_candidate_text.detach().clone()
                self.best_prompt_image = prompt_candidate_image.detach().clone()

            objective_type_str = "maximized" if self.maximize_loss else "minimized"
            logger.info(f"*** New best {objective_type_str} (clean eval) loss found: {self.best_objective_loss_value:.4f} (at call {self.num_call}) ***")

        # --- Intermediate Testing Block (conditional) ---
        if self.test_every_gens is not None and self.test_every_gens > 0:
            trigger_interval = self.test_every_gens if self.parallel else self.test_every_gens * self.popsize
            
            if self.num_call > 0 and (self.num_call % trigger_interval == 0):
                current_generation = self.num_call if self.parallel else self.num_call // self.popsize
                
                # Calculate train accuracy with the current best prompts
                acc_train_current = self.test_on_train_set()
                self.train_acc.append(acc_train_current.item())
                self.best_train_accuracy = acc_train_current.item()

                eval_loss_type_str = "clean"
                obj_str = "maximize" if self.maximize_loss else "minimize"

                logger.info(f"\n--- Intermediate Test at Generation ~{current_generation} (Prompts from {eval_loss_type_str} eval, obj: {obj_str} loss) ---")
                
                # --- MODIFIED: Log both attacked and clean test accuracy if applicable ---
                acc_test = self.test() # This uses the primary test_loader (attacked or clean)
                self.acc.append(acc_test.item())
                self.best_accuracy = max(acc_test.item(), self.best_accuracy)
                
                test_set_type = "Attacked" if self.cfg.get('attack_test') else "Clean"
                logger.info(f"Train Accuracy: {self.best_train_accuracy:.4f}")
                logger.info(f"Test {test_set_type} Accuracy: {acc_test:.4f} (Best Test {test_set_type}: {self.best_accuracy:.4f})")
                
                if self.test_loader_clean is not None:
                    acc_clean_baseline = self.test(use_clean_loader=True)
                    self.acc_clean_during_attack_run.append(acc_clean_baseline.item())
                    logger.info(f"Test Clean (Baseline) Accuracy: {acc_clean_baseline:.4f}")
                # --- END MODIFIED ---

                output_dir = os.path.join(self.output_dir,self.task_name)
                
                initial_prompt_str_fn = f"_initPrompt" if self.initial_prompt_text is not None else ""
                learned_pos_str_fn = f"_pos{self.learned_prompt_pos}"
                
                noise_str_fn = ""
                if self.noise_type_text != 'none' or self.noise_type_visual != 'none':
                    noise_str_fn = f"_noiseT_{self.noise_type_text}_noiseV_{self.noise_type_visual}_level_{self.noise_level}"

                fname = "{}{}{}_{}_{}_parallel{}_maxLoss{}{}.pth".format(
                    self.k_shot, self.task_name, initial_prompt_str_fn, learned_pos_str_fn,
                    self.opt_name, self.backbone.replace("/", "-"),
                    self.parallel,
                    self.maximize_loss,
                    noise_str_fn
                )

                content = {
                    "task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                    "best_accuracy":self.best_accuracy, "acc":self.acc,
                    "acc_clean_during_attack_run": self.acc_clean_during_attack_run,
                    "best_train_accuracy": self.best_train_accuracy, "train_acc": self.train_acc,
                    "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                    "training_dataset_snapshot": self._training_dataset_snapshot,
                    "historical_losses":self.loss,
                    "best_objective_loss_value": self.best_objective_loss_value,
                    "maximize_loss_setting": self.maximize_loss,
                    "num_call":self.num_call,
                    "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict()
                }
                Analysis_Util.save_results(content,output_dir,fname)

        if self.parallel:
            return_value = [l * -1 if self.maximize_loss else l for l in loss_values_final]
        else:
            return_value = loss_values_final * -1 if self.maximize_loss else loss_values_final
        return return_value

    @torch.no_grad()
    def test_on_train_set(self):
        if self.best_prompt_text is None or self.best_prompt_image is None:
            logger.warning("Train accuracy test skipped: no best tuned prompt available.")
            return torch.tensor(0.0)

        correct = 0.
        total = 0.

        original_text_encoder_parallel = self.text_encoder.parallel
        original_image_encoder_parallel = self.image_encoder.parallel
        self.text_encoder.parallel = False
        self.image_encoder.parallel = False

        text_features = self.text_encoder(self.best_prompt_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_prompt = self.best_prompt_image

        for batch in self.train_loader:
            temp_original_class_parallel_attr = self.parallel
            self.parallel = False
            image, label = self.parse_batch(batch)
            self.parallel = temp_original_class_parallel_attr

            total += image.size(0)
            image = image.to(self.dtype)

            image_features = self.image_encoder(image, image_prompt)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        self.text_encoder.parallel = original_text_encoder_parallel
        self.image_encoder.parallel = original_image_encoder_parallel

        acc = correct / total
        return acc

    @torch.no_grad()
    def test(self, use_clean_loader=False):
        if self.best_prompt_text is None and self.n_prompt_tokens_L > 0:
             logger.warning("Test skipped for text prompts: no best tuned prompt available for evaluation.")
             # return torch.tensor(0.0)
        if self.best_prompt_image is None and self.n_prompt_tokens_V > 0:
            logger.warning("Test skipped for visual prompts: no best tuned prompt available for evaluation.")
            # return torch.tensor(0.0)
        if self.best_prompt_text is None or self.best_prompt_image is None:
            logger.warning("Test skipped: no best tuned prompt available for evaluation.")
            return torch.tensor(0.0)

        # --- NEW: Select the appropriate data loader ---
        if use_clean_loader and self.test_loader_clean:
            active_loader = self.test_loader_clean
        else:
            active_loader = self.test_loader
        # --- END NEW ---

        correct = 0.
        total = 0.
        
        # Store and temporarily override parallel flags for encoders during test
        original_text_encoder_parallel = self.text_encoder.parallel
        original_image_encoder_parallel = self.image_encoder.parallel
        self.text_encoder.parallel = False
        self.image_encoder.parallel = False

        # Use the best tuned prompts for all tests
        current_text_features_for_test = self.text_encoder(self.best_prompt_text)
        current_text_features_for_test = current_text_features_for_test / current_text_features_for_test.norm(dim=-1,keepdim=True)
        current_image_prompt_for_test = self.best_prompt_image

        for batch in active_loader: # Use the selected loader
            temp_original_class_parallel_attr = self.parallel
            self.parallel = False # Affects parse_batch's internal logic
            image,label = self.parse_batch(batch) # image [B,C,H,W], label [B]
            self.parallel = temp_original_class_parallel_attr # Restore

            total += image.size(0)

            eval_image = image.to(self.dtype)
            
            image_features = self.image_encoder(eval_image, current_image_prompt_for_test)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale*image_features@current_text_features_for_test.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        self.text_encoder.parallel = original_text_encoder_parallel
        self.image_encoder.parallel = original_image_encoder_parallel
        
        acc = correct/total
        return acc

    def load_dataset(self):
        # --- MODIFIED: Set up comprehensive attack configs ---
        train_attack_cfg = None
        if self.cfg.get("use_attacked_dataset", False) and self.cfg.get("attack_train", False):
            train_attack_cfg = {
                "model": self,
                "type": self.cfg.get("attack_type_train", "pgd"),
                "ratio": self.cfg.get("attack_train_ratio", 0.5),
                # PGD params
                "eps": self.cfg.get("pgd_eps_train", 8/255.0),
                "alpha": self.cfg.get("pgd_alpha_train", 2/255.0),
                "steps": self.cfg.get("pgd_steps_train", 10),
                # CW params (shared for simplicity, can be split if needed)
                "c": self.cfg.get("cw_c", 1.0),
                "lr": self.cfg.get("cw_lr", 0.01),
                # Note: CW uses 'steps' from PGD for iteration count
            }
            train_attack_cfg["steps"] = self.cfg.get("cw_steps", 20) if train_attack_cfg["type"] == "cw" else train_attack_cfg["steps"]
            
        test_attack_cfg = None
        if self.cfg.get("use_attacked_dataset", False) and self.cfg.get("attack_test", False):
            test_attack_cfg = {
                "model": self,
                "type": self.cfg.get("attack_type_test", "pgd"),
                "ratio": self.cfg.get("attack_test_ratio", 0.5),
                # PGD params
                "eps": self.cfg.get("pgd_eps_test", 8/255.0),
                "alpha": self.cfg.get("pgd_alpha_test", 2/255.0),
                "steps": self.cfg.get("pgd_steps_test", 20),
                # CW params
                "c": self.cfg.get("cw_c", 1.0),
                "lr": self.cfg.get("cw_lr", 0.01),
            }
            test_attack_cfg["steps"] = self.cfg.get("cw_steps", 20) if test_attack_cfg["type"] == "cw" else test_attack_cfg["steps"]

        if test_attack_cfg is not None:
            logger.info("Attack on test set is enabled. Also loading a clean test set for baseline comparison.")
            if self.task_name == 'CIFAR10':
                _, self.test_loader_clean = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess, root=self.data_dir, attack_config=None)
            elif self.task_name in ['StanfordCars', 'OxfordPets', 'UCF-101', 'DTD', 'EuroSAT', 'Food101', 'caltech101', 'SUN397', 'ImageNet']:
                task_config = self.cfg.get(self.task_name, {})
                dataset_dir = task_config.get('dataset_dir', self.task_name + "_Gen")
                _, self.test_loader_clean = load_test(batch_size=self.batch_size, preprocess=self.preprocess,
                                                            root=self.data_dir, dataset_dir=dataset_dir, attack_config=None)
            else: # CIFAR100 and other unhandled cases
                _, self.test_loader_clean = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)

        # --- DATASET LOADING (no changes needed below this line) ---
        if self.task_name == 'CIFAR100':
            self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed, attack_config=train_attack_cfg)
            self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess, attack_config=test_attack_cfg)
        elif self.task_name == 'CIFAR10':
            self.dataset = CIFAR10(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar10(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed, root=self.data_dir, attack_config=train_attack_cfg)
            self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess, root=self.data_dir, attack_config=test_attack_cfg)
        elif self.task_name == 'StanfordCars':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen", attack_config=train_attack_cfg)
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen", attack_config=test_attack_cfg)
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        # ... and so on for all other datasets ...
        # (The rest of the dataset loading logic is repetitive and doesn't need to change)
        elif self.task_name == 'ImageNet':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet", attack_config=train_attack_cfg)
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet", attack_config=test_attack_cfg)
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)


    def parse_batch(self,batch):
        image = batch["image"]
        label = batch["label"]
        image = image.to(device=self.device, dtype=self.dtype if image.dtype != torch.uint8 else torch.float32)
        if image.dtype == torch.uint8: # common for PIL loaded images
            image = image.float() / 255.0 # Normalize if uint8
        
        label = label.to(device=self.device)
        
        if self.parallel: 
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label