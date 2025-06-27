import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchvision.datasets import CIFAR100, CIFAR10
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from dataset.cifar10 import load_train_cifar10, load_test_cifar10
from dataset.clip_cifar10_pgd import load_train_cifar10_pgd, load_test_cifar10_pgd
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
        self.initial_prompt_text = cfg.get("initial_prompt_text", None)
        self.learned_prompt_pos = cfg.get("learned_prompt_pos", "prefix")
        self.test_every_gens = cfg.get("test_every_n_gens", None) # <-- NEW
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.backbone,device=self.device)
        self.loss = []
        self.acc = []
        self.acc_attack = []
        self.train_acc = []
        self.test_attack_config = cfg.get("test_attack", {"enabled": False})
        self.adv_train_config = cfg.get("adv_train", {"enabled": False})
        self.adv_train_attack_prompt_type = self.adv_train_config.get("attack_prompt_type", "on-the-fly")
        self.adv_train_attack_type = self.adv_train_config.get("attack_type", "pgd")
        self._training_dataset_snapshot = None # Holder for the dataset

        self.load_dataset()
        self._capture_training_dataset() # Capture the dataset for saving

        self.maximize_loss = cfg.get("maximize_loss", False)
        self.best_objective_loss_value = None
        if self.maximize_loss:
            self.best_objective_loss_value = -float('inf')
            logger.info(f"--- Prompt Optimization Mode: MAXIMIZE Loss (Targeting value: {self.best_objective_loss_value}) ---")
        else:
            self.best_objective_loss_value = float('inf')
            logger.info(f"--- Prompt Optimization Mode: MINIMIZE Loss (Targeting value: {self.best_objective_loss_value}) ---")

        if self.adv_train_config["enabled"]:
            logger.info("--- Adversarial Prompt Optimization ENABLED ---")
            logger.info(f"  Training Attack Type: {self.adv_train_attack_type.upper()}")
            logger.info(f"  Training Attack Config: \
                        Epsilon={self.adv_train_config['epsilon']}, \
                        Alpha={self.adv_train_config['alpha']}, \
                        Iter={self.adv_train_config['num_iter']}")
            logger.info(f"  Adversarial Attack Prompt Type for Training: {self.adv_train_attack_prompt_type}")
            logger.info(f"  Adversarial Training Sample Ratio: {self.adv_train_config.get('sample_ratio', 1.0)}")
            if not self.adv_train_config.get('all_call', False) and self.test_every_gens is not None:
                logger.info(f"  Periodic adversarial tuning will occur every {self.test_every_gens} generations.")
            elif self.adv_train_config.get('all_call', False):
                logger.info("  Adversarial tuning will occur for the whole progress.")
        else:
            logger.info("--- Standard (Clean) Prompt Optimization ---")
        
        if self.test_attack_config["enabled"] or self.adv_train_config["enabled"]:
            logger.info("Adversarial Operations (Test or Train) ENABLED.")
            mean = self.preprocess.transforms[-1].mean
            std = self.preprocess.transforms[-1].std
            self.norm_mean = torch.tensor(mean).to(self.device).view(3, 1, 1)
            self.norm_std = torch.tensor(std).to(self.device).view(3, 1, 1)
            self.norm_upper_limit = ((1 - self.norm_mean) / self.norm_std).to(self.device)
            self.norm_lower_limit = ((0 - self.norm_mean) / self.norm_std).to(self.device)
            if self.test_attack_config["enabled"]:
                attack_type = self.test_attack_config.get('attack_type', 'N/A').upper()
                logger.info(f"  Test Attack Type: {attack_type}")
                logger.info(f"  Test Attack Config: \
                            Epsilon={self.test_attack_config.get('epsilon', 'N/A')},\
                            Alpha={self.test_attack_config.get('alpha', 'N/A')},\
                            Iter={self.test_attack_config.get('num_iter', 'N/A')}")
        else:
            logger.info("Adversarial Testing DISABLED.")


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

        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0.0
        self.best_accuracy_attack = 0.0
        self.best_train_accuracy = 0.0
        self.sigma = cfg["sigma"]
        # Language Linear Layer
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
        # Vision Linear Layer
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

    def generate_text_prompts(self,intrinsic_vectors):
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, self.ctx_dim_L)
            if self.init_prompt is not None:
                z = z + self.init_prompt

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
        self.num_call += 1 # num_call increments per fitness evaluation (per population member if sequential, or per population if parallel)
        
        is_current_eval_adversarial = False
        if self.adv_train_config["enabled"]:
            if self.adv_train_config.get("all_call", False):
                is_current_eval_adversarial = True
            elif self.test_every_gens is not None and self.test_every_gens > 0:
                # Synchronize periodic adv training with periodic testing
                trigger_interval = self.test_every_gens if self.parallel else self.test_every_gens * self.popsize
                if self.num_call > 0 and (self.num_call % trigger_interval == 0):
                    is_current_eval_adversarial = True


        loss_accumulator = 0
        logit_scale = self.logit_scale.exp()

        current_prompt_text_for_eval_single = None
        current_prompt_image_for_eval_single = None

        # For "constant" or "perturbed" during adv_train, common across batch/pop
        text_features_for_attack_generation = None
        # For "perturbed" specifically
        text_prompt_for_attack_generation_perturbed = None

        if self.parallel: # optimizer is evaluating a whole population
            loss_accumulator = [0.0] * self.popsize
            all_pop_text_features = []
            for p_text in prompt_text_list_or_tensor:
                features = self.text_encoder(p_text)
                features = features / features.norm(dim=-1, keepdim=True)
                all_pop_text_features.append(features)
            pop_txt_features = torch.stack(all_pop_text_features) # [pop_size, n_cls, D]

            if is_current_eval_adversarial and self.adv_train_attack_prompt_type == "constant":
                text_features_for_attack_generation = self.get_original_text_features()
        else: # optimizer is evaluating a single candidate
            text_features = self.text_encoder(prompt_text_list_or_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            current_prompt_text_for_eval_single = prompt_text_list_or_tensor
            current_prompt_image_for_eval_single = prompt_image_list_or_tensor

            if is_current_eval_adversarial:
                if self.adv_train_attack_prompt_type == "constant":
                    text_features_for_attack_generation = self.get_original_text_features()
                elif self.adv_train_attack_prompt_type == "on-the-fly":
                    text_features_for_attack_generation = text_features # Use current candidate's features
                elif self.adv_train_attack_prompt_type == "perturbed":
                    text_features_for_attack_generation = text_features # Start with current candidate's features
                    text_prompt_for_attack_generation_perturbed = current_prompt_text_for_eval_single.clone().detach()


        for batch_idx, batch in enumerate(self.train_loader):
            # `clean_image_orig` is [B_orig, C, H, W], `label_orig` is [B_orig]
            clean_image_orig, label_orig = self.parse_batch(batch)

            if self.parallel: # Population evaluation
                B_actual_batch = label_orig.shape[0] # This is B_orig
                
                # pop_clean_image_batch will be [popsize, B_orig, C, H, W]
                pop_clean_image_batch = clean_image_orig.view(self.popsize, B_actual_batch, *clean_image_orig.shape[1:])

                for i in range(self.popsize):
                    current_clean_images_for_member = pop_clean_image_batch[i] # [B_orig, C, H, W]
                    current_labels_for_member = label_orig # [B_orig]
                    current_txt_features_for_loss = pop_txt_features[i]
                    current_img_prompt_for_loss = prompt_image_list_or_tensor[i]
                    
                    loss_for_member_batch_i = 0

                    if is_current_eval_adversarial:
                        adv_sample_ratio = self.adv_train_config.get('sample_ratio', 1.0)
                        num_total_samples_member = current_clean_images_for_member.size(0)

                        # Determine text features/prompt for PGD attack generation for this member
                        pgd_text_features_guidance = current_txt_features_for_loss # Default to on-the-fly for this pop member
                        pgd_text_prompt_to_perturb = None
                        if self.adv_train_attack_prompt_type == "constant":
                            pgd_text_features_guidance = text_features_for_attack_generation # Shared constant features
                        elif self.adv_train_attack_prompt_type == "perturbed":
                            pgd_text_prompt_to_perturb = prompt_text_list_or_tensor[i].clone().detach()

                        if adv_sample_ratio < 1.0 and num_total_samples_member > 0:
                            adv_images_to_perturb, adv_labels, clean_images_part, clean_labels_part = \
                                self._split_batch_by_class_for_adv_train(current_clean_images_for_member, current_labels_for_member, adv_sample_ratio)
                            
                            num_adv_samples = adv_images_to_perturb.size(0)
                            num_clean_samples = clean_images_part.size(0)
                            perturbed_text_prompt_from_attack = None
                            if num_adv_samples > 0:
                                with torch.enable_grad():
                                    adv_images_perturbed, perturbed_text_prompt_from_attack = self._run_adversarial_attack(
                                        images=adv_images_to_perturb, labels=adv_labels,
                                        text_features_for_attack=pgd_text_features_guidance,
                                        image_prompt=current_img_prompt_for_loss, config=self.adv_train_config,
                                        attack_type=self.adv_train_attack_type,
                                        text_prompt_to_perturb=pgd_text_prompt_to_perturb
                                    )
                                adv_images_perturbed = adv_images_perturbed.to(self.dtype)
                                
                                text_features_for_adv_loss = current_txt_features_for_loss
                                if self.adv_train_attack_prompt_type == "perturbed" and perturbed_text_prompt_from_attack is not None:
                                    text_features_for_adv_loss = self.text_encoder(perturbed_text_prompt_from_attack)
                                    text_features_for_adv_loss = text_features_for_adv_loss / text_features_for_adv_loss.norm(dim=-1, keepdim=True)

                                adv_image_features = self.image_encoder(adv_images_perturbed, current_img_prompt_for_loss)
                                adv_image_features = adv_image_features / adv_image_features.norm(dim=-1, keepdim=True)
                                adv_logits = logit_scale * adv_image_features @ text_features_for_adv_loss.t()
                                loss_for_member_batch_i += self.metric(adv_logits, adv_labels)

                            if num_clean_samples > 0:
                                clean_images_part_typed = clean_images_part.to(self.dtype)
                                clean_image_features = self.image_encoder(clean_images_part_typed, current_img_prompt_for_loss)
                                clean_image_features = clean_image_features / clean_image_features.norm(dim=-1, keepdim=True)
                                # Clean part uses original text features of the current member
                                clean_logits = logit_scale * clean_image_features @ current_txt_features_for_loss.t()
                                loss_for_member_batch_i += self.metric(clean_logits, clean_labels_part)
                        else: # Full adversarial batch for this member
                            with torch.enable_grad():
                                eval_image_i, perturbed_text_prompt_from_attack = self._run_adversarial_attack(
                                    images=current_clean_images_for_member, labels=current_labels_for_member,
                                    text_features_for_attack=pgd_text_features_guidance,
                                    image_prompt=current_img_prompt_for_loss, config=self.adv_train_config,
                                    attack_type=self.adv_train_attack_type,
                                    text_prompt_to_perturb=pgd_text_prompt_to_perturb
                                )
                            eval_image_i = eval_image_i.to(self.dtype)
                            
                            text_features_for_loss_i = current_txt_features_for_loss
                            if self.adv_train_attack_prompt_type == "perturbed" and perturbed_text_prompt_from_attack is not None:
                                text_features_for_loss_i = self.text_encoder(perturbed_text_prompt_from_attack)
                                text_features_for_loss_i = text_features_for_loss_i / text_features_for_loss_i.norm(dim=-1, keepdim=True)

                            image_features_i = self.image_encoder(eval_image_i, current_img_prompt_for_loss)
                            image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
                            tmp_logits = logit_scale * image_features_i @ text_features_for_loss_i.t()
                            loss_for_member_batch_i += self.metric(tmp_logits, current_labels_for_member)
                    else: # Clean evaluation for this member
                        eval_image_i = current_clean_images_for_member.to(self.dtype)
                        image_features_i = self.image_encoder(eval_image_i, current_img_prompt_for_loss)
                        image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
                        tmp_logits = logit_scale * image_features_i @ current_txt_features_for_loss.t()
                        loss_for_member_batch_i += self.metric(tmp_logits, current_labels_for_member)
                    
                    loss_accumulator[i] += loss_for_member_batch_i.item() if isinstance(loss_for_member_batch_i, torch.Tensor) else loss_for_member_batch_i

            else: # Not parallel (single candidate evaluation)
                current_clean_images = clean_image_orig
                current_labels = label_orig
                current_txt_features_for_loss = text_features
                current_img_prompt_for_loss = current_prompt_image_for_eval_single
                
                loss_for_candidate_batch = 0

                if is_current_eval_adversarial:
                    adv_sample_ratio = self.adv_train_config.get('sample_ratio', 1.0)
                    num_total_samples = current_clean_images.size(0)

                    if adv_sample_ratio < 1.0 and num_total_samples > 0:
                        adv_images_to_perturb, adv_labels, clean_images_part, clean_labels_part = \
                            self._split_batch_by_class_for_adv_train(current_clean_images, current_labels, adv_sample_ratio)

                        num_adv_samples = adv_images_to_perturb.size(0)
                        num_clean_samples = clean_images_part.size(0)
                        perturbed_text_prompt_from_attack = None
                        if num_adv_samples > 0:
                            with torch.enable_grad():
                                adv_images_perturbed, perturbed_text_prompt_from_attack = self._run_adversarial_attack(
                                    images=adv_images_to_perturb, labels=adv_labels,
                                    text_features_for_attack=text_features_for_attack_generation,
                                    image_prompt=current_img_prompt_for_loss, config=self.adv_train_config,
                                    attack_type=self.adv_train_attack_type,
                                    text_prompt_to_perturb=text_prompt_for_attack_generation_perturbed
                                )
                            adv_images_perturbed = adv_images_perturbed.to(self.dtype)

                            text_features_for_adv_loss = current_txt_features_for_loss
                            if self.adv_train_attack_prompt_type == "perturbed" and perturbed_text_prompt_from_attack is not None:
                                text_features_for_adv_loss = self.text_encoder(perturbed_text_prompt_from_attack)
                                text_features_for_adv_loss = text_features_for_adv_loss / text_features_for_adv_loss.norm(dim=-1, keepdim=True)
                            
                            adv_image_features = self.image_encoder(adv_images_perturbed, current_img_prompt_for_loss)
                            adv_image_features = adv_image_features / adv_image_features.norm(dim=-1, keepdim=True)
                            adv_logits = logit_scale * adv_image_features @ text_features_for_adv_loss.t()
                            loss_for_candidate_batch += self.metric(adv_logits, adv_labels)

                        if num_clean_samples > 0:
                            clean_images_part_typed = clean_images_part.to(self.dtype)
                            clean_image_features = self.image_encoder(clean_images_part_typed, current_img_prompt_for_loss)
                            clean_image_features = clean_image_features / clean_image_features.norm(dim=-1, keepdim=True)
                            # Clean part uses original candidate features
                            clean_logits = logit_scale * clean_image_features @ current_txt_features_for_loss.t()
                            loss_for_candidate_batch += self.metric(clean_logits, clean_labels_part)
                    else: # Full adversarial batch for this candidate
                        with torch.enable_grad():
                            eval_image, perturbed_text_prompt_from_attack = self._run_adversarial_attack(
                                images=current_clean_images, labels=current_labels,
                                text_features_for_attack=text_features_for_attack_generation,
                                image_prompt=current_img_prompt_for_loss, config=self.adv_train_config,
                                attack_type=self.adv_train_attack_type,
                                text_prompt_to_perturb=text_prompt_for_attack_generation_perturbed
                            )
                        eval_image = eval_image.to(self.dtype)

                        text_features_for_loss_cand = current_txt_features_for_loss
                        if self.adv_train_attack_prompt_type == "perturbed" and perturbed_text_prompt_from_attack is not None:
                            text_features_for_loss_cand = self.text_encoder(perturbed_text_prompt_from_attack)
                            text_features_for_loss_cand = text_features_for_loss_cand / text_features_for_loss_cand.norm(dim=-1, keepdim=True)
                        
                        image_features = self.image_encoder(eval_image, current_img_prompt_for_loss)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        logits = logit_scale * image_features @ text_features_for_loss_cand.t()
                        loss_for_candidate_batch += self.metric(logits, current_labels)
                else: # Clean evaluation for this candidate
                    eval_image = current_clean_images.to(self.dtype)
                    image_features = self.image_encoder(eval_image, current_img_prompt_for_loss)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits = logit_scale * image_features @ current_txt_features_for_loss.t()
                    loss_for_candidate_batch += self.metric(logits, current_labels)
                
                loss_accumulator += loss_for_candidate_batch.item() if isinstance(loss_for_candidate_batch, torch.Tensor) else loss_for_candidate_batch


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
            prompt_candidate_text = current_prompt_text_for_eval_single
            prompt_candidate_image = current_prompt_image_for_eval_single

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
            adv_status_str = "adversarial" if is_current_eval_adversarial else "clean"
            attack_type_str_info = f" (AttackType: {self.adv_train_attack_type}, AttackGen: {self.adv_train_attack_prompt_type}"
            if is_current_eval_adversarial and self.adv_train_config.get('sample_ratio', 1.0) < 1.0:
                attack_type_str_info += f", SampleRatio: {self.adv_train_config.get('sample_ratio', 1.0)}"
            attack_type_str_info += ")" if is_current_eval_adversarial else ""
            
            logger.info(f"*** New best {objective_type_str} ({adv_status_str} eval{attack_type_str_info}) loss found: {self.best_objective_loss_value:.4f} (at call {self.num_call}) ***")

        # --- Intermediate Testing Block (conditional) ---
        if self.test_every_gens is not None and self.test_every_gens > 0:
            trigger_interval = self.test_every_gens if self.parallel else self.test_every_gens * self.popsize
            
            if self.num_call > 0 and (self.num_call % trigger_interval == 0):
                current_generation = self.num_call if self.parallel else self.num_call // self.popsize
                
                # Calculate train accuracy with the current best prompts
                acc_train_current = self.test_on_train_set()
                self.train_acc.append(acc_train_current.item())
                self.best_train_accuracy = acc_train_current.item()

                eval_loss_type_str = "adversarial" if is_current_eval_adversarial else "clean"
                obj_str = "maximize" if self.maximize_loss else "minimize"
                attack_gen_type_str = f"(AttackType: {self.adv_train_attack_type}, AttackGen: {self.adv_train_attack_prompt_type}"
                if is_current_eval_adversarial and self.adv_train_config.get('sample_ratio', 1.0) < 1.0:
                    attack_gen_type_str += f", SampleRatio: {self.adv_train_config.get('sample_ratio', 1.0)}"
                attack_gen_type_str += ")" if is_current_eval_adversarial else ""

                logger.info(f"\n--- Intermediate Test at Generation ~{current_generation} (Prompts from {eval_loss_type_str} eval {attack_gen_type_str}, obj: {obj_str} loss) ---")
                acc_clean = self.test(attack_config=None)
                self.acc.append(acc_clean.item())
                self.best_accuracy = max(acc_clean.item(), self.best_accuracy)
                logger.info(f"Train Accuracy: {self.best_train_accuracy:.4f}")
                logger.info(f"Test Clean Accuracy: {acc_clean:.4f} (Best Test Clean: {self.best_accuracy:.4f})")

                acc_attacked = torch.tensor(0.0)
                if self.test_attack_config["enabled"] and self.best_prompt_text is not None:
                    attack_name = self.test_attack_config['attack_type'].upper()
                    acc_attacked = self.test(attack_config=self.test_attack_config)
                    self.acc_attack.append(acc_attacked.item())
                    self.best_accuracy_attack = max(acc_attacked.item(), self.best_accuracy_attack)
                    logger.info(f"{attack_name} Accuracy (Test): {acc_attacked:.4f} (Best {attack_name}: {self.best_accuracy_attack:.4f})")
                elif self.test_attack_config["enabled"]:
                    logger.info("Attacked Accuracy (Test): Skipped (no best prompt yet)")
                elif not self.test_attack_config["enabled"]:
                    logger.info("Attacked Accuracy (Test): Disabled in config")

                output_dir = os.path.join(self.output_dir,self.task_name)
                adv_train_attack_type_str_fn = f"_advAttackType{self.adv_train_attack_type}" if self.adv_train_config["enabled"] else ""
                adv_train_attack_prompt_type_str_fn = f"_advPromptGen{self.adv_train_attack_prompt_type}" if self.adv_train_config["enabled"] else ""
                adv_train_sample_ratio_str_fn = f"_advSampleRatio{self.adv_train_config.get('sample_ratio', 1.0)}" if self.adv_train_config["enabled"] and self.adv_train_config.get('sample_ratio', 1.0) < 1.0 else ""
                
                initial_prompt_str_fn = f"_initPrompt" if self.initial_prompt_text is not None else ""
                learned_pos_str_fn = f"_pos{self.learned_prompt_pos}"
                test_attack_str_fn = f"_testAttack{self.test_attack_config['attack_type']}" if self.test_attack_config['enabled'] else "_testAttackNone"

                fname = "{}{}{}_{}_{}_parallel{}_advTrain{}{}{}{}{}_maxLoss{}.pth".format(
                    self.k_shot, self.task_name, initial_prompt_str_fn, learned_pos_str_fn,
                    self.opt_name, self.backbone.replace("/", "-"),
                    self.parallel,
                    self.adv_train_config["enabled"],
                    adv_train_attack_type_str_fn,
                    adv_train_attack_prompt_type_str_fn,
                    adv_train_sample_ratio_str_fn,
                    test_attack_str_fn,
                    self.maximize_loss
                )

                content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                        "best_accuracy":self.best_accuracy, "acc":self.acc,
                        "best_train_accuracy": self.best_train_accuracy, "train_acc": self.train_acc,
                        "best_accuracy_attack": self.best_accuracy_attack, "acc_attack": self.acc_attack,
                        "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                        "training_dataset_snapshot": self._training_dataset_snapshot,
                        "historical_losses":self.loss,
                        "best_objective_loss_value": self.best_objective_loss_value,
                        "maximize_loss_setting": self.maximize_loss,
                        "num_call":self.num_call,
                        "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                        "test_attack_config": self.test_attack_config,
                        "adv_train_config": self.adv_train_config}
                Analysis_Util.save_results(content,output_dir,fname)

        if self.parallel:
            return_value = [l * -1 if self.maximize_loss else l for l in loss_values_final]
        else:
            return_value = loss_values_final * -1 if self.maximize_loss else loss_values_final
        return return_value

    def _split_batch_by_class_for_adv_train(self, images, labels, ratio):
        """
        Splits a batch of images and labels into adversarial and clean sets,
        ensuring the split ratio is applied per class.
        """
        adv_indices = []
        clean_indices = []
        
        # Find all unique classes present in this batch
        unique_classes = torch.unique(labels)
        
        for class_id in unique_classes:
            # Find the indices of all samples belonging to the current class
            class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
            
            # Calculate how many samples of this class to make adversarial
            num_class_samples = len(class_indices)
            num_adv_for_class = int(num_class_samples * ratio)
            
            # Shuffle indices to ensure random selection within the class for each batch
            shuffled_class_indices = class_indices[torch.randperm(num_class_samples)]

            # Split the indices for this class
            adv_indices.append(shuffled_class_indices[:num_adv_for_class])
            clean_indices.append(shuffled_class_indices[num_adv_for_class:])

        # Concatenate all indices from all classes
        adv_indices = torch.cat(adv_indices) if adv_indices else torch.tensor([], dtype=torch.long, device=labels.device)
        clean_indices = torch.cat(clean_indices) if clean_indices else torch.tensor([], dtype=torch.long, device=labels.device)
        
        # Gather the images and labels based on the final index lists
        adv_images = images[adv_indices]
        adv_labels = labels[adv_indices]
        clean_images = images[clean_indices]
        clean_labels = labels[clean_indices]
        
        return adv_images, adv_labels, clean_images, clean_labels
        
    def _run_adversarial_attack(self, images, labels, text_features_for_attack, image_prompt, config, attack_type, text_prompt_to_perturb=None):
        """
        Generates adversarial examples based on the configured attack type.
        Assumes torch.enable_grad() is handled by the caller.
        """
        images_orig = images.clone().detach()
        epsilon = config['epsilon']
        
        if attack_type == "gaussian":
            noise = torch.randn_like(images_orig, device=self.device) * epsilon
            final_perturbed_image = (images_orig + noise).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)
            return final_perturbed_image, None # No text perturbation for gaussian
        
        # --- Parameters for PGD/FGSM ---
        alpha_text_prompt = config.get('alpha_text_prompt', config.get('alpha'))
        
        if attack_type == "fgsm":
            alpha_img = epsilon # FGSM uses epsilon as the step size
            num_iter = 1
        elif attack_type == "pgd":
            alpha_img = config['alpha']
            num_iter = config['num_iter']
        else:
            raise ValueError(f"Unsupported adversarial attack type for iterative method: {attack_type}")

        # --- Initialize Perturbations ---
        # Image perturbation with random start
        delta_img = torch.zeros_like(images_orig, requires_grad=True, device=self.device).to(images_orig.dtype)
        delta_img.data.uniform_(-epsilon, epsilon)
        delta_img.data = torch.clamp(images_orig + delta_img.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images_orig
        delta_img.data = delta_img.data.to(images_orig.dtype)
        
        # Text prompt perturbation
        delta_text_prompt = None
        if text_prompt_to_perturb is not None:
            delta_text_prompt = torch.zeros_like(text_prompt_to_perturb, requires_grad=True, device=self.device).to(text_prompt_to_perturb.dtype)

        # --- Iterative Attack Loop ---
        for _ in range(num_iter):
            grads_to_compute = []
            delta_img.requires_grad_(True)
            grads_to_compute.append(delta_img)
            perturbed_image = (images_orig + delta_img).to(self.dtype)

            # Determine text features for this iteration (either fixed or perturbed)
            effective_text_features_for_iter = text_features_for_attack
            if text_prompt_to_perturb is not None and delta_text_prompt is not None:
                delta_text_prompt.requires_grad_(True)
                grads_to_compute.append(delta_text_prompt)
                perturbed_text_prompt_iter = text_prompt_to_perturb + delta_text_prompt
                effective_text_features_for_iter = self.text_encoder(perturbed_text_prompt_iter)
                effective_text_features_for_iter = effective_text_features_for_iter / effective_text_features_for_iter.norm(dim=-1, keepdim=True)

            # --- Forward Pass ---
            original_im_parallel_state_eval = self.image_encoder.parallel
            self.image_encoder.parallel = False # Ensure single processing during attack step
            image_features = self.image_encoder(perturbed_image, image_prompt)
            self.image_encoder.parallel = original_im_parallel_state_eval

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ effective_text_features_for_iter.t()
            loss = F.cross_entropy(logits, labels)

            # --- Backward Pass ---
            all_grads = torch.autograd.grad(loss, grads_to_compute, only_inputs=True)
            
            # --- Update Step for Image ---
            delta_img_grad = all_grads[0]
            grad_sign_img = delta_img_grad.sign()
            delta_img.data = delta_img.data + alpha_img * grad_sign_img.to(delta_img.dtype)
            delta_img.data = torch.clamp(delta_img.data, -epsilon, epsilon)
            delta_img.data = torch.clamp(images_orig + delta_img.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images_orig

            # --- Update Step for Text (if applicable) ---
            if text_prompt_to_perturb is not None and delta_text_prompt is not None:
                delta_text_prompt_grad = all_grads[1]
                delta_text_prompt.data = delta_text_prompt.data + alpha_text_prompt * delta_text_prompt_grad.to(delta_text_prompt.dtype)

        # --- Finalize Perturbations ---
        final_perturbed_image = (images_orig + delta_img.detach()).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)
        final_perturbed_text_prompt = None
        if text_prompt_to_perturb is not None and delta_text_prompt is not None:
            final_perturbed_text_prompt = (text_prompt_to_perturb + delta_text_prompt.detach()).to(text_prompt_to_perturb.dtype)

        return final_perturbed_image, final_perturbed_text_prompt


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
    def test(self, attack_config=None):
        if self.best_prompt_text is None or self.best_prompt_image is None:
            logger.warning("Test skipped: no best tuned prompt available for evaluation.")
            return torch.tensor(0.0)

        correct = 0.
        total = 0.
        
        # Store and temporarily override parallel flags for encoders during test
        original_text_encoder_parallel = self.text_encoder.parallel
        original_image_encoder_parallel = self.image_encoder.parallel
        self.text_encoder.parallel = False
        self.image_encoder.parallel = False

        desc = "Testing Clean"
        is_attack_test = False
        
        # Use the best tuned prompts for all tests
        current_text_features_for_test = self.text_encoder(self.best_prompt_text)
        current_text_features_for_test = current_text_features_for_test / current_text_features_for_test.norm(dim=-1,keepdim=True)
        current_image_prompt_for_test = self.best_prompt_image

        if attack_config is not None and attack_config.get("enabled", False):
            is_attack_test = True

        for batch in self.test_loader:
            temp_original_class_parallel_attr = self.parallel
            self.parallel = False # Affects parse_batch's internal logic
            image,label = self.parse_batch(batch) # image [B,C,H,W], label [B]
            self.parallel = temp_original_class_parallel_attr # Restore

            total += image.size(0)

            eval_image = image.to(self.dtype)
            final_text_features_for_eval = current_text_features_for_test

            if is_attack_test:
                with torch.enable_grad():
                    eval_image, _ = self._run_adversarial_attack( 
                        images=image,
                        labels=label,
                        text_features_for_attack=current_text_features_for_test, 
                        image_prompt=current_image_prompt_for_test,
                        config=attack_config, 
                        attack_type=attack_config['attack_type'],
                        text_prompt_to_perturb=None 
                    )
                eval_image = eval_image.to(self.dtype)

            image_features = self.image_encoder(eval_image, current_image_prompt_for_test)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale*image_features@final_text_features_for_eval.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        self.text_encoder.parallel = original_text_encoder_parallel
        self.image_encoder.parallel = original_image_encoder_parallel
        
        acc = correct/total
        return acc


    def load_dataset(self):
        if self.task_name == 'CIFAR100':
            self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed)
            self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)
        elif self.task_name == 'CIFAR10':
            self.dataset = CIFAR10(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar10(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed)
            self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess)
        elif self.task_name == 'CIFAR10_PGD': # This seems to imply pre-attacked data, be careful
            self.train_data,self.train_loader = load_train_cifar10_pgd(batch_size=self.batch_size,shots=self.k_shot)
            if self.test_attack_config["enabled"]: # If attack test enabled, use PGD test set
                self.test_data, self.test_loader = load_test_cifar10_pgd(batch_size=self.batch_size)
            else: # Else use clean CIFAR10 test set
                self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess)
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.n_cls = len(self.classes)

        elif self.task_name == 'StanfordCars':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'OxfordPets':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'UCF-101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'DTD':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'EuroSAT':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'Food101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'caltech101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'SUN397':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'ImageNet':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)


    def parse_batch(self,batch):
        image = batch["image"]
        label = batch["label"]
        image = image.to(device=self.device, dtype=self.dtype if image.dtype != torch.uint8 else torch.float32)
        if image.dtype == torch.uint8: # common for PIL loaded images
            image = image.float() / 255.0 # Normalize if uint8
        
        label = label.to(device=self.device)
        
        # This repetition is for when eval is processing a whole population (self.parallel=True)
        # and each member of the population needs to be evaluated on the same batch of images.
        if self.parallel: 
            image = image.repeat(self.popsize, 1, 1, 1)
            # label remains [B_orig] and is used for each repeated image set.
        return image, label