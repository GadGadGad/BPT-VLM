# Shallow_Prompt_CLIP_PGD.py

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
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.backbone,device=self.device)
        self.loss = []
        self.acc = []
        self.acc_pgd = []
        self.pgd_config = cfg.get("pgd", {"enabled": False})
        self.pgd_original_prompt = self.pgd_config.get("original_prompt", False)
        self.adv_train_config = cfg.get("adv_train", {"enabled": False})
        self.adv_train_attack_prompt_type = self.adv_train_config.get("attack_prompt_type", "on-the-fly") # New attribute

        self.load_dataset()

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
            logger.info(f"  Training PGD Config: \
                        Epsilon={self.adv_train_config['epsilon']}, \
                        Alpha={self.adv_train_config['alpha']}, \
                        Iter={self.adv_train_config['num_iter']}")
            logger.info(f"  Adversarial Attack Prompt Type for Training: {self.adv_train_attack_prompt_type}")
            logger.info(f"  Adversarial tuning will occur when self.num_call % self.test_every == 0." \
                if not self.adv_train_config.get('all_call',False) else \
                "  Adversarial tuning will occur for the whole progress.")
        else:
            logger.info("--- Standard (Clean) Prompt Optimization ---")
        if self.pgd_config["enabled"] or self.adv_train_config["enabled"]:
            logger.info("PGD Testing ENABLED.")
            mean = self.preprocess.transforms[-1].mean
            std = self.preprocess.transforms[-1].std
            self.norm_mean = torch.tensor(mean).to(self.device).view(3, 1, 1)
            self.norm_std = torch.tensor(std).to(self.device).view(3, 1, 1)
            self.norm_upper_limit = ((1 - self.norm_mean) / self.norm_std).to(self.device)
            self.norm_lower_limit = ((0 - self.norm_mean) / self.norm_std).to(self.device)
            logger.info(f"  Test PGD Config: \
                        Epsilon={self.pgd_config.get('epsilon', 'N/A')},\
                        Alpha={self.pgd_config.get('alpha', 'N/A')},\
                        Iter={self.pgd_config.get('num_iter', 'N/A')}")
        else:
            logger.info("PGD Testing DISABLED.")
        # -----------------------------------------

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
        self.best_accuracy_pgd = 0.0
        self.test_every = cfg["test_every"] if self.parallel else cfg["test_every"]*self.popsize
        self.sigma = cfg["sigma"]
        # Lauguage Linear Layer
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
        #mu = 0.0
        mu = mu_hat*3072/self.intrinsic_dim_V
        std = std_hat * np.sqrt(3072/self.intrinsic_dim_V) * self.sigma
        logger.info('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)


    def get_text_information(self,caption=None):
        prompt_prefix = " ".join(["X"] * self.n_prompt_tokens_L)
        if caption is None:
            classnames = [name.replace("_", " ").replace("-"," ") for name in self.classes]
            pattern_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_pattern_prompts= torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls":self.n_cls, "n_prompt_tokens_L":self.n_prompt_tokens_L,
                       "init_pattern_embedding":init_pattern_embedding, "tokenized_pattern_prompts":tokenized_pattern_prompts,
                       "batch_size":self.batch_size,"pop_size":self.popsize,"parallel":self.parallel}
        else:
            pattern_prompt = prompt_prefix + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls":1,"n_prompt_tokens_L":self.n_prompt_tokens_L,
                       "init_pattern_embedding":init_pattern_embedding, "tokenized_pattern_prompts":tokenized_pattern_prompts,"batch_size":self.batch_size,
                       "pop_size":self.popsize,"parallel":self.parallel}
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
        self.num_call += 1
        is_current_eval_adversarial = False
        if self.adv_train_config["enabled"]:
            if self.adv_train_config.get("all_call", False):
                is_current_eval_adversarial = True
            elif self.num_call > 0 and self.test_every > 0 and (self.num_call % self.test_every == 0):
                is_current_eval_adversarial = True

        loss_accumulator = 0
        logit_scale = self.logit_scale.exp()

        current_prompt_text_for_eval_single = None
        current_prompt_image_for_eval_single = None

        # Determine text features for attack generation if needed
        text_features_for_attack_generation = None # For "constant" or "perturbed" during adv_train
        text_prompt_for_attack_generation_perturbed = None # For "perturbed" specifically

        if self.parallel:
            loss_accumulator = [0.0] * self.popsize
            # `prompt_text_list_or_tensor` is a list of prompt tensors for each member of population
            # `text_features` will be [pop_size, n_cls, D]
            all_pop_text_features = []
            for p_text in prompt_text_list_or_tensor:
                features = self.text_encoder(p_text)
                features = features / features.norm(dim=-1, keepdim=True)
                all_pop_text_features.append(features)
            pop_txt_features = torch.stack(all_pop_text_features) # [pop_size, n_cls, D]

            if is_current_eval_adversarial and self.adv_train_attack_prompt_type == "constant":
                # For parallel + constant, we need one set of attack text features, used by all pop members
                # Note: This assumes the "constant" prompt is class-agnostic in its structure or uses a fixed template.
                # If the constant prompt structure were different per population member, this would need adjustment.
                # For simplicity here, we use the standard `get_original_text_features()` which is class-specific.
                text_features_for_attack_generation = self.get_original_text_features()
        else:
            # `prompt_text_list_or_tensor` is a single prompt tensor
            # `text_features` will be [n_cls, D]
            text_features = self.text_encoder(prompt_text_list_or_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            current_prompt_text_for_eval_single = prompt_text_list_or_tensor
            current_prompt_image_for_eval_single = prompt_image_list_or_tensor

            if is_current_eval_adversarial:
                if self.adv_train_attack_prompt_type == "constant":
                    text_features_for_attack_generation = self.get_original_text_features()
                elif self.adv_train_attack_prompt_type == "on-the-fly":
                    text_features_for_attack_generation = text_features # Use current learnable prompt's features
                elif self.adv_train_attack_prompt_type == "perturbed":
                    # For perturbed, attack uses current text_features initially, then text prompt itself is perturbed.
                    text_features_for_attack_generation = text_features
                    text_prompt_for_attack_generation_perturbed = current_prompt_text_for_eval_single.clone().detach()


        for batch in self.train_loader:
            clean_image, label = self.parse_batch(batch)
            if self.parallel:
                B = label.shape[0]
                pop_clean_image = clean_image.view(self.popsize, B, *clean_image.shape[1:])

                for i in range(self.popsize):
                    # Text features for final loss calculation with current learnable prompt
                    current_txt_features_for_loss = pop_txt_features[i]
                    current_img_prompt_for_loss = prompt_image_list_or_tensor[i]
                    current_clean_images = pop_clean_image[i]

                    # Determine text features for PGD attack generation for this population member
                    current_txt_features_for_pgd_gen = current_txt_features_for_loss # Default to on-the-fly
                    current_text_prompt_for_pgd_gen_perturbed = prompt_text_list_or_tensor[i].clone().detach() if self.adv_train_attack_prompt_type == "perturbed" else None

                    if is_current_eval_adversarial:
                        if self.adv_train_attack_prompt_type == "constant":
                            # `text_features_for_attack_generation` was already prepared (same for all pop members if parallel)
                            current_txt_features_for_pgd_gen = text_features_for_attack_generation
                            current_text_prompt_for_pgd_gen_perturbed = None # Not applicable for constant

                        # Note: "on-the-fly" uses `current_txt_features_for_loss` by default
                        # "perturbed" also starts with `current_txt_features_for_loss` for the PGD text features arg,
                        # but the `_pgd_attack` function will handle perturbing the text prompt itself if `text_prompt_to_perturb` is passed.

                        with torch.enable_grad():
                            eval_image_i, _ = self._pgd_attack(
                                images=current_clean_images,
                                labels=label,
                                text_features_for_attack=current_txt_features_for_pgd_gen, # Features used to guide image perturbation
                                image_prompt=current_img_prompt_for_loss, # Image prompt for loss (and potentially attack if visual prompts were also perturbed)
                                config=self.adv_train_config,
                                text_prompt_to_perturb=current_text_prompt_for_pgd_gen_perturbed if self.adv_train_attack_prompt_type == "perturbed" else None
                            )
                        eval_image_i = eval_image_i.to(self.dtype)
                        # For "perturbed", the text prompt used for the final loss might also be the perturbed one if that's the desired behavior.
                        # Here, we assume the final loss uses the original candidate learnable prompt for consistency,
                        # and the "perturbed" attack's goal was to find a strong image attack using a (temporarily) perturbed text prompt.
                        # If the paper implies the *final loss* uses the perturbed text prompt, `current_txt_features_for_loss` would need to be re-derived.
                        # For simplicity, we keep the loss on the original candidate prompt but attack was made harder.
                    else:
                        eval_image_i = current_clean_images.to(self.dtype)

                    original_im_parallel = self.image_encoder.parallel
                    self.image_encoder.parallel = False
                    image_features_i = self.image_encoder(eval_image_i, current_img_prompt_for_loss)
                    self.image_encoder.parallel = original_im_parallel

                    image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
                    tmp_logits = logit_scale * image_features_i @ current_txt_features_for_loss.t()
                    loss_accumulator[i] += self.metric(tmp_logits, label).item()
            else: # Not parallel
                current_clean_images = clean_image
                current_txt_features_for_loss = text_features # From the single candidate prompt
                current_img_prompt_for_loss = current_prompt_image_for_eval_single

                final_text_prompt_for_loss = current_prompt_text_for_eval_single
                final_text_features_for_loss = current_txt_features_for_loss

                if is_current_eval_adversarial:
                    # `text_features_for_attack_generation` and `text_prompt_for_attack_generation_perturbed` were set earlier
                    text_prompt_to_perturb_for_pgd = text_prompt_for_attack_generation_perturbed if self.adv_train_attack_prompt_type == "perturbed" else None

                    with torch.enable_grad():
                        eval_image, perturbed_text_prompt_if_any = self._pgd_attack(
                            images=current_clean_images,
                            labels=label,
                            text_features_for_attack=text_features_for_attack_generation, # Features guiding image attack
                            image_prompt=current_img_prompt_for_loss,
                            config=self.adv_train_config,
                            text_prompt_to_perturb=text_prompt_to_perturb_for_pgd
                        )
                    eval_image = eval_image.to(self.dtype)

                    if self.adv_train_attack_prompt_type == "perturbed" and perturbed_text_prompt_if_any is not None:
                        # If APT implies the *final loss* is on the perturbed prompt, update features for loss
                        # For now, let's assume the paper means the attack generation uses perturbed prompt,
                        # but the optimization step for the original prompt uses the loss against that original prompt.
                        # final_text_features_for_loss = self.text_encoder(perturbed_text_prompt_if_any)
                        # final_text_features_for_loss = final_text_features_for_loss / final_text_features_for_loss.norm(dim=-1, keepdim=True)
                        pass # Sticking to original prompt for loss calculation after 'perturbed' attack
                else:
                    eval_image = current_clean_images.to(self.dtype)

                image_features = self.image_encoder(eval_image, current_img_prompt_for_loss)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ final_text_features_for_loss.t()
                loss_accumulator += self.metric(logits, label).item()

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
            attack_type_str = f" (Attack Gen: {self.adv_train_attack_prompt_type})" if is_current_eval_adversarial else ""
            logger.info(f"*** New best {objective_type_str} ({adv_status_str} eval{attack_type_str}) loss found: {self.best_objective_loss_value:.4f} (at call {self.num_call}) ***")

        if self.num_call > 0 and self.test_every > 0 and (self.num_call % self.test_every == 0):
            eval_loss_type_str = "adversarial" if is_current_eval_adversarial else "clean"
            obj_str = "maximize" if self.maximize_loss else "minimize"
            attack_gen_type_str = f"(Attack Gen: {self.adv_train_attack_prompt_type})" if is_current_eval_adversarial else ""
            logger.info(f"\n--- Testing at call {self.num_call} (Prompts from {eval_loss_type_str} eval {attack_gen_type_str}, objective: {obj_str} loss) ---")
            acc_clean = self.test(attack_config=None)
            self.acc.append(acc_clean.item())
            self.best_accuracy = max(acc_clean.item(), self.best_accuracy)
            logger.info(f"Clean Accuracy: {acc_clean:.4f} (Best Clean: {self.best_accuracy:.4f})")

            acc_attacked = torch.tensor(0.0)
            if self.pgd_config["enabled"] and self.best_prompt_text is not None:
                acc_attacked = self.test(attack_config=self.pgd_config)
                self.acc_pgd.append(acc_attacked.item())
                self.best_accuracy_pgd = max(acc_attacked.item(), self.best_accuracy_pgd)
                pgd_test_type_str = " (Original Prompts)" if self.pgd_config.get("original_prompt", False) else ""
                logger.info(f"PGD Accuracy (Test{pgd_test_type_str}): {acc_attacked:.4f} (Best PGD{pgd_test_type_str}: {self.best_accuracy_pgd:.4f})")
            elif self.pgd_config["enabled"]:
                 logger.info("PGD Accuracy (Test): Skipped (no best prompt yet)")
            elif not self.pgd_config["enabled"]:
                 logger.info("PGD Accuracy (Test): Disabled in config")


            output_dir = os.path.join(self.output_dir,self.task_name)
            adv_train_attack_prompt_type_str = f"_advPromptGen{self.adv_train_attack_prompt_type}" if self.adv_train_config["enabled"] else ""
            fname = "{}_{}_{}_parallel{}_advTrain{}{}_pgdTest{}_pgdOrg{}_maxLoss{}.pth".format(
                self.task_name, self.opt_name, self.backbone.replace("/","-"),
                self.parallel,
                self.adv_train_config["enabled"],
                adv_train_attack_prompt_type_str,
                self.pgd_config["enabled"],
                self.pgd_config.get("original_prompt", False),
                self.maximize_loss
            )

            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                       "best_accuracy":self.best_accuracy, "acc":self.acc,
                       "best_accuracy_pgd": self.best_accuracy_pgd, "acc_pgd": self.acc_pgd,
                       "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                       "historical_losses":self.loss,
                       "best_objective_loss_value": self.best_objective_loss_value,
                       "maximize_loss_setting": self.maximize_loss,
                       "num_call":self.num_call,
                       "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                       "pgd_config_test": self.pgd_config,
                       "adv_train_config": self.adv_train_config}
            Analysis_Util.save_results(content,output_dir,fname)

        if self.parallel:
            return_value = [l * -1 if self.maximize_loss else l for l in loss_values_final]
        else:
            return_value = loss_values_final * -1 if self.maximize_loss else loss_values_final
        return return_value


    def _pgd_attack(self, images, labels, text_features_for_attack, image_prompt, config, text_prompt_to_perturb=None):
        images_orig = images.clone().detach()
        labels = labels.clone().detach()
        current_text_features_for_attack = text_features_for_attack.clone().detach()
        current_text_prompt = None
        if text_prompt_to_perturb is not None:
            current_text_prompt = text_prompt_to_perturb.clone().detach() # This is [n_prompt_tokens_L, ctx_dim_L] per class

        epsilon = config['epsilon']
        alpha_img = config['alpha'] # Renamed for clarity
        num_iter = config['num_iter']
        alpha_text_prompt = config.get('alpha_text_prompt', alpha_img) # Step size for text prompt perturbation

        delta_img = torch.zeros_like(images_orig, requires_grad=True, device=self.device).to(images_orig.dtype)
        delta_img.data.uniform_(-epsilon, epsilon)
        delta_img.data = torch.clamp(images_orig + delta_img.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images_orig
        delta_img.data = delta_img.data.to(images_orig.dtype)

        delta_text_prompt = None
        if current_text_prompt is not None: # If we are perturbing the text prompt
            # Assuming current_text_prompt is [n_cls, n_prompt_tokens_L, ctx_dim_L] or similar if it's already processed by text_encoder
            # For APT paper, it's likely the raw context vectors [V]1..[V]M that are perturbed.
            # Here, `text_prompt_to_perturb` is assumed to be the context vectors for each class.
            # For simplicity, let's assume `text_prompt_to_perturb` are the embeddings that `text_encoder` takes.
            # If `text_prompt_to_perturb` is the raw `[M, D_ctx]` vector(s), then TextEncoder needs to re-run.
            # Let's assume `text_prompt_to_perturb` is the `[n_prompt_tokens_L, ctx_dim_L]` for a single class or
            # the full [n_cls, total_len, D_emb] tensor from TextEncoder.
            # For APT, it's the M context vectors that are learnable and thus perturbable.
            # `text_prompt_to_perturb` is what generate_text_prompts returns, a single [M, D_ctx] for one intrinsic vector.
            # Let's assume text_prompt_to_perturb is a single [M_L, D_ctx_L] learnable context.
            # And text_features_for_attack are derived from this.
            # This makes text_prompt_to_perturb the thing that needs grad.
            delta_text_prompt = torch.zeros_like(current_text_prompt, requires_grad=True, device=self.device).to(current_text_prompt.dtype)
            # No initial random perturbation for text prompt delta, start from 0.
            # No clamping for text prompt delta to an epsilon-ball in the same way as images,
            # as text embeddings don't have a fixed [0,1] range.
            # APT paper Algorithm 1, line 9: δ' = δ' + α'·∇tL. No explicit clamping for δ'.

        for iter_idx in range(num_iter):
            delta_img.requires_grad_(True)
            perturbed_image = (images_orig + delta_img).to(self.dtype)

            # Handle text prompt perturbation
            effective_text_features_for_iter = current_text_features_for_attack
            if current_text_prompt is not None and delta_text_prompt is not None:
                delta_text_prompt.requires_grad_(True)
                perturbed_text_prompt_iter = current_text_prompt + delta_text_prompt
                # Need to re-encode the perturbed text prompt to get features
                # This assumes self.text_encoder can handle the [M, D_ctx] directly for all classes.
                # This part needs careful handling of how text_encoder expects its input if perturbing context vectors.
                # For simplicity, if text_prompt_to_perturb is the direct [M, D_ctx] that TextEncoder.forward inserts,
                # then TextEncoder needs to be called with this perturbed_text_prompt_iter.
                effective_text_features_for_iter = self.text_encoder(perturbed_text_prompt_iter) # This call will re-embed and encode
                effective_text_features_for_iter = effective_text_features_for_iter / effective_text_features_for_iter.norm(dim=-1, keepdim=True)


            original_im_parallel = self.image_encoder.parallel
            self.image_encoder.parallel = False
            image_features = self.image_encoder(perturbed_image, image_prompt)
            self.image_encoder.parallel = original_im_parallel

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ effective_text_features_for_iter.t()

            loss = F.cross_entropy(logits, labels)

            # Calculate gradients
            grads_to_compute = [delta_img]
            if current_text_prompt is not None and delta_text_prompt is not None:
                grads_to_compute.append(delta_text_prompt)

            all_grads = torch.autograd.grad(loss, grads_to_compute,
                                             only_inputs=True,
                                             retain_graph=False, # False if not needed for further grad ops in this iter
                                             create_graph=False
                                             )
            delta_img_grad = all_grads[0]

            # Update image delta
            grad_sign_img = delta_img_grad.sign()
            delta_img.data = delta_img.data + alpha_img * grad_sign_img.to(delta_img.dtype)
            delta_img.data = torch.clamp(delta_img.data, -epsilon, epsilon)
            delta_img.data = torch.clamp(images_orig + delta_img.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images_orig

            # Update text prompt delta if applicable
            if current_text_prompt is not None and delta_text_prompt is not None:
                delta_text_prompt_grad = all_grads[1]
                # APT paper does not specify a sign for text prompt updates, uses raw grad.
                delta_text_prompt.data = delta_text_prompt.data + alpha_text_prompt * delta_text_prompt_grad.to(delta_text_prompt.dtype)
                # No clamping for delta_text_prompt to epsilon ball in the paper.

        final_perturbed_image = (images_orig + delta_img.detach()).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)
        final_perturbed_text_prompt = None
        if current_text_prompt is not None and delta_text_prompt is not None:
            final_perturbed_text_prompt = (current_text_prompt + delta_text_prompt.detach()).to(current_text_prompt.dtype)

        return final_perturbed_image, final_perturbed_text_prompt


    @torch.no_grad()
    def test(self, attack_config=None):
        if self.best_prompt_text is None or self.best_prompt_image is None:
            return torch.tensor(0.0)

        correct = 0.
        total = 0.
        original_parallel_state = self.parallel
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False

        desc = "Testing Clean"
        is_attack_test = False
        current_text_features_for_test = None
        current_image_prompt_for_test = None
        current_text_prompt_raw_for_attack_test = None # For "perturbed" test if needed

        if attack_config is not None and attack_config.get("enabled", False):
            desc = f"Testing PGD(eps={attack_config['epsilon']}, iter={attack_config['num_iter']})"
            is_attack_test = True

            if self.pgd_config.get("original_prompt", False):
                desc += " (Original Prompts)"
                current_text_features_for_test = self.get_original_text_features()
                current_image_prompt_for_test = None
                # For "perturbed" attack on original prompts, we'd need the original prompt structure.
                # This case is less common for APT-style "perturbed" text prompt attacks.
            else: # Use tuned prompts
                desc += " (Current Best Tuned Prompts)"
                if self.best_prompt_text is None or self.best_prompt_image is None:
                    logger.warning("Tuned PGD test skipped as best prompts are not available.")
                    return torch.tensor(0.0)
                current_text_features_for_test = self.text_encoder(self.best_prompt_text)
                current_text_features_for_test = current_text_features_for_test / current_text_features_for_test.norm(dim=-1,keepdim=True)
                current_image_prompt_for_test = self.best_prompt_image
                current_text_prompt_raw_for_attack_test = self.best_prompt_text # Pass the raw tuned prompt for potential perturbation

        else: # Clean test
            if self.best_prompt_text is None or self.best_prompt_image is None:
                logger.warning(f"Clean test skipped as best prompts are not available.")
                return torch.tensor(0.0)
            current_text_features_for_test = self.text_encoder(self.best_prompt_text)
            current_text_features_for_test = current_text_features_for_test / current_text_features_for_test.norm(dim=-1,keepdim=True)
            current_image_prompt_for_test = self.best_prompt_image


        for batch in tqdm(self.test_loader, desc=desc, leave=False):
            image,label = self.parse_batch(batch)
            total += image.size(0)

            eval_image = image.to(self.dtype)
            final_text_features_for_eval = current_text_features_for_test

            if is_attack_test:
                text_prompt_to_perturb_for_test_attack = None
                # Decide if text prompt perturbation is part of this test attack
                # Typically, PGD test config might not include text prompt perturbation details,
                # but if it did, or if "adv_train_attack_prompt_type" was 'perturbed' and we wanted to mirror that in testing.
                # For simplicity, let's assume standard PGD test only perturbs images unless explicitly configured for text.
                # if attack_config.get("perturb_text_in_test", False) and current_text_prompt_raw_for_attack_test is not None:
                # text_prompt_to_perturb_for_test_attack = current_text_prompt_raw_for_attack_test

                with torch.enable_grad():
                    eval_image, perturbed_text_prompt_from_attack = self._pgd_attack(
                        image,
                        label,
                        current_text_features_for_test, # Features to guide image attack
                        current_image_prompt_for_test,
                        attack_config, # Test PGD config
                        text_prompt_to_perturb=None # Standard test PGD usually doesn't perturb text prompt
                    )
                eval_image = eval_image.to(self.dtype)
                # If perturbed_text_prompt_from_attack was generated and meant to be used for eval:
                # if perturbed_text_prompt_from_attack is not None:
                # final_text_features_for_eval = self.text_encoder(perturbed_text_prompt_from_attack)
                # final_text_features_for_eval = final_text_features_for_eval / final_text_features_for_eval.norm(dim=-1, keepdim=True)


            image_features = self.image_encoder(eval_image, current_image_prompt_for_test)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale*image_features@final_text_features_for_eval.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        self.parallel = original_parallel_state
        self.text_encoder.parallel = original_parallel_state
        self.image_encoder.parallel = original_parallel_state

        acc = correct/total
        return acc


    def load_dataset(self):
        if self.task_name == 'CIFAR100':
            self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess)
            self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)
        elif self.task_name == 'CIFAR10':
            self.dataset = CIFAR10(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar10(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess)
            self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess)
        elif self.task_name == 'CIFAR10_PGD':
            self.train_data,self.train_loader = load_train_cifar10_pgd(batch_size=self.batch_size,shots=self.k_shot)
            if self.pgd_config["enabled"]:
                self.test_data, self.test_loader = load_test_cifar10_pgd(batch_size=self.batch_size)
            else:
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
        image = image.to(device=self.device, dtype=self.dtype)
        label = label.to(device=self.device)
        if self.parallel:
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label