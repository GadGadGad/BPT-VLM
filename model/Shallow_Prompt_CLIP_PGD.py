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
        self.train_acc = [] # New: To store training accuracy from eval
        self.acc_pgd = []
        self.pgd_config = cfg.get("pgd", {"enabled": False})
        self.pgd_original_prompt = self.pgd_config.get("original_prompt", False)
        self.adv_train_config = cfg.get("adv_train", {"enabled": False})
        self.adv_train_attack_prompt_type = self.adv_train_config.get("attack_prompt_type", "on-the-fly")
        self.adv_train_attack_type = self.adv_train_config.get("attack_type", "pgd")

        self.load_dataset()
        
        self.resume_from_pth = cfg.get("resume_from_pth", None)
        if self.resume_from_pth:
            self.load_state_from_pth(self.resume_from_pth)
            
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
            logger.info(f"  Training Attack Type: {self.adv_train_attack_type}")
            logger.info(f"  Training PGD Config: \
                        Epsilon={self.adv_train_config['epsilon']}, \
                        Alpha={self.adv_train_config['alpha']}, \
                        Iter={self.adv_train_config['num_iter']}")
            logger.info(f"  Adversarial Attack Prompt Type for Training: {self.adv_train_attack_prompt_type}")
            logger.info(f"  Adversarial Training Sample Ratio: {self.adv_train_config.get('sample_ratio', 1.0)}")
            logger.info(f"  Adversarial tuning will occur when self.num_call % self.test_every == 0." \
                if not self.adv_train_config.get('all_call',False) else \
                "  Adversarial tuning will occur for the whole progress.")
        else:
            logger.info("--- Standard (Clean) Prompt Optimization ---")
        if self.pgd_config["enabled"] or self.adv_train_config["enabled"]:
            logger.info("PGD Operations (Test or Train) ENABLED.")
            mean = self.preprocess.transforms[-1].mean
            std = self.preprocess.transforms[-1].std
            self.norm_mean = torch.tensor(mean).to(self.device).view(3, 1, 1)
            self.norm_std = torch.tensor(std).to(self.device).view(3, 1, 1)
            self.norm_upper_limit = ((1 - self.norm_mean) / self.norm_std).to(self.device)
            self.norm_lower_limit = ((0 - self.norm_mean) / self.norm_std).to(self.device)
            if self.pgd_config["enabled"]:
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
        self.best_train_acc = 0.0 # New: Track best training accuracy
        self.best_accuracy_pgd = 0.0
        self.test_every = cfg["test_every"] if self.parallel else cfg["test_every"]*self.popsize # test_every is in terms of num_call
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
            final_loss = ce_loss
        elif self.loss_type == "focal":
            gamma = 2
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            final_loss = focal_loss
        return final_loss
# Add this method inside the PromptCLIP_Shallow class
    def load_state_from_pth(self, pth_path):
        """
        Loads the state from a saved .pth file to continue tuning.
        """
        if not os.path.exists(pth_path):
            logger.warning(f"Resume path not found: {pth_path}. Starting from scratch.")
            return

        logger.info(f"--- Resuming tuning from checkpoint: {pth_path} ---")
        content = torch.load(pth_path, map_location=self.device)

        # Load the state of the linear projection layers
        if "Linear_L" in content and "Linear_V" in content:
            self.linear_L.load_state_dict(content["Linear_L"])
            self.linear_V.load_state_dict(content["Linear_V"])
            logger.info("Loaded Linear_L and Linear_V state dicts.")
        else:
            logger.warning("Could not find Linear_L/Linear_V state dicts in checkpoint.")

        # Load the best prompts found so far
        if "best_prompt_text" in content and content["best_prompt_text"] is not None:
            self.best_prompt_text = content["best_prompt_text"].to(self.device)
            self.best_prompt_image = content["best_prompt_image"].to(self.device)
            logger.info("Loaded best text and image prompts.")
        
        # Load historical data to append to it
        self.loss = content.get("historical_losses", [])
        self.acc = content.get("acc", [])
        self.train_acc = content.get("train_acc_history", [])
        self.acc_pgd = content.get("acc_pgd", [])
        self.num_call = content.get("num_call", 0)
        self.best_accuracy = content.get("best_accuracy", 0.0)
        self.best_train_acc = content.get("best_train_accuracy", 0.0)
        self.best_accuracy_pgd = content.get("best_accuracy_pgd", 0.0)
        self.best_objective_loss_value = content.get("best_objective_loss_value", float('inf'))
        if self.maximize_loss:
            self.best_objective_loss_value = content.get("best_objective_loss_value", -float('inf'))
        
        logger.info(f"Resumed from call number: {self.num_call}")
        logger.info(f"Resumed with best objective loss: {self.best_objective_loss_value:.4f}")
        logger.info(f"Resumed with best test accuracy: Clean={self.best_accuracy:.4f}, PGD={self.best_accuracy_pgd:.4f}")
    @torch.no_grad()
    def eval(self, prompt_zip):
        prompt_text_list, prompt_image_list = prompt_zip[0], prompt_zip[1]
        self.num_call += 1 # num_call increments per fitness evaluation

        is_current_eval_adversarial = False
        if self.adv_train_config["enabled"]:
            if self.adv_train_config.get("all_call", False):
                is_current_eval_adversarial = True
            elif self.num_call > 0 and self.test_every > 0 and (self.num_call % self.test_every == 0):
                is_current_eval_adversarial = True

        logit_scale = self.logit_scale.exp()

        # --- SINGLE/SEQUENTIAL EVALUATION PATH ---
        if not self.parallel:
            self.num_call += (self.popsize - 1)
            loss_accumulator = 0
            correct_accumulator = 0.0
            total_samples = 0.0
            text_features = self.text_encoder(prompt_text_list)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_features_for_attack_generation = text_features
            text_prompt_for_attack_generation_perturbed = None
            if is_current_eval_adversarial:
                if self.adv_train_attack_prompt_type == "constant":
                    text_features_for_attack_generation = self.get_original_text_features()
                elif self.adv_train_attack_prompt_type == "perturbed":
                    text_prompt_for_attack_generation_perturbed = prompt_text_list.clone().detach()

            for batch_idx, batch in enumerate(self.train_loader):
                clean_image_orig, label_orig = self.parse_batch(batch)
                eval_image, eval_labels = clean_image_orig, label_orig
                text_features_for_loss = text_features

                if is_current_eval_adversarial:
                    attack_type_kwargs = {
                        "images": clean_image_orig,
                        "labels": label_orig,
                        "text_features_for_attack": text_features_for_attack_generation,
                        "image_prompt": prompt_image_list,
                        "config": self.adv_train_config,
                        "text_prompt_to_perturb": text_prompt_for_attack_generation_perturbed,
                        "is_parallel": False
                    }
                    if self.adv_train_attack_type == "pgd":
                        with torch.enable_grad():
                            eval_image, perturbed_prompt = self._run_adversarial_attack(**attack_type_kwargs)
                    else:
                        eval_image, perturbed_prompt = self._run_adversarial_attack(**attack_type_kwargs)
                    
                    if self.adv_train_attack_prompt_type == "perturbed" and perturbed_prompt is not None:
                        text_features_for_loss = self.text_encoder(perturbed_prompt)
                        text_features_for_loss = text_features_for_loss / text_features_for_loss.norm(dim=-1, keepdim=True)
                
                image_features = self.image_encoder(eval_image.to(self.dtype), prompt_image_list)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features_for_loss.t()
                loss_accumulator += torch.sum(self.metric(logits, eval_labels)).item()
                
                # Accuracy calculation
                prediction = logits.argmax(dim=-1)
                correct_accumulator += (prediction == eval_labels).float().sum().item()
                total_samples += eval_labels.size(0)


            loss_values_final = loss_accumulator / len(self.train_data)
            train_acc_final = correct_accumulator / total_samples
            prompt_candidate_text = prompt_text_list
            prompt_candidate_image = prompt_image_list
            epoch_best_loss_in_eval = loss_values_final

        # --- FULLY PARALLEL EVALUATION PATH ---
        else:
            loss_accumulator = torch.zeros(self.popsize, device=self.device)
            correct_accumulator = torch.zeros(self.popsize, device=self.device)

            # [FIXED] 1. Generate all text features for the population in a single, vectorized call.
            # self.text_encoder.parallel is True. Pass the list of prompts directly.
            pop_text_features_flat = self.text_encoder(prompt_text_list) # -> [P * C, D]
            pop_text_features_flat = pop_text_features_flat / pop_text_features_flat.norm(dim=-1, keepdim=True)
            pop_txt_features = pop_text_features_flat.view(self.popsize, self.n_cls, -1) # -> [P, C, D]

            text_features_for_attack_generation = pop_txt_features # Default 'on-the-fly'
            if is_current_eval_adversarial and self.adv_train_attack_prompt_type == "constant":
                # Use same original prompt features for all pop members
                text_features_for_attack_generation = self.get_original_text_features().unsqueeze(0).expand(self.popsize, -1, -1)

            for batch_idx, batch in enumerate(self.train_loader):
                clean_images_pop, labels_orig = self.parse_batch(batch)
                B_actual = labels_orig.shape[0]

                eval_images_pop = clean_images_pop.to(self.dtype)
                pop_text_features_for_loss = pop_txt_features

                if is_current_eval_adversarial:
                    adv_sample_ratio = self.adv_train_config.get('sample_ratio', 1.0)
                    num_adv_samples = int(B_actual * adv_sample_ratio)
                    
                    if num_adv_samples > 0:
                        adv_labels = labels_orig[:num_adv_samples]
                        clean_images_reshaped = clean_images_pop.view(self.popsize, B_actual, *clean_images_pop.shape[1:])
                        adv_images_to_perturb_pop = clean_images_reshaped[:, :num_adv_samples, :, :, :].reshape(self.popsize * num_adv_samples, *clean_images_pop.shape[1:])

                        # [FIXED] Call the parallel-aware attack function
                        attack_kwargs = {
                            "images": adv_images_to_perturb_pop,
                            "labels": adv_labels.repeat(self.popsize),
                            "image_prompt": prompt_image_list,
                            "config": self.adv_train_config,
                            "text_features_for_attack": text_features_for_attack_generation,
                            "text_prompt_to_perturb": None, # Perturbed text prompt not supported in parallel mode
                            "is_parallel": True
                        }
                        if self.adv_train_attack_type == 'pgd':
                            with torch.enable_grad():
                                perturbed_images, _ = self._run_adversarial_attack(**attack_kwargs)
                        else:
                            perturbed_images, _ = self._run_adversarial_attack(**attack_kwargs)
                        
                        perturbed_images_reshaped = perturbed_images.view(self.popsize, num_adv_samples, *perturbed_images.shape[1:])
                        eval_images_reshaped = eval_images_pop.view(self.popsize, B_actual, *eval_images_pop.shape[1:])
                        eval_images_reshaped[:, :num_adv_samples, :, :, :] = perturbed_images_reshaped
                        eval_images_pop = eval_images_reshaped.reshape(self.popsize * B_actual, *eval_images_pop.shape[1:])

                pop_image_features = self.image_encoder(eval_images_pop, prompt_image_list)
                pop_image_features = pop_image_features / pop_image_features.norm(dim=-1, keepdim=True)
                
                pop_image_features_reshaped = pop_image_features.view(self.popsize, B_actual, -1)
                pop_txt_features_t = pop_text_features_for_loss.transpose(1, 2)

                logits = logit_scale * torch.bmm(pop_image_features_reshaped, pop_txt_features_t)
                logits_flat = logits.view(self.popsize * B_actual, self.n_cls)
                labels_flat = labels_orig.unsqueeze(0).expand(self.popsize, -1).reshape(self.popsize * B_actual)
                
                batch_losses = self.metric(logits_flat, labels_flat)
                batch_losses_per_member = batch_losses.view(self.popsize, B_actual).sum(dim=1)
                loss_accumulator += batch_losses_per_member
                
                # Accuracy calculation (parallel)
                predictions = logits.argmax(dim=-1) # [P, B]
                labels_expanded = labels_orig.unsqueeze(0).expand(self.popsize, -1) # [P, B]
                correct_batch_per_member = (predictions == labels_expanded).float().sum(dim=1) # [P]
                correct_accumulator += correct_batch_per_member

            loss_values_final = (loss_accumulator / len(self.train_data)).tolist()
            train_accs_final = (correct_accumulator / len(self.train_data)).tolist()
            
            if self.maximize_loss:
                epoch_best_loss_in_eval = max(loss_values_final)
            else:
                epoch_best_loss_in_eval = min(loss_values_final)
            
            best_idx_in_eval = loss_values_final.index(epoch_best_loss_in_eval)
            prompt_candidate_text = prompt_text_list[best_idx_in_eval]
            prompt_candidate_image = prompt_image_list[best_idx_in_eval]

        # --- COMMON LOGIC FOR SAVING/LOGGING ---
        if self.parallel:
            self.loss.append([l for l in loss_values_final])
            self.train_acc.append([acc for acc in train_accs_final])
            current_best_train_acc = train_accs_final[best_idx_in_eval]
        else:
            self.loss.append(loss_values_final)
            self.train_acc.append(train_acc_final)
            current_best_train_acc = train_acc_final
        
        self.best_train_acc = max(self.best_train_acc, current_best_train_acc)

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
            
            logger.info(f"*** New best {objective_type_str} ({adv_status_str} eval{attack_type_str_info}) loss found: {self.best_objective_loss_value:.4f} "
                        f"(Train Acc: {current_best_train_acc:.4f}, Best Train Acc: {self.best_train_acc:.4f}) (at call {self.num_call}) ***")

        if self.num_call > 0 and self.test_every > 0 and (self.num_call % self.test_every == 0):
            eval_loss_type_str = "adversarial" if is_current_eval_adversarial else "clean"
            obj_str = "maximize" if self.maximize_loss else "minimize"
            attack_gen_type_str = f"(AttackType: {self.adv_train_attack_type}, AttackGen: {self.adv_train_attack_prompt_type}"
            if is_current_eval_adversarial and self.adv_train_config.get('sample_ratio', 1.0) < 1.0:
                attack_gen_type_str += f", SampleRatio: {self.adv_train_config.get('sample_ratio', 1.0)}"
            attack_gen_type_str += ")" if is_current_eval_adversarial else ""

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
            adv_train_attack_type_str_fn = f"_advAttackType{self.adv_train_attack_type}" if self.adv_train_config["enabled"] else ""
            adv_train_attack_prompt_type_str_fn = f"_advPromptGen{self.adv_train_attack_prompt_type}" if self.adv_train_config["enabled"] else ""
            adv_train_sample_ratio_str_fn = f"_advSampleRatio{self.adv_train_config.get('sample_ratio', 1.0)}" if self.adv_train_config["enabled"] and self.adv_train_config.get('sample_ratio', 1.0) < 1.0 else ""
            
            fname = "{}{}_{}_{}_parallel{}_advTrain{}{}{}{}_pgdTest{}_pgdOrg{}_maxLoss{}.pth".format(
                self.k_shot, self.task_name, self.opt_name, self.backbone.replace("/","-"),
                self.parallel, self.adv_train_config["enabled"],
                adv_train_attack_type_str_fn, adv_train_attack_prompt_type_str_fn, adv_train_sample_ratio_str_fn,
                self.pgd_config["enabled"], self.pgd_config.get("original_prompt", False), self.maximize_loss)

            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                    "best_accuracy":self.best_accuracy, "acc":self.acc,
                    "best_train_accuracy": self.best_train_acc, "train_acc_history": self.train_acc,
                    "best_accuracy_pgd": self.best_accuracy_pgd, "acc_pgd": self.acc_pgd,
                    "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                    "historical_losses":self.loss, "best_objective_loss_value": self.best_objective_loss_value,
                    "maximize_loss_setting": self.maximize_loss, "num_call":self.num_call,
                    "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                    "pgd_config_test": self.pgd_config, "adv_train_config": self.adv_train_config}
            Analysis_Util.save_results(content,output_dir,fname)

        if self.parallel:
            return_value = [l * -1 if self.maximize_loss else l for l in loss_values_final]
        else:
            return_value = loss_values_final * -1 if self.maximize_loss else loss_values_final
        return return_value

    def _run_adversarial_attack(self, images, labels, text_features_for_attack, image_prompt, config, text_prompt_to_perturb=None, is_parallel=False):
        attack_type = self.adv_train_attack_type if "attack_type" in config else self.adv_train_attack_type
        images_orig = images.clone().detach()

        if attack_type == "gaussian":
            epsilon = config['epsilon']
            noise = torch.randn_like(images_orig, device=self.device) * epsilon
            final_perturbed_image = (images_orig + noise).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)
            return final_perturbed_image, None

        elif attack_type == "pgd":
            epsilon = config['epsilon']
            alpha_img = config['alpha']
            num_iter = config['num_iter']

            delta_img = torch.zeros_like(images_orig, requires_grad=True, device=self.device).to(images_orig.dtype)
            delta_img.data.uniform_(-epsilon, epsilon)
            delta_img.data = torch.clamp(images_orig + delta_img.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images_orig
            
            for iter_idx in range(num_iter):
                perturbed_image = (images_orig + delta_img).to(self.dtype)
                perturbed_image.requires_grad_(True)
                
                # --- Non-Parallel Path ---
                if not is_parallel:
                    image_features = self.image_encoder(perturbed_image, image_prompt)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits = self.logit_scale.exp() * image_features @ text_features_for_attack.t()
                    loss = F.cross_entropy(logits, labels)
                # --- Parallel Path ---
                else:
                    P = self.popsize
                    N_adv = images.shape[0] // P
                    
                    image_features = self.image_encoder(perturbed_image, image_prompt) # [P*N_adv, D]
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    image_features_bmm = image_features.view(P, N_adv, -1) # [P, N_adv, D]
                    text_features_bmm = text_features_for_attack.transpose(1, 2) # [P, D, C]
                    
                    logits = self.logit_scale.exp() * torch.bmm(image_features_bmm, text_features_bmm) # [P, N_adv, C]
                    logits_flat = logits.view(P * N_adv, self.n_cls)
                    loss = F.cross_entropy(logits_flat, labels)
                
                loss.backward()
                
                grad_sign_img = perturbed_image.grad.sign()
                delta_img.data = delta_img.data + alpha_img * grad_sign_img.to(delta_img.dtype)
                delta_img.data = torch.clamp(delta_img.data, -epsilon, epsilon)
                delta_img.data = torch.clamp(images_orig + delta_img.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images_orig
                delta_img.data = delta_img.data.to(images_orig.dtype)

            final_perturbed_image = (images_orig + delta_img.detach()).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)
            return final_perturbed_image, None
        else:
            raise ValueError(f"Unsupported adversarial attack type: {attack_type}")

    @torch.no_grad()
    def test(self, attack_config=None):
        if self.best_prompt_text is None or self.best_prompt_image is None:
            if attack_config is not None and attack_config.get("enabled", False) and self.pgd_config.get("original_prompt", False):
                 pass
            else:
                logger.warning("Test skipped: no best tuned prompt available for evaluation.")
                return torch.tensor(0.0)

        correct, total = 0., 0.
        original_text_encoder_parallel = self.text_encoder.parallel
        original_image_encoder_parallel = self.image_encoder.parallel
        self.text_encoder.parallel, self.image_encoder.parallel = False, False

        desc, is_attack_test = "Testing Clean", False
        current_text_features_for_test, current_image_prompt_for_test = None, None

        if attack_config and attack_config.get("enabled", False):
            is_attack_test = True
            desc = f"Testing {attack_config.get('attack_type', 'PGD').upper()}"
            if self.pgd_config.get("original_prompt", False):
                desc += " (Original Prompts)"
                current_text_features_for_test = self.get_original_text_features()
            else:
                if self.best_prompt_text is None:
                    logger.warning("Tuned PGD test skipped as best prompts are not available.")
                    return torch.tensor(0.0)
                current_text_features_for_test = self.text_encoder(self.best_prompt_text)
                current_text_features_for_test = current_text_features_for_test / current_text_features_for_test.norm(dim=-1,keepdim=True)
                current_image_prompt_for_test = self.best_prompt_image
        else:
            if self.best_prompt_text is None:
                logger.warning("Clean test skipped as best prompts are not available.")
                return torch.tensor(0.0)
            current_text_features_for_test = self.text_encoder(self.best_prompt_text)
            current_text_features_for_test = current_text_features_for_test / current_text_features_for_test.norm(dim=-1,keepdim=True)
            current_image_prompt_for_test = self.best_prompt_image

        for batch in self.test_loader:
            temp_original_class_parallel_attr = self.parallel
            self.parallel = False
            image,label = self.parse_batch(batch)
            self.parallel = temp_original_class_parallel_attr

            total += image.size(0)
            eval_image = image.to(self.dtype)

            if is_attack_test:
                with torch.enable_grad():
                    eval_image, _ = self._run_adversarial_attack( 
                        image, label, current_text_features_for_test, 
                        current_image_prompt_for_test, attack_config, 
                        is_parallel=False)
            
            image_features = self.image_encoder(eval_image, current_image_prompt_for_test)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)
            logits = self.logit_scale.exp() * image_features @ current_text_features_for_test.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        self.text_encoder.parallel = original_text_encoder_parallel
        self.image_encoder.parallel = original_image_encoder_parallel
        return correct / total

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
        elif self.task_name == 'CIFAR10_PGD': 
            self.train_data,self.train_loader = load_train_cifar10_pgd(batch_size=self.batch_size,shots=self.k_shot, seed=self.seed)
            if self.pgd_config["enabled"]: 
                self.test_data, self.test_loader = load_test_cifar10_pgd(batch_size=self.batch_size)
            else: 
                self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess)
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.n_cls = len(self.classes)
        else: 
            train_path = os.path.join(self.data_dir, self.task_name)
            test_path = os.path.join(self.data_dir, self.task_name)
            if self.task_name == 'ImageNet':
                train_path = os.path.join(self.data_dir, "imagenet")
                test_path = os.path.join(self.data_dir, "imagenet")

            self.train_data, self.train_loader = load_train(batch_size=self.batch_size, shots=self.k_shot, preprocess=self.preprocess, root=train_path, dataset_dir=self.task_name, seed=self.seed)
            self.test_data, self.test_loader = load_test(batch_size=self.batch_size, preprocess=self.preprocess, root=test_path, dataset_dir=self.task_name)
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)

    def parse_batch(self,batch):
        image, label = batch["image"], batch["label"]
        image = image.to(device=self.device, dtype=self.dtype if image.dtype != torch.uint8 else torch.float32)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        label = label.to(device=self.device)
        
        if self.parallel: 
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label