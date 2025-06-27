import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
import random
from collections import defaultdict
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

# Helper class for the dynamically generated pre-attacked dataset
class PreAttackedDictDataset(torch.utils.data.Dataset):
    def __init__(self, items, classes):
        self.items = items # List of dictionaries {"image": tensor, "label": tensor}
        self.classes = classes

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

# Helper class to combine clean and attacked data based on a ratio
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, attacked_dataset, ratio):
        self.clean_dataset = clean_dataset
        self.attacked_dataset = attacked_dataset
        self.ratio = ratio
        self.classes = clean_dataset.classes
        
        logger.info(f"Building combined dataset with {ratio*100:.0f}% attacked data (per-class)...")
        self.combined_items = self._build_combined_data()
        logger.info(f"Combined dataset created with {len(self.combined_items)} total samples.")

    def _build_combined_data(self):
        clean_by_class = defaultdict(list)
        attacked_by_class = defaultdict(list)

        for i in range(len(self.clean_dataset)):
            item = self.clean_dataset[i]
            clean_by_class[item['label'].item()].append(item)
        
        for i in range(len(self.attacked_dataset)):
            item = self.attacked_dataset[i]
            attacked_by_class[item['label'].item()].append(item)

        final_items = []
        for label in sorted(clean_by_class.keys()):
            clean_samples = clean_by_class[label]
            attacked_samples = attacked_by_class[label]
            
            num_in_class = len(clean_samples)
            num_attacked = round(num_in_class * self.ratio)
            num_clean = num_in_class - num_attacked

            # To ensure we get a random subset, shuffle before picking
            random.shuffle(clean_samples)
            random.shuffle(attacked_samples)

            final_items.extend(attacked_samples[:num_attacked])
            final_items.extend(clean_samples[:num_clean])

        random.shuffle(final_items) # Shuffle the final list to mix classes
        return final_items

    def __getitem__(self, index):
        return self.combined_items[index]

    def __len__(self):
        return len(self.combined_items)


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
        self.loss = []
        self.acc = []
        self.train_acc = []
        self.pre_attack_gen_config = cfg.get("pre_attack_gen", {"enabled": False})
        self._training_dataset_snapshot = None # Holder for the dataset

        # --- 1. Initialize Model Components FIRST ---
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

        # Other model parts
        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0.0
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

        # --- 2. Load Dataset (this sets self.classes) ---
        self.load_dataset()
        self._capture_training_dataset()

        # --- 3. Set Encoder Context (NOW that self.classes is available) ---
        # This is the crucial change: set context before any other method might need it.
        logger.info("Setting encoder contexts...")
        self.text_encoder.set_context(self.get_text_information())
        self.image_encoder.set_context(self.get_image_information())


        # --- 4. Final Setup ---
        self.maximize_loss = cfg.get("maximize_loss", False)
        self.best_objective_loss_value = None
        if self.maximize_loss:
            self.best_objective_loss_value = -float('inf')
            logger.info(f"--- Prompt Optimization Mode: MAXIMIZE Loss (Targeting value: {self.best_objective_loss_value}) ---")
            if self.pre_attack_gen_config.get("enabled", False):
                 logger.warning("Using MAXIMIZE loss on a pre-attacked dataset. This may lead to unexpected behavior (e.g., learning to misclassify even more).")
        else:
            self.best_objective_loss_value = float('inf')
            logger.info(f"--- Prompt Optimization Mode: MINIMIZE Loss (Targeting value: {self.best_objective_loss_value}) ---")

        if self.pre_attack_gen_config.get("enabled", False):
            ratio = self.pre_attack_gen_config.get('ratio', 1.0)
            logger.info(f"--- Pre-attacked Dataset Generation was ENABLED ({ratio*100:.0f}% Attacked) ---")
            logger.info(f"  Generation Params: Epsilon={self.pre_attack_gen_config['epsilon']}, Alpha={self.pre_attack_gen_config['alpha']}, Iter={self.pre_attack_gen_config['num_iter']}")
            logger.info("--- On-the-fly adversarial attacks are DISABLED. Using the pre-generated dataset for all evaluations. ---")
        else:
            logger.info("--- Standard (Clean) Prompt Optimization ---")


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
        
        # Use the underlying dataset to avoid dataloader shuffling issues
        dataset_to_capture = self.train_loader.dataset
        for i in range(len(dataset_to_capture)):
            item = dataset_to_capture[i]
            all_images.append(item["image"].cpu().unsqueeze(0))
            all_labels.append(item["label"].cpu().unsqueeze(0))

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
                
                if self.learned_prompt_pos == "prefix":
                    template = f"{prompt_prefix_placeholder} {initial_prompt} {name}."
                elif self.learned_prompt_pos == "middle":
                    template = f"{initial_prompt} {prompt_prefix_placeholder} {name}."
                elif self.learned_prompt_pos == "suffix":
                    template = f"{initial_prompt} {name} {prompt_prefix_placeholder}."
                else: # Default to prefix
                    template = f"{prompt_prefix_placeholder} {initial_prompt} {name}."

                pattern_prompts.append(" ".join(template.split()))

            tokenized_pattern_prompts = torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            
            x_token_id = clip.tokenize("X")[0, 1].item()
            ctx_start_idx = (tokenized_pattern_prompts == x_token_id).nonzero(as_tuple=True)[1].min().item()

            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            
            context = {
                "n_cls": self.n_cls, 
                "n_prompt_tokens_L": self.n_prompt_tokens_L,
                "init_pattern_embedding": init_pattern_embedding, 
                "tokenized_pattern_prompts": tokenized_pattern_prompts,
                "ctx_start_idx": ctx_start_idx,
                "batch_size": self.batch_size,
                "pop_size": self.popsize,
                "parallel": self.parallel
            }
        else:
            pattern_prompt = prompt_prefix_placeholder + " " + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            ctx_start_idx = 1
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
        self.num_call += 1
        
        loss_accumulator = 0
        logit_scale = self.logit_scale.exp()

        if self.parallel:
            loss_accumulator = [0.0] * self.popsize
            all_pop_text_features = []
            for p_text in prompt_text_list_or_tensor:
                features = self.text_encoder(p_text)
                features = features / features.norm(dim=-1, keepdim=True)
                all_pop_text_features.append(features)
            pop_txt_features = torch.stack(all_pop_text_features)
        else:
            text_features = self.text_encoder(prompt_text_list_or_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            current_prompt_image_for_eval_single = prompt_image_list_or_tensor

        for batch_idx, batch in enumerate(self.train_loader):
            images_orig, labels_orig = self.parse_batch(batch)

            if self.parallel:
                B_actual_batch = labels_orig.shape[0]
                pop_images_batch = images_orig.view(self.popsize, B_actual_batch, *images_orig.shape[1:])

                for i in range(self.popsize):
                    eval_image_i = pop_images_batch[i].to(self.dtype)
                    img_prompt_i = prompt_image_list_or_tensor[i]
                    txt_features_i = pop_txt_features[i]
                    
                    image_features_i = self.image_encoder(eval_image_i, img_prompt_i)
                    image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
                    tmp_logits = logit_scale * image_features_i @ txt_features_i.t()
                    loss_for_member_batch_i = self.metric(tmp_logits, labels_orig)
                    loss_accumulator[i] += loss_for_member_batch_i.item()

            else:
                eval_image = images_orig.to(self.dtype)
                image_features = self.image_encoder(eval_image, current_prompt_image_for_eval_single)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()
                loss_for_candidate_batch = self.metric(logits, labels_orig)
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
            ratio = self.pre_attack_gen_config.get('ratio', 0) if self.pre_attack_gen_config.get('enabled') else 0
            dataset_type_str = f"{ratio*100:.0f}% attacked"
            logger.info(f"*** New best {objective_type_str} ({dataset_type_str} eval) loss found: {self.best_objective_loss_value:.4f} (at call {self.num_call}) ***")

        if self.test_every_gens is not None and self.test_every_gens > 0:
            trigger_interval = self.test_every_gens if self.parallel else self.test_every_gens * self.popsize
            
            if self.num_call > 0 and (self.num_call % trigger_interval == 0):
                current_generation = self.num_call if self.parallel else self.num_call // self.popsize
                
                acc_train_current = self.test_on_train_set()
                self.train_acc.append(acc_train_current.item())
                self.best_train_accuracy = acc_train_current.item()
                
                ratio = self.pre_attack_gen_config.get('ratio', 0) if self.pre_attack_gen_config.get('enabled') else 0
                dataset_type_str = f"{ratio*100:.0f}% Attacked"
                logger.info(f"\n--- Intermediate Test at Generation ~{current_generation} (on {dataset_type_str} Data) ---")
                acc_test = self.test()
                self.acc.append(acc_test.item())
                self.best_accuracy = max(acc_test.item(), self.best_accuracy)
                logger.info(f"Train Accuracy: {self.best_train_accuracy:.4f}")
                logger.info(f"Test Accuracy: {acc_test:.4f} (Best Test: {self.best_accuracy:.4f})")
                
                output_dir = os.path.join(self.output_dir,self.task_name)
                initial_prompt_str_fn = f"_initPrompt" if self.initial_prompt_text is not None else ""
                learned_pos_str_fn = f"_pos{self.learned_prompt_pos}"
                pre_attack_gen_ratio_str_fn = f"_ratio{self.pre_attack_gen_config['ratio']}" if self.pre_attack_gen_config['enabled'] and self.pre_attack_gen_config['ratio'] < 1.0 else ""
                pre_attack_gen_str_fn = f"_preAttackGen{pre_attack_gen_ratio_str_fn}" if self.pre_attack_gen_config['enabled'] else ""

                fname = "{}{}_{}_{}_parallel{}{}{}_maxLoss{}.pth".format(
                    self.k_shot, self.task_name, initial_prompt_str_fn, learned_pos_str_fn,
                    self.opt_name, self.backbone.replace("/", "-"),
                    self.parallel,
                    pre_attack_gen_str_fn,
                    self.maximize_loss
                )

                content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                        "best_accuracy":self.best_accuracy, "acc":self.acc,
                        "best_train_accuracy": self.best_train_accuracy, "train_acc": self.train_acc,
                        "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                        "training_dataset_snapshot": self._training_dataset_snapshot,
                        "historical_losses":self.loss,
                        "best_objective_loss_value": self.best_objective_loss_value,
                        "maximize_loss_setting": self.maximize_loss,
                        "num_call":self.num_call,
                        "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                        "pre_attack_gen_config": self.pre_attack_gen_config}
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
    def test(self):
        if self.best_prompt_text is None or self.best_prompt_image is None:
            logger.warning("Test skipped: no best tuned prompt available for evaluation.")
            return torch.tensor(0.0)

        correct = 0.
        total = 0.
        
        original_text_encoder_parallel = self.text_encoder.parallel
        original_image_encoder_parallel = self.image_encoder.parallel
        self.text_encoder.parallel = False
        self.image_encoder.parallel = False

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

    def _generate_pre_attacked_dataset(self, clean_train_loader, clean_test_loader, gen_config):
        logger.info("--- Starting Pre-attacked Dataset Generation ---")
        logger.info(f"Generation Config: Epsilon={gen_config['epsilon']}, Alpha={gen_config['alpha']}, Iter={gen_config['num_iter']}")

        mean = self.preprocess.transforms[-1].mean
        std = self.preprocess.transforms[-1].std
        norm_mean = torch.tensor(mean).to(self.device).view(3, 1, 1)
        norm_std = torch.tensor(std).to(self.device).view(3, 1, 1)
        norm_upper_limit = ((1 - norm_mean) / norm_std).to(self.device)
        norm_lower_limit = ((0 - norm_mean) / norm_std).to(self.device)

        with torch.no_grad():
            guidance_text_features = self.get_original_text_features()

        original_main_parallel_flag = self.parallel
        original_encoder_parallel_flag = self.image_encoder.parallel
        self.parallel = False
        self.image_encoder.parallel = False

        def _run_pgd_attack(images, labels):
            images_orig = images.clone().detach()
            epsilon = gen_config['epsilon']
            alpha = gen_config['alpha']
            num_iter = gen_config['num_iter']
            
            delta_img = torch.zeros_like(images_orig, requires_grad=True, device=self.device).to(images_orig.dtype)
            delta_img.data.uniform_(-epsilon, epsilon)
            delta_img.data = torch.clamp(images_orig + delta_img.data, min=norm_lower_limit, max=norm_upper_limit) - images_orig
            delta_img.data = delta_img.data.to(images_orig.dtype)

            for _ in range(num_iter):
                delta_img.requires_grad_(True)
                perturbed_image = (images_orig + delta_img).to(self.dtype)

                image_features = self.image_encoder(perturbed_image, None)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = self.logit_scale.exp() * image_features @ guidance_text_features.t()
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                
                grad_sign_img = delta_img.grad.sign()
                delta_img.data = delta_img.data + alpha * grad_sign_img.to(delta_img.dtype)
                delta_img.data = torch.clamp(delta_img.data, -epsilon, epsilon)
                delta_img.data = torch.clamp(images_orig + delta_img.data, min=norm_lower_limit, max=norm_upper_limit) - images_orig
                delta_img.grad.zero_()

            return (images_orig + delta_img.detach()).clamp(min=norm_lower_limit, max=norm_upper_limit).to(self.dtype)

        def _process_loader(loader, desc):
            attacked_items = []
            logger.info(f"Generating attacked {desc} set...")
            for batch in tqdm(loader, desc=f"Attacking {desc} Set"):
                images, labels = self.parse_batch(batch)

                with torch.enable_grad():
                    attacked_images = _run_pgd_attack(images, labels)
                
                for i in range(attacked_images.size(0)):
                    attacked_items.append({
                        "image": attacked_images[i].cpu(),
                        "label": labels[i].cpu()
                    })

            return attacked_items

        attacked_train_items = _process_loader(clean_train_loader, "train")
        attacked_test_items = _process_loader(clean_test_loader, "test")

        self.parallel = original_main_parallel_flag
        self.image_encoder.parallel = original_encoder_parallel_flag

        attacked_train_dataset = PreAttackedDictDataset(attacked_train_items, self.classes)
        attacked_test_dataset = PreAttackedDictDataset(attacked_test_items, self.classes)

        logger.info("--- Pre-attacked Dataset Generation Finished ---")
        return attacked_train_dataset, attacked_test_dataset

    def load_dataset(self):
        is_pre_attack_gen_task = self.pre_attack_gen_config.get("enabled", False)
        ratio = self.pre_attack_gen_config.get("ratio", 1.0)

        if is_pre_attack_gen_task:
            if not self.task_name.endswith("_PGD_GEN"):
                raise ValueError(f"Task '{self.task_name}' is invalid for pre-attack generation. Name must end with _PGD_GEN.")
            
            base_task_name = self.task_name.replace("_PGD_GEN", "")
            logger.info(f"Pre-attack generation enabled. Base dataset: '{base_task_name}'.")

            self._load_specific_dataset(base_task_name)
            
            if ratio == 0.0:
                logger.warning("Pre-attack generation was enabled, but ratio is 0.0. Using a fully CLEAN dataset.")
            elif ratio == 1.0:
                logger.info("Ratio is 1.0. Generating a fully ATTACKED dataset.")
                attacked_train_data, attacked_test_data = self._generate_pre_attacked_dataset(self.train_loader, self.test_loader, self.pre_attack_gen_config)
                self.train_data = attacked_train_data
                self.test_data = attacked_test_data
                self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
                self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
            elif 0.0 < ratio < 1.0:
                logger.info(f"Ratio is {ratio}. Generating a MIXED dataset.")
                clean_train_data, clean_test_data = self.train_data, self.test_data
                attacked_train_data, attacked_test_data = self._generate_pre_attacked_dataset(self.train_loader, self.test_loader, self.pre_attack_gen_config)
                
                self.train_data = CombinedDataset(clean_train_data, attacked_train_data, ratio)
                self.test_data = CombinedDataset(clean_test_data, attacked_test_data, ratio)

                self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
                self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
            else:
                 raise ValueError(f"pre_attack_gen_ratio must be between 0.0 and 1.0, but got {ratio}")

        else:
            self._load_specific_dataset(self.task_name)

    def _load_specific_dataset(self, task_name):
        if task_name == 'CIFAR100':
            self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed)
            self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)
        elif task_name == 'CIFAR10':
            self.dataset = CIFAR10(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar10(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess, seed=self.seed)
            self.test_data, self.test_loader = load_test_cifar10(batch_size=self.batch_size, preprocess=self.preprocess)
        elif task_name == 'CIFAR10_PGD':
            self.train_data,self.train_loader = load_train_cifar10_pgd(batch_size=self.batch_size,shots=self.k_shot)
            self.test_data, self.test_loader = load_test_cifar10_pgd(batch_size=self.batch_size)
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.n_cls = len(self.classes)

        elif task_name == 'StanfordCars':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'OxfordPets':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'UCF-101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'DTD':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'EuroSAT':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'Food101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'caltech101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'SUN397':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif task_name == 'ImageNet':
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
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        label = label.to(device=self.device)
        
        if self.parallel: 
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label