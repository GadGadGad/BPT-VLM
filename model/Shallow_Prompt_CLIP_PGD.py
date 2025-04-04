import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchvision.datasets import CIFAR100, CIFAR10
# Assuming these imports are correct for your project structure
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from dataset.cifar10 import load_train_cifar10, load_test_cifar10
from model.shallow_encoder import TextEncoder,VisionEncoder
from model.analysis_utils import Analysis_Util
from dataset.general import load_train,load_test
from tqdm import tqdm

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
        self.load_dataset()
        self.loss = []
        self.acc = []
        self.acc_pgd = []

        # --- Adversarial Training Configuration ---
        self.adv_train_config = cfg.get("adv_train", {"enabled": False})
        if self.adv_train_config["enabled"]:
            print("--- Adversarial Prompt Optimization ENABLED ---")
            if "epsilon" not in self.adv_train_config: self.adv_train_config["epsilon"] = 4/255
            if "alpha" not in self.adv_train_config: self.adv_train_config["alpha"] = 1/255
            if "num_iter" not in self.adv_train_config: self.adv_train_config["num_iter"] = 5 # Fewer steps for training speed
            print(f"  Training PGD Config: Epsilon={self.adv_train_config['epsilon']}, Alpha={self.adv_train_config['alpha']}, Iter={self.adv_train_config['num_iter']}")
        else:
            print("--- Standard (Clean) Prompt Optimization ---")
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
        self.best_accuracy = 0
        self.best_accuracy_pgd = 0
        self.min_loss = None
        self.loss = []
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
        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
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
        print('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)

        self.pgd_config = cfg.get("pgd", {"enabled": False}) # Get PGD config for TESTING
        if self.pgd_config["enabled"] or self.adv_train_config["enabled"]:
             # Need normalization stats if either training or testing uses PGD
            print("PGD Attack Enabled (Testing or Training).")
            try:
                mean = self.preprocess.transforms[-1].mean
                std = self.preprocess.transforms[-1].std
                self.norm_mean = torch.tensor(mean).to(self.device).view(3, 1, 1)
                self.norm_std = torch.tensor(std).to(self.device).view(3, 1, 1)
                self.norm_upper_limit = ((1 - self.norm_mean) / self.norm_std).to(self.device)
                self.norm_lower_limit = ((0 - self.norm_mean) / self.norm_std).to(self.device)
                print(f"  Test PGD Config: Epsilon={self.pgd_config.get('epsilon', 'N/A')}, Alpha={self.pgd_config.get('alpha', 'N/A')}, Iter={self.pgd_config.get('num_iter', 'N/A')}")
            except Exception as e:
                 print(f"Warning: Could not extract mean/std from preprocess for PGD clipping. PGD might not work correctly. Disabling PGD. Error: {e}")
                 self.pgd_config["enabled"] = False
                 self.adv_train_config["enabled"] = False # Disable if we can't get stats


    def get_text_information(self,caption=None):
        # (No changes needed here)
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

    def get_image_information(self):
        # (No changes needed here)
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context

    def generate_text_prompts(self,intrinsic_vectors):
        # (No changes needed here)
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, -1)
            if self.init_prompt is not None:
                z = z + self.init_prompt  # Az + p_0

            prompt_list.append(z)
        return prompt_list

    def generate_visual_prompts(self,intrinsic_vectors):
        # (No changes needed here)
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V,-1)
            #z = z + self.position_V
            visual_prompt_list.append(z)

        return visual_prompt_list

    def metric(self,logits,label):
        # (No changes needed here)
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

    # MODIFIED: eval now calculates loss potentially using adversarial examples
    # Note: @torch.no_grad() is kept, but PGD attack itself will use torch.enable_grad()
    @torch.no_grad()
    def eval(self,prompt_zip):
        prompt_text_list, prompt_image_list = prompt_zip[0], prompt_zip[1] # These are lists if parallel, tensors otherwise
        self.num_call += 1
        loss = 0
        logit_scale = self.logit_scale.exp()

        # Pre-compute text features for all prompts in the population/batch
        if self.parallel:
            # text_features: [popsize * n_cls, D]
            text_features = self.text_encoder(prompt_text_list)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Reshape for easier access: [popsize, n_cls, D]
            pop_txt_features = text_features.view(self.popsize, self.n_cls, -1)
            loss = [0.0] * self.popsize # Initialize list for parallel losses
        else:
            # text_features: [n_cls, D]
            text_features = self.text_encoder(prompt_text_list)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Store single prompts for potential best update
            current_prompt_text = prompt_text_list
            current_prompt_image = prompt_image_list


        for batch in self.train_loader:
            # Get clean batch data
            clean_image, label = self.parse_batch(batch) # parse_batch handles device/dtype/repeat
            # clean_image shape [B*pop, C, H, W] if parallel else [B, C, H, W]
            # label shape [B] (original batch size)

            if self.parallel:
                B = label.shape[0] # Original batch size
                # Reshape clean images for easier slicing if needed: [popsize, B, C, H, W]
                pop_clean_image = clean_image.view(self.popsize, B, *clean_image.shape[1:])

                # --- Parallel Adversarial Attack & Loss Calculation ---
                for i in range(self.popsize):
                    current_txt_features = pop_txt_features[i] # [n_cls, D]
                    current_img_prompt = prompt_image_list[i] # [n_prompt_tok_V, D_V]
                    current_clean_images = pop_clean_image[i] # [B, C, H, W]

                    # Determine image input for this population member
                    if self.adv_train_config["enabled"]:
                        # Generate adversarial examples for THIS prompt pair (i)
                        # Requires gradients for the attack calculation
                        with torch.enable_grad():
                            eval_image_i = self._pgd_attack(
                                images=current_clean_images,
                                labels=label,
                                text_features=current_txt_features,
                                image_prompt=current_img_prompt,
                                config=self.adv_train_config # Use training PGD config
                            )
                        # Ensure correct dtype after attack
                        eval_image_i = eval_image_i.to(self.dtype)
                    else:
                        # Use clean images if adversarial training is disabled
                        eval_image_i = current_clean_images.to(self.dtype)

                    # Compute image features (no grad needed here)
                    # Need to run image_encoder non-parallel for single prompt eval
                    original_im_parallel = self.image_encoder.parallel
                    self.image_encoder.parallel = False
                    image_features_i = self.image_encoder(eval_image_i, current_img_prompt)
                    self.image_encoder.parallel = original_im_parallel # Restore state

                    image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True) # [B, D]

                    # Calculate logits and loss for this population member
                    tmp_logits = logit_scale * image_features_i @ current_txt_features.t() # [B, n_cls]
                    loss[i] += self.metric(tmp_logits, label).item() # Accumulate loss, ensure it's float

            else: # --- Non-Parallel Adversarial Attack & Loss Calculation ---
                current_clean_images = clean_image # [B, C, H, W]

                # Determine image input
                if self.adv_train_config["enabled"]:
                     # Generate adversarial examples for the single prompt pair
                     with torch.enable_grad():
                         eval_image = self._pgd_attack(
                            images=current_clean_images,
                            labels=label,
                            text_features=text_features, # Already [n_cls, D]
                            image_prompt=current_prompt_image, # Already [n_tok, D]
                            config=self.adv_train_config
                         )
                     eval_image = eval_image.to(self.dtype)
                else:
                     eval_image = current_clean_images.to(self.dtype)

                # Compute image features (no grad needed here)
                image_features = self.image_encoder(eval_image, current_prompt_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [B, D]

                # Calculate logits and loss
                logits = logit_scale * image_features @ text_features.t() # [B, n_cls]
                loss += self.metric(logits, label).item() # Accumulate loss

        # --- Loss Finalization & Best Prompt Update ---
        epoch_min_loss = float('inf')
        best_idx_in_batch = -1

        if self.parallel:
            # Normalize loss by dataset size
            loss = [x / len(self.train_data) for x in loss]
            epoch_min_loss = min(loss)
            best_idx_in_batch = loss.index(epoch_min_loss)
            # Store the prompts corresponding to the minimum loss in this parallel batch
            current_prompt_text = prompt_text_list[best_idx_in_batch]
            current_prompt_image = prompt_image_list[best_idx_in_batch]
        else:
            loss /= len(self.train_data)
            epoch_min_loss = loss
            best_idx_in_batch = 0 # Only one prompt if not parallel

        self.loss.append(loss) # Appends list if parallel, float otherwise

        # Update best prompt based on minimum loss observed *so far*
        if self.min_loss is None or epoch_min_loss < self.min_loss:
            self.min_loss = epoch_min_loss
            # Detach and clone the best prompts from the current evaluation
            self.best_prompt_text = current_prompt_text.detach().clone()
            self.best_prompt_image = current_prompt_image.detach().clone()
            print(f"*** New best {'adversarial' if self.adv_train_config['enabled'] else 'clean'} loss found: {self.min_loss:.4f} (at call {self.num_call}) ***")


        # --- Periodic Testing ---
        if self.num_call % self.test_every == 0:
            print(f"\n--- Testing at call {self.num_call} (Prompts optimized with {'adversarial' if self.adv_train_config['enabled'] else 'clean'} loss) ---")
            # Test clean accuracy
            acc_clean = self.test(attack_config=None)
            self.acc.append(acc_clean.item()) # Store clean accuracy
            self.best_accuracy = max(acc_clean.item(), self.best_accuracy)
            print(f"Clean Accuracy: {acc_clean:.4f} (Best Clean: {self.best_accuracy:.4f})")

            # Test PGD accuracy if evaluation PGD is enabled
            acc_attacked = torch.tensor(0.0) # Default to 0 if not enabled
            if self.pgd_config["enabled"] and self.best_prompt_text is not None:
                # Use the TEST pgd_config here
                acc_attacked = self.test(attack_config=self.pgd_config)
                self.acc_pgd.append(acc_attacked.item()) # Store PGD accuracy
                self.best_accuracy_pgd = max(acc_attacked.item(), self.best_accuracy_pgd)
                print(f"PGD Accuracy (Test): {acc_attacked:.4f} (Best PGD  : {self.best_accuracy_pgd:.4f})")
            elif self.pgd_config["enabled"]:
                 print("PGD Accuracy (Test): Skipped (no best prompt yet)")
            elif not self.pgd_config["enabled"]:
                 print("PGD Accuracy (Test): Disabled in config")


            #---------------save_results-----------------------------------
            output_dir = os.path.join(self.output_dir,self.task_name)
            fname = "{}_{}_{}_advOpt{}.pth".format(
                self.task_name, self.opt_name, self.backbone.replace("/","-"),
                self.adv_train_config["enabled"] # Add flag to filename
            )

            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                       "best_accuracy":self.best_accuracy, "acc":self.acc,
                       "best_accuracy_pgd": self.best_accuracy_pgd, "acc_pgd": self.acc_pgd, # Save PGD results
                       "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                       "loss":self.loss,"num_call":self.num_call,
                       "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                       "pgd_config_test": self.pgd_config, # Save TEST PGD config used
                       "adv_train_config": self.adv_train_config} # Save ADV TRAIN config used
            Analysis_Util.save_results(content,output_dir,fname)
            # ---------------save_results-----------------------------------

        # Return the loss (list if parallel, float otherwise) to the optimizer
        return loss


    def _pgd_attack(self, images, labels, text_features, image_prompt, config):
        """ Performs PGD attack - NO CHANGES NEEDED HERE conceptually, but ensure inputs are correct """
        # Ensure images require grad only if they don't already (e.g., first iter)
        # Make a clone to avoid modifying original images if they are used elsewhere
        images = images.clone().detach()
        labels = labels.clone().detach() # Ensure labels are detached

        epsilon = config['epsilon']
        alpha = config['alpha']
        num_iter = config['num_iter']

        # Ensure delta is same dtype and device as images from the start
        delta = torch.zeros_like(images, requires_grad=True, device=self.device).to(images.dtype)
        delta.data.uniform_(-epsilon, epsilon)
        # Project initial perturbation to valid range *relative to original image*
        delta.data = torch.clamp(images + delta.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images
        delta.data = delta.data.to(images.dtype)

        for _ in range(num_iter):
            delta.requires_grad_(True)
            perturbed_image = (images + delta).to(self.dtype) # Ensure dtype for encoder

            # --- Forward pass for attack gradient ---
            # Temporarily disable parallel in image encoder if it's enabled globally
            # This is crucial because we're attacking for a *specific* prompt here.
            original_im_parallel = self.image_encoder.parallel
            self.image_encoder.parallel = False
            # Need to handle potential list of prompts if called from parallel eval
            # No, image_prompt passed here is already the specific one for this attack instance
            image_features = self.image_encoder(perturbed_image, image_prompt)
            self.image_encoder.parallel = original_im_parallel # Restore state

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ text_features.t()
            # --------------------------------------

            loss = F.cross_entropy(logits, labels)

            # --- Check for NaNs in loss ---
            if torch.isnan(loss):
                print("Warning: NaN loss detected during PGD attack. Stopping attack for this batch.")
                # Return the image perturbed up to the previous step, or original if first step
                return (images + delta.detach()).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)
            # -------------------------------

            delta_grad = torch.autograd.grad(loss, delta,
                                             only_inputs=True, # Ensure only delta grad is computed
                                             retain_graph=False, # Don't need to retain graph after this
                                             create_graph=False # Don't need graph for delta_grad itself
                                             )[0] # autograd.grad returns a tuple
            if delta_grad is None:
                print(f"Warning: delta_grad is None during PGD attack iter {i}. Stopping attack for this batch.")
                # Return the image perturbed up to the previous step
                return (images + delta.detach()).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)

            grad_sign = delta_grad.sign()
            delta.data = delta.data + alpha * grad_sign.to(delta.dtype)
            # Project delta to L-infinity ball
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            # Project perturbed image to valid input space range
            delta.data = torch.clamp(images + delta.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images
            # Zero gradients for next iteration
            # delta.grad.zero_()

        # Return final perturbed image, detached and clamped
        return (images + delta.detach()).clamp(min=self.norm_lower_limit, max=self.norm_upper_limit).to(self.dtype)

    # test method remains largely the same - it uses the TEST pgd_config
    @torch.no_grad()
    def test(self, attack_config=None):
        """ Evaluate accuracy, optionally with PGD attack using TEST config """
        if self.best_prompt_text is None or self.best_prompt_image is None:
            print("Warning: Trying to test without best prompts found yet. Returning 0 accuracy.")
            return torch.tensor(0.0)

        correct = 0.
        total = 0.
        # Ensure evaluation is done sequentially (no parallelism)
        original_parallel_state = self.parallel
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False # Ensure non-parallel for testing

        # Pre-compute text features using the best text prompt
        text_features = self.text_encoder(self.best_prompt_text)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)

        desc = "Testing Clean"
        is_attack_test = False
        if attack_config and attack_config.get("enabled", False):
             # Use settings from the passed attack_config (should be self.pgd_config for standard testing)
             desc = f"Testing PGD(eps={attack_config['epsilon']}, iter={attack_config['num_iter']})"
             is_attack_test = True


        for batch in tqdm(self.test_loader, desc=desc, leave=False):
            image,label = self.parse_batch(batch) # parse_batch handles device transfer
            total += image.size(0)

            eval_image = image.to(self.dtype) # Default to clean image, ensure dtype

            # --- Apply PGD Attack if configured for this test run ---
            if is_attack_test:
                 # Enable gradients for attack step ONLY
                 with torch.enable_grad():
                      # Use the passed attack_config (e.g., self.pgd_config)
                      eval_image = self._pgd_attack(image, label, text_features, self.best_prompt_image, attack_config)
                 eval_image = eval_image.to(self.dtype) # Ensure dtype after attack
            # --- End PGD Attack ---

            # Compute image features (always no_grad here)
            image_features = self.image_encoder(eval_image, self.best_prompt_image)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale*image_features@text_features.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        # Restore original parallel state
        self.parallel = original_parallel_state
        # We might need to restore encoder parallel states if they were different
        self.text_encoder.parallel = original_parallel_state
        self.image_encoder.parallel = original_parallel_state


        acc = correct/total # Use total derived from batches
        return acc


    def load_dataset(self):
        # (No changes needed here)
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
        # Modified slightly to ensure label is not repeated when parallel
        image = batch["image"]
        label = batch["label"]
        image = image.to(device=self.device, dtype=self.dtype) # Apply dtype here
        label = label.to(device=self.device) # Labels usually int64, no dtype change
        if self.parallel:
            # Repeat image for each member of the population
            image = image.repeat(self.popsize, 1, 1, 1)
            # DO NOT repeat labels. Labels correspond to the original batch size.
        return image, label