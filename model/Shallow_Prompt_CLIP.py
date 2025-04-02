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

        self.pgd_config = cfg.get("pgd", {"enabled": False}) # Get PGD config, default to disabled
        if self.pgd_config["enabled"]:
            print("PGD Attack Testing Enabled.")
            # Precompute normalization bounds for clipping
            try:
                mean = self.preprocess.transforms[-1].mean
                std = self.preprocess.transforms[-1].std
                self.norm_mean = torch.tensor(mean).to(self.device).view(3, 1, 1)
                self.norm_std = torch.tensor(std).to(self.device).view(3, 1, 1)
                self.norm_upper_limit = ((1 - self.norm_mean) / self.norm_std)
                self.norm_lower_limit = ((0 - self.norm_mean) / self.norm_std)
                print(f"  PGD Epsilon: {self.pgd_config['epsilon']}, Alpha: {self.pgd_config['alpha']}, Iter: {self.pgd_config['num_iter']}")
            except Exception as e:
                 print(f"Warning: Could not extract mean/std from preprocess for PGD clipping. PGD might not work correctly. Error: {e}")
                 self.pgd_config["enabled"] = False # Disable if we can't get stats

    def get_text_information(self,caption=None):
        # classification task - caption - None
        # refcoco ask - caption - str
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
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context



    def generate_text_prompts(self,intrinsic_vectors):
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
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V,-1)
            #z = z + self.position_V
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
    def eval(self,prompt_zip):
        prompt_text,prompt_image = prompt_zip[0],prompt_zip[1]
        self.num_call += 1
        loss = 0
        current_prompt_text = None # Store the text prompt associated with min loss in this batch
        current_prompt_image = None # Store the image prompt associated with min loss in this batch

        if self.parallel:
            loss = [0]*self.popsize
            text_features = self.text_encoder(prompt_text) # if parallel, text_features.shape = [n_cls * popsize, *, *]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Store prompts now in case we need them later for best_prompt update
            current_prompt_text = prompt_text
            current_prompt_image = prompt_image
        else:
            # Handle single prompt case
            text_features = self.text_encoder(prompt_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            current_prompt_text = prompt_text
            current_prompt_image = prompt_image


        for batch in self.train_loader:
            image,label = self.parse_batch(batch) # image shape [B*pop, C, H, W] if parallel else [B, C, H, W]
            image_features = self.image_encoder(image,prompt_image) # prompt_image is list if parallel, tensor otherwise
            image_features = image_features / image_features.norm(dim=-1,keepdim=True) # shape [B*pop, D] or [B, D]
            logit_scale = self.logit_scale.exp()
            if self.parallel:
                B = int(image_features.shape[0]/self.popsize) # Original batch size
                pop_img_features = image_features.view(self.popsize, B, -1) # [pop, B, D]
                pop_txt_features = text_features.view(self.popsize, self.n_cls, -1) # [pop, n_cls, D]

                for i in range(self.popsize):
                    # tmp_text_features = text_features[i*self.n_cls:(i+1)*self.n_cls] # shape [n_cls, D]
                    # tmp_image_features = image_features[i*B:(i+1)*B] # shape [B, D]
                    tmp_logits = logit_scale * pop_img_features[i] @ pop_txt_features[i].t() # [B, n_cls]
                    loss[i]+=self.metric(tmp_logits,label) # label shape [B]
            else:
                logits = logit_scale*image_features@text_features.t()
                loss +=self.metric(logits,label)

        epoch_min_loss = float('inf')
        best_idx_in_batch = -1

        if self.parallel:
            loss = [x/len(self.train_data) for x in loss]
            epoch_min_loss = min(loss)
            best_idx_in_batch = loss.index(epoch_min_loss)
        else:
            loss /= len(self.train_data)
            epoch_min_loss = loss
            best_idx_in_batch = 0 # Only one prompt if not parallel
        self.loss.append(loss) # Appends list if parallel, float otherwise

        if self.min_loss is None or epoch_min_loss < self.min_loss:
            self.min_loss = epoch_min_loss
            # Update best prompts based on the index found
            if self.parallel:
                self.best_prompt_text = current_prompt_text[best_idx_in_batch].detach().clone()
                self.best_prompt_image = current_prompt_image[best_idx_in_batch].detach().clone()
            else:
                # Ensure it's detached and cloned even in non-parallel case
                self.best_prompt_text = current_prompt_text.detach().clone()
                self.best_prompt_image = current_prompt_image.detach().clone()
            print(f"*** New best loss found: {self.min_loss:.4f} (at call {self.num_call}) ***")


        if self.num_call % self.test_every == 0:
            print(f"\n--- Testing at call {self.num_call} ---")
            # Test clean accuracy
            acc_clean = self.test(attack_config=None)
            self.acc.append(acc_clean.item()) # Store clean accuracy
            self.best_accuracy = max(acc_clean.item(), self.best_accuracy)
            print(f"Clean Accuracy: {acc_clean:.4f} (Best Clean: {self.best_accuracy:.4f})")

            # Test PGD accuracy if enabled
            acc_attacked = torch.tensor(0.0) # Default to 0 if not enabled
            if self.pgd_config["enabled"] and self.best_prompt_text is not None:
                acc_attacked = self.test(attack_config=self.pgd_config)
                self.acc_pgd.append(acc_attacked.item()) # Store PGD accuracy
                self.best_accuracy_pgd = max(acc_attacked.item(), self.best_accuracy_pgd)
                print(f"PGD Accuracy  : {acc_attacked:.4f} (Best PGD  : {self.best_accuracy_pgd:.4f})")
            elif self.pgd_config["enabled"]:
                 print("PGD Accuracy  : Skipped (no best prompt yet or PGD disabled)")


            #---------------save_results-----------------------------------
            output_dir = os.path.join(self.output_dir,self.task_name)

            fname = "{}_{}_{}.pth".format(self.task_name, self.opt_name, self.backbone.replace("/","-"))

            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                       "best_accuracy":self.best_accuracy, "acc":self.acc,
                       "best_accuracy_pgd": self.best_accuracy_pgd, "acc_pgd": self.acc_pgd, # Save PGD results
                       "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,
                       "loss":self.loss,"num_call":self.num_call,
                       "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                       "pgd_config": self.pgd_config} # Save PGD config used
            Analysis_Util.save_results(content,output_dir,fname)
            # ---------------save_results-----------------------------------
        return loss

    def _pgd_attack(self, images, labels, text_features, image_prompt, config):
        """ Performs PGD attack """
        epsilon = config['epsilon']
        alpha = config['alpha']
        num_iter = config['num_iter']

        # Start with random perturbation
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        delta.data.uniform_(-epsilon, epsilon)
        delta.data = torch.clamp(images + delta.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images # Project to valid range

        for _ in range(num_iter):
            # Need gradients for delta
            delta.requires_grad_(True)
            perturbed_image = images + delta

            # Forward pass with perturbed image
            # Ensure image_encoder and text_encoder are not in parallel mode for attack grad calculation
            temp_parallel = self.image_encoder.parallel
            self.image_encoder.parallel = False
            image_features = self.image_encoder(perturbed_image, image_prompt)
            self.image_encoder.parallel = temp_parallel # Restore state

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            # Calculate loss (CrossEntropy is typical for PGD)
            loss = F.cross_entropy(logits, labels)

            # Backward pass to get gradients w.r.t. delta
            loss.backward()

            # PGD step
            delta.data = delta.data + alpha * delta.grad.sign()
            # Project delta to L-infinity ball
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            # Project perturbed image to valid range [0, 1] equivalent in normalized space
            delta.data = torch.clamp(images + delta.data, min=self.norm_lower_limit, max=self.norm_upper_limit) - images
            # Zero gradients for next iteration
            delta.grad.zero_()

        # Return final perturbed image
        return (images + delta).detach()


    @torch.no_grad()
    def test(self, attack_config=None):
        """ Evaluate accuracy, optionally with PGD attack """
        if self.best_prompt_text is None or self.best_prompt_image is None:
            print("Warning: Trying to test without best prompts found yet. Returning 0 accuracy.")
            return torch.tensor(0.0)

        correct = 0.
        total = 0.
        # Ensure evaluation is done sequentially (no parallelism)
        original_parallel_state = self.parallel
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False

        # Pre-compute text features using the best text prompt
        text_features = self.text_encoder(self.best_prompt_text)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)

        desc = "Testing Clean"
        if attack_config and attack_config.get("enabled", False):
             desc = f"Testing PGD(eps={attack_config['epsilon']}, iter={attack_config['num_iter']})"

        for batch in tqdm(self.test_loader, desc=desc, leave=False):
            image,label = self.parse_batch(batch) # parse_batch handles device transfer
            total += image.size(0)

            eval_image = image # Default to clean image

            # --- Apply PGD Attack if configured ---
            if attack_config and attack_config.get("enabled", False):
                 # Enable gradients for attack step ONLY
                 with torch.enable_grad():
                      eval_image = self._pgd_attack(image, label, text_features, self.best_prompt_image, attack_config)
            # --- End PGD Attack ---

            # Compute image features (always no_grad here)
            image_features = self.image_encoder(eval_image, self.best_prompt_image)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale*image_features@text_features.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()

        # Restore original parallel state
        self.parallel=self.text_encoder.parallel = self.image_encoder.parallel = original_parallel_state

        acc = correct/total # Use total derived from batches
        # print(f"Accuracy ({desc}): {acc:.4f}") # Optional: print here too
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

