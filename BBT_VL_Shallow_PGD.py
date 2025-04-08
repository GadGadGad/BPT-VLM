import torch
import argparse
import yaml
from tqdm import tqdm
from algorithm.CMA_ES import shallow_cma
from algorithm.LM_CMA_ES import Shallow_LMCMAES
from algorithm.MMES import Shallow_MMES
from algorithm.LMMAES import Shallow_LMMAES
from model.Shallow_Prompt_CLIP_PGD import PromptCLIP_Shallow
import numpy as np
import time
import os #
from model.analysis_utils import Analysis_Util
__classification__ = ["CIFAR100","CIFAR10","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet","refcoco"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_maes"]
__dataset__ = "/kaggle/working/BPT-VLM/dataset"
__output__ = "/kaggle/working/BPT-VLM/dataset/result"
# __output__ = "/home/yu/result"
# __backbone__ = "ViT-B/32"

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="CIFAR100", type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument("--backbone", default="ViT-B/32", type=str)

args = parser.parse_args()
assert "shallow" in args.opt, "Only shallow prompt tuning is supported in this file."

config_path = "./configs/shallow_prompt.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")
with open(config_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["opt_name"] = args.opt
cfg["data_dir"] = __dataset__
cfg["output_dir"] = __output__
cfg["backbone"] = args.backbone
cfg["parallel"] = args.parallel

# Merge dataset-specific config
if args.task_name in cfg:
    for k,v in cfg[args.task_name].items():
        cfg[k]=v
else:
    print(f"Warning: Task '{args.task_name}' not found in config. Using default settings.")

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]


# Build CLIP model and PGD config
if args.task_name in __classification__:
    prompt_clip = PromptCLIP_Shallow(args.task_name, cfg) # Pass the whole cfg
else:
     # Handle non-classification tasks if necessary (assuming they might exist later)
     # prompt_clip = ...
     pass # Placeholder


# --- Fitness Function Definition (for specific optimizers like LM-CMA-ES) ---
# This function evaluates a *single* individual solution.
def fitness_eval(prompt_zip_np):
    prompt_zip_np = np.array(prompt_zip_np) # Ensure it's numpy
    prompt_text_intrinsic = prompt_zip_np[:intrinsic_dim_L]
    prompt_image_intrinsic = prompt_zip_np[intrinsic_dim_L:]

    # Generate prompts (needs to be done for a single individual here)
    prompt_text_list = prompt_clip.generate_text_prompts([prompt_text_intrinsic])
    prompt_image_list = prompt_clip.generate_visual_prompts([prompt_image_intrinsic]) 
    # Evaluate the single prompt pair
    # Set parallel to False temporarily for single eval call consistency
    original_parallel = prompt_clip.parallel
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False
    fitness = prompt_clip.eval(list(zip(prompt_text_list, prompt_image_list))[0]).item() # Pass the single tuple
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = original_parallel # Restore

    # Logging (similar to original, adjust if needed)
    if prompt_clip.num_call % (prompt_clip.test_every) == 0:
        print("-------------------------Epoch {}---------------------------".format(prompt_clip.num_call/prompt_clip.test_every))
        print("Evaluation in fitness_eval (call {})".format(prompt_clip.num_call)) # Indicate where eval happens
        print("current loss: {}".format(prompt_clip.min_loss))
        print("Best Prompt Embedding - Acc : {:.4f}".format(prompt_clip.best_accuracy))
        if prompt_clip.pgd_config.get("enabled", False):
            print("Best Prompt Embedding - PGD Acc : {:.4f}".format(prompt_clip.best_accuracy_pgd))

    return fitness # Return single fitness value


ndim_problem = intrinsic_dim_L + intrinsic_dim_V
pro = {'fitness_function': fitness_eval, 'ndim_problem': ndim_problem}

# Opt Config needs careful checking - n_individuals vs popsize
opt_cfg = {
    'fitness_threshold': 1e-10,
    'seed_rng': cfg.get('seed', 0), # Use seed from main cfg if available
    'max_runtime': cfg.get('max_runtime', 20800), # Use from main cfg if available
    'x': cfg.get('initial_mean', 0 * np.ones((ndim_problem,))), # Allow initial mean override
    'sigma': cfg['sigma'],
    'verbose_frequency': cfg.get('verbose_frequency', 5),
    'n_individuals': cfg["popsize"], # Use popsize from main cfg
    'is_restart': cfg.get('is_restart', False) # Use from main cfg if available
}


opt = None
if args.opt == "shallow_cma":
    opt = shallow_cma(cfg) # Assumes shallow_cma takes the cfg dict
elif args.opt == "shallow_lmcmaes":
    # Note: LM-CMA-ES uses the fitness_eval function defined above
    opt = Shallow_LMCMAES(pro, opt_cfg)
    print("Using LM-CMA-ES (PyPop based) - Evaluation via single fitness_eval function.")
elif args.opt == "shallow_mmes":
    opt = Shallow_MMES(pro, opt_cfg)
    print("Using MMES (PyPop based) - Evaluation via single fitness_eval function.")
elif args.opt == "shallow_lmmaes":
    opt = Shallow_LMMAES(pro,opt_cfg)
    print("Using LMMAES (PyPop based) - Evaluation via single fitness_eval function.")
else:
    raise ValueError(f"Unsupported optimizer: {args.opt}")


print('Population Size: {}'.format(cfg["popsize"]))
print(f"Using Backbone: {cfg['backbone']}")
print(f"Task: {args.task_name}")
print(f"Optimizer: {args.opt}")
print(f"Parallel Evaluation during Search: {cfg['parallel']}")
print(f"Device: {device}")


start_time = time.time()

if args.opt in __pypop__:
    if args.task_name in __classification__:
        # Set context before optimizing
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)
        # The optimize call will internally use the 'fitness_eval' function
        res = opt.optimize()
        print("Optimization Result (PyPop):", res)
    else:
        print(f"Warning: PyPop optimizer path not fully defined for task {args.task_name}")
        # image_context = prompt_clip.get_image_information()
        # prompt_clip.image_encoder.set_context(image_context)
        # res = opt.optimize() # May need adaptation for non-classification tasks

else: # Handle custom CMA-ES loop (like original shallow_cma might have)
    if args.task_name in __classification__:
        # Set context before optimizing
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)

        print("Starting Optimization Loop...")
        # Assume opt (e.g., shallow_cma) has ask() and tell()
        while not opt.stop():
            solutions = opt.ask() # Get population solutions [popsize, ndim]

            # Generate prompts from intrinsic vectors
            prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
            prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])

            # Evaluate fitness (either parallel or sequential)
            if cfg["parallel"]:
                # Parallel evaluation: pass lists directly
                # prompt_clip.eval handles the internal parallel logic
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = True # Ensure parallel mode
                fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list]) # Returns a list of losses
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False # Revert after eval
                # fitnesses = [x.item() for x in tqdm(fitnesses, ncols=80, desc="Eval Parallel Pop")] # Removed item() call as eval should return list of floats now
                print(f"Eval Parallel Pop (call {prompt_clip.num_call})") # Simple print instead of tqdm bar

            else:
                # Sequential evaluation: iterate and call eval
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False # Ensure sequential mode
                fitnesses = []
                for i, p_zip in enumerate(tqdm(zip(prompt_text_list, prompt_image_list), total=len(solutions), ncols=80, desc="Eval Sequential Pop")):
                     fit = prompt_clip.eval(p_zip) # eval now handles single input correctly
                     fitnesses.append(fit.item() if isinstance(fit, torch.Tensor) else fit) # Handle tensor or float return
                     # The logging inside eval handles printing loss/acc periodically

            # Tell optimizer the results
            opt.tell(solutions, fitnesses)

            # Optional: Print progress summary less frequently than test_every
            if prompt_clip.num_call % (cfg["popsize"] * 5) == 0: # Every 5 generations approx
                 print(f"Generation ~{int(prompt_clip.num_call / cfg['popsize'])}, Min Loss: {prompt_clip.min_loss:.4f}, Best Acc: {prompt_clip.best_accuracy:.4f}")

    else:
        # Handle non-classification tasks if needed
        print(f"Warning: Non-PyPop optimizer path not fully defined for task {args.task_name}")
        # image_context =prompt_clip.get_image_information()
        # prompt_clip.image_encoder.set_context(image_context)
        # ... (similar loop structure) ...

print("\n--- Optimization Finished ---")
end_time = time.time()
print(f"Total Optimization Time: {end_time - start_time:.2f} seconds")

print("\n--- Final Evaluation ---")
final_acc_clean = prompt_clip.test(attack_config=None)
print(f"Final Clean Accuracy: {final_acc_clean:.4f}")

final_acc_pgd = torch.tensor(0.0)
if prompt_clip.pgd_config["enabled"]:
    final_acc_pgd = prompt_clip.test(attack_config=prompt_clip.pgd_config)
    print(f"Final PGD Accuracy  : {final_acc_pgd:.4f}")
else:
    print("Final PGD Accuracy  : Skipped (PGD not enabled in config)")

output_dir = os.path.join(cfg["output_dir"], args.task_name)
fname = "{}_{}_{}_final.pth".format(args.task_name, cfg["opt_name"], cfg["backbone"].replace("/", "-"))
content = {
    "task_name": args.task_name, "opt_name": cfg["opt_name"], "backbone": cfg["backbone"],
    "best_accuracy": prompt_clip.best_accuracy, "acc": prompt_clip.acc,
    "best_accuracy_pgd": prompt_clip.best_accuracy_pgd, "acc_pgd": prompt_clip.acc_pgd,
    "best_prompt_text": prompt_clip.best_prompt_text, "best_prompt_image": prompt_clip.best_prompt_image,
    "loss": prompt_clip.loss, "num_call": prompt_clip.num_call,
    "final_acc_clean": final_acc_clean.item(), # Store final scalar value
    "final_acc_pgd": final_acc_pgd.item() if prompt_clip.pgd_config["enabled"] else None, # Store final scalar value
    "Linear_L": prompt_clip.linear_L.state_dict(), "Linear_V": prompt_clip.linear_V.state_dict(),
    "pgd_config": prompt_clip.pgd_config,
    "optimization_time_seconds": end_time - start_time
}
Analysis_Util.save_results(content, output_dir, fname)
print(f"Final results saved to {os.path.join(output_dir, fname)}")

print("--- Run Complete ---")
