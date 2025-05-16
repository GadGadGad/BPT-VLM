# BBT_VL_Shallow_PGD.py

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
import os
import logging # Import logging
from model.analysis_utils import Analysis_Util

# --- Argument Parsing ---
__classification__ = ["CIFAR100","CIFAR10","CIFAR10_PGD","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet","refcoco"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_maes"]
__dataset__ = "/kaggle/working/dataset"
__output__ = "/kaggle/working/dataset/result"
# __output__ = "/home/yu/result"
# __backbone__ = "ViT-B/32"

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="CIFAR100", type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument("--backbone", default="ViT-B/32", type=str)
parser.add_argument("--pgd_test", action='store_true', help='Enable PGD Attack during final testing')
parser.add_argument("--adv_train", action='store_true', help='Enable Adversarial Training')
parser.add_argument("--pgd_original_prompt", action='store_true', help='Use original CLIP prompts for PGD testing instead of tuned ones')
args = parser.parse_args()
assert "shallow" in args.opt, "Only shallow prompt tuning is supported in this file."

# --- Config Loading ---
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

if args.task_name in cfg:
    for k,v in cfg[args.task_name].items():
        cfg[k]=v
else:
    # Use logging for warning instead of print
    logging.warning(f"Task '{args.task_name}' not found in config. Using default settings.")

if 'pgd' not in cfg:
    cfg['pgd'] = {}
cfg['pgd']['enabled'] = args.pgd_test
cfg['pgd']['original_prompt'] = args.pgd_original_prompt

if 'adv_train' not in cfg:
    cfg['adv_train'] = {}
cfg['adv_train']['enabled'] = args.adv_train

# --- Determine Output Directory and Base Filename ---
output_dir = os.path.join(cfg["output_dir"], args.task_name)
Analysis_Util.mkdir_if_missing(output_dir) # Ensure directory exists before logging setup

# Define base filename structure (used for both log and pth files)
fname_base = "{}_{}_{}_advOpt{}_pgdOrg{}".format(
    args.task_name,
    cfg["opt_name"],
    cfg["backbone"].replace("/", "-"),
    cfg["adv_train"]["enabled"],
    cfg["pgd"]["original_prompt"],
)
log_filename = fname_base + ".log"
log_filepath = os.path.join(output_dir, log_filename)

# --- Logging Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console Handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File Handler (Log to file)
fh = logging.FileHandler(log_filepath)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("--- Starting Run ---")
logger.info(f"Arguments: {args}")
logger.info(f"Loaded Config: {cfg}")
logger.info(f"Log file path: {log_filepath}")


# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]

# Build CLIP model
if args.task_name in __classification__:
    prompt_clip = PromptCLIP_Shallow(args.task_name, cfg) # Pass the whole cfg
else:
     # Handle non-classification tasks if necessary
     logger.error(f"Task type for '{args.task_name}' not fully implemented.")
     # You might want to exit or raise an error here depending on requirements
     exit() # Example: Exit if task is not supported

# --- Fitness Evaluation Function ---
def fitness_eval(prompt_zip_np):
    prompt_zip_np = np.array(prompt_zip_np) # Ensure it's numpy
    prompt_text_intrinsic = prompt_zip_np[:intrinsic_dim_L]
    prompt_image_intrinsic = prompt_zip_np[intrinsic_dim_L:]

    prompt_text_list = prompt_clip.generate_text_prompts([prompt_text_intrinsic]) 
    prompt_image_list = prompt_clip.generate_visual_prompts([prompt_image_intrinsic]) 

    # Ensure evaluation happens sequentially for single fitness calls
    original_parallel = prompt_clip.parallel
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False
    fit_value = prompt_clip.eval([prompt_text_list[0], prompt_image_list[0]])
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = original_parallel # Restore

    return fit_value.item() if isinstance(fit_value, torch.Tensor) else fit_value

# --- Optimization Setup ---
ndim_problem = intrinsic_dim_L + intrinsic_dim_V
pro = {'fitness_function': fitness_eval, 'ndim_problem': ndim_problem}

opt_cfg = {
    'fitness_threshold': 1e-10,
    'seed_rng': cfg.get('seed', 0),
    'budget': cfg.get('budget', 25200),
    'x': cfg.get('initial_mean', 0 * np.ones((ndim_problem,))),
    'sigma': cfg['sigma'],
    'verbose_frequency': cfg.get('verbose_frequency', 5),
    'n_individuals': cfg["popsize"],
    # 'is_restart': cfg.get('is_restart', False) # Uncomment if needed by algorithm
}

# --- Load Algorithm ---
opt = None
if args.opt == "shallow_cma":
    opt = shallow_cma(cfg) #
    logger.info("Using custom shallow_cma.")
elif args.opt == "shallow_lmcmaes":
    opt = Shallow_LMCMAES(pro, opt_cfg)
    logger.info("Using LM-CMA-ES (PyPop based) - Evaluation via single fitness_eval function.")
elif args.opt == "shallow_mmes":
    opt = Shallow_MMES(pro, opt_cfg)
    logger.info("Using MMES (PyPop based) - Evaluation via single fitness_eval function.")
elif args.opt == "shallow_lmmaes":
    opt = Shallow_LMMAES(pro, opt_cfg)
    logger.info("Using LMMAES (PyPop based) - Evaluation via single fitness_eval function.")
else:
    logger.error(f"Unsupported optimizer: {args.opt}")
    raise ValueError(f"Unsupported optimizer: {args.opt}")

# --- Log Setup Details ---
logger.info(f"Task: {args.task_name}")
logger.info(f"Optimizer: {args.opt}")
logger.info(f'Population Size: {cfg["popsize"]}')
logger.info(f"Using Backbone: {cfg['backbone']}")
logger.info(f"Parallel Evaluation during Search: {cfg['parallel']}")
logger.info(f"Adversarial Training (PGD during optimization): {cfg['adv_train']['enabled']}")
logger.info(f"PGD Attack during Final Test: {cfg['pgd']['enabled']}")
logger.info(f"Device: {device}")
logger.info(f"Intrinsic Dimensions: L={intrinsic_dim_L}, V={intrinsic_dim_V}")
logger.info(f"Budget: {opt_cfg['budget']}")


# --- Black-box prompt tuning ---
start_time = time.time()
logger.info("--- Starting Optimization Loop ---")

if args.opt in __pypop__:
    if args.task_name in __classification__:
        # Set context before optimizing
        logger.info("Setting up text and image context for PyPop optimizer.")
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)
        res = opt.optimize()
        logger.info(f"Optimization Result (PyPop): {res}")
    else:
        logger.warning(f"PyPop optimizer path not fully defined for task {args.task_name}")
        # Handle non-classification tasks if needed
        # image_context = prompt_clip.get_image_information()
        # prompt_clip.image_encoder.set_context(image_context)
        # res = opt.optimize() # May need adaptation

else: # Handle non-PyPop optimizers (like the assumed shallow_cma)
    if args.task_name in __classification__:
        logger.info("Setting up text and image context for non-PyPop optimizer.")
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)

        while not opt.stop():
            solutions = opt.ask() # Get population solutions [popsize, ndim]

            prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
            prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])

            if cfg["parallel"]:
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = True
                # Pass a list containing two lists: [list_of_text_prompts, list_of_image_prompts]
                fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list])
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False # Revert after eval
                logger.info(f"Evaluated Parallel Pop (call {prompt_clip.num_call})")

            else:
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False # Ensure sequential mode
                fitnesses = []
                # Use tqdm for progress bar, but logging inside prompt_clip.eval handles detailed output
                for i, p_zip in enumerate(tqdm(zip(prompt_text_list, prompt_image_list), total=len(solutions), ncols=80, desc="Eval Sequential Pop")):
                     # Pass the single tuple: (text_prompt_tensor, image_prompt_tensor)
                     fit = prompt_clip.eval(p_zip)
                     fitnesses.append(fit.item() if isinstance(fit, torch.Tensor) else fit)
                logger.info(f"Evaluated Sequential Pop (call {prompt_clip.num_call})")


            opt.tell(solutions, fitnesses)

            # Log progress periodically based on number of calls
            if prompt_clip.num_call % (cfg["popsize"] * opt_cfg['verbose_frequency']) == 0:
                 logger.info(f"Generation ~{int(prompt_clip.num_call / cfg['popsize'])}, Min Loss: {prompt_clip.min_loss:.4f}, Best Acc: {prompt_clip.best_accuracy:.4f}, Best PGD Acc: {prompt_clip.best_accuracy_pgd:.4f}")

    else:
        logger.warning(f"Non-PyPop optimizer path not fully defined for task {args.task_name}")
        # Handle non-classification tasks if needed
        # image_context =prompt_clip.get_image_information()
        # prompt_clip.image_encoder.set_context(image_context)
        # ... (similar loop structure) ...

# --- Final Evaluation ---
logger.info("\n--- Optimization Finished ---")
end_time = time.time()
optimization_time = end_time - start_time
logger.info(f"Total Optimization Time: {optimization_time:.2f} seconds")

logger.info("\n--- Final Evaluation using Best Prompts ---")
final_acc_clean = prompt_clip.test(attack_config=None)
logger.info(f"Final Clean Accuracy: {final_acc_clean:.4f}")

final_acc_pgd = torch.tensor(0.0)
if cfg['pgd']['enabled']:
    if prompt_clip.best_prompt_text is not None:
        pgd_test_type_str = " (Original Prompts)" if cfg['pgd']['original_prompt'] else ""
        final_acc_pgd = prompt_clip.test(attack_config=prompt_clip.pgd_config)
        logger.info(f"Final PGD Accuracy{pgd_test_type_str}  : {final_acc_pgd:.4f}")
    else:
        logger.info("Final PGD Accuracy  : Skipped (no best prompt available)")
        final_acc_pgd = None 
else:
    logger.info("Final PGD Accuracy  : Skipped (PGD test not enabled in args/config)")
    final_acc_pgd = None 
    
# --- Save Final Results ---
pth_filename = fname_base + "_final.pth"
final_results_path = os.path.join(output_dir, pth_filename)

content = {
    "task_name": args.task_name, "opt_name": cfg["opt_name"], "backbone": cfg["backbone"],
    "best_accuracy": prompt_clip.best_accuracy, "acc": prompt_clip.acc,
    "best_accuracy_pgd": prompt_clip.best_accuracy_pgd, "acc_pgd": prompt_clip.acc_pgd,
    "best_prompt_text": prompt_clip.best_prompt_text, "best_prompt_image": prompt_clip.best_prompt_image,
    "loss": prompt_clip.loss, "num_call": prompt_clip.num_call,
    "final_acc_clean": final_acc_clean.item(),
    "final_acc_pgd": final_acc_pgd.item() if isinstance(final_acc_pgd, torch.Tensor) else final_acc_pgd,
    "Linear_L": prompt_clip.linear_L.state_dict(),
    "Linear_V": prompt_clip.linear_V.state_dict(),
    "pgd_config_test": prompt_clip.pgd_config,
    "adv_train_config": prompt_clip.adv_train_config,
    "optimization_time_seconds": optimization_time,
    "config_used": cfg, 
    "args_used": vars(args)
}

Analysis_Util.save_results(content, output_dir, pth_filename)
logger.info(f"Final results saved to {final_results_path}")
logger.info("--- Run Complete ---")