import torch
import argparse
import yaml
from tqdm import tqdm
from algorithm.CMA_ES import shallow_cma
from algorithm.LM_CMA_ES import Shallow_LMCMAES
from algorithm.MMES import Shallow_MMES
from algorithm.LMMAES import Shallow_LMMAES
# Modified import to point to the correct file
from model.Shallow_Prompt_CLIP_Adv import PromptCLIP_Shallow
import numpy as np
import time
import os
import logging
from model.analysis_utils import Analysis_Util

__classification__ = ["CIFAR100","CIFAR10","CIFAR10_PGD","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet","refcoco"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_maes"]
__dataset__ = "/kaggle/working/dataset"
__output__ = "/kaggle/working/dataset/result"

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="CIFAR100", type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument("--backbone", default="ViT-B/32", type=str)
parser.add_argument("--k_shot", default=16, type=int, help='How many shot to use')
parser.add_argument("--test_every_n_gens", type=int, default=None, help="Run test evaluation every N generations. If not set, testing only occurs at the very end.")


prompt_group = parser.add_argument_group('Initial Prompt Configuration')
prompt_group.add_argument("--initial_prompt_text", type=str, default=None, help="Initial text prompt (e.g., 'a photo of a'). If None, no initial prompt is used.")
prompt_group.add_argument("--learned_prompt_pos", type=str, default="prefix", choices=["prefix", "middle", "suffix"], help="Position of the learned prompt relative to the initial prompt and class name.")

parser.add_argument("--maximize_loss", action='store_true', help='Tune prompts to maximize the loss instead of minimizing it')

# --- NEW: PGD Attacked Dataset Generation Arguments ---
attack_group = parser.add_argument_group('PGD Attacked Dataset Configuration')
attack_group.add_argument("--use_attacked_dataset", action='store_true', help="Enable generation/use of a PGD-attacked dataset.")
attack_group.add_argument("--attack_ratio", type=float, default=0.5, help="Ratio of images to attack in the dataset.")
attack_group.add_argument("--attack_train", action='store_true', help="Apply attack to the training set.")
attack_group.add_argument("--attack_test", action='store_true', help="Apply attack to the test set.")
attack_group.add_argument("--pgd_eps", type=float, default=8/255.0, help="PGD attack epsilon.")
attack_group.add_argument("--pgd_alpha", type=float, default=2/255.0, help="PGD attack alpha (step size).")
attack_group.add_argument("--pgd_steps", type=int, default=10, help="Number of PGD attack steps.")
# --- END NEW ---


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
cfg["maximize_loss"] = args.maximize_loss
cfg["k_shot"] = args.k_shot
cfg["initial_prompt_text"] = args.initial_prompt_text
cfg["learned_prompt_pos"] = args.learned_prompt_pos
cfg["test_every_n_gens"] = args.test_every_n_gens

# --- NEW: Add PGD args to config dict ---
cfg["use_attacked_dataset"] = args.use_attacked_dataset
cfg["attack_ratio"] = args.attack_ratio
cfg["attack_train"] = args.attack_train
cfg["attack_test"] = args.attack_test
cfg["pgd_eps"] = args.pgd_eps
cfg["pgd_alpha"] = args.pgd_alpha
cfg["pgd_steps"] = args.pgd_steps
# --- END NEW ---

if args.task_name in cfg:
    for k,v in cfg[args.task_name].items():
        cfg[k]=v
else:
    logging.warning(f"Task '{args.task_name}' not found in config. Using default settings.")

# White-box attack configurations removed

output_dir = os.path.join(cfg["output_dir"], args.task_name)
Analysis_Util.mkdir_if_missing(output_dir)


initial_prompt_str_fn = f"_initPrompt" if cfg["initial_prompt_text"] is not None else ""
learned_pos_str_fn = f"_pos{cfg['learned_prompt_pos']}"


fname_base = "{}{}_{}_{}_parallel{}{}_maxLoss{}".format(
    cfg["k_shot"],
    args.task_name,
    cfg["opt_name"],
    cfg["backbone"].replace("/", "-"),
    args.parallel,
    initial_prompt_str_fn,
    learned_pos_str_fn,
    cfg["maximize_loss"]
)
log_filename = fname_base + ".log"
log_filepath = os.path.join(output_dir, log_filename)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(log_filepath)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("--- Starting Run ---")
logger.info(f"Arguments: {args}")
logger.info(f"Loaded Config: {cfg}")
logger.info(f"Log file path: {log_filepath}")

device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]

if args.task_name in __classification__:
    prompt_clip = PromptCLIP_Shallow(args.task_name, cfg)
else:
     logger.error(f"Task type for '{args.task_name}' not fully implemented.")
     exit()

def fitness_eval(prompt_zip_np):
    prompt_zip_np = np.array(prompt_zip_np)
    prompt_text_intrinsic = prompt_zip_np[:intrinsic_dim_L]
    prompt_image_intrinsic = prompt_zip_np[intrinsic_dim_L:]

    prompt_text_list = prompt_clip.generate_text_prompts([prompt_text_intrinsic])
    prompt_image_list = prompt_clip.generate_visual_prompts([prompt_image_intrinsic])

    original_parallel = prompt_clip.parallel
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False
    fit_value = prompt_clip.eval([prompt_text_list[0], prompt_image_list[0]])
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = original_parallel

    return fit_value.item() if isinstance(fit_value, torch.Tensor) else fit_value

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
}

opt = None
if args.opt == "shallow_cma":
    opt = shallow_cma(cfg)
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

logger.info(f"Task: {args.task_name}")
logger.info(f"Optimizer: {args.opt}")
logger.info(f'Population Size: {cfg["popsize"]}')
logger.info(f"Using Backbone: {cfg['backbone']}")
logger.info(f"Parallel Evaluation during Search: {cfg['parallel']}")
logger.info(f"Device: {device}")
logger.info(f"Intrinsic Dimensions: L={intrinsic_dim_L}, V={intrinsic_dim_V}")

if cfg['test_every_n_gens'] is not None:
    logger.info(f"Intermediate Testing Frequency: Every {cfg['test_every_n_gens']} generations.")
else:
    logger.info("Intermediate Testing: Disabled (will only test at the end of the run).")

logger.info(f"Initial Prompt Text: '{cfg['initial_prompt_text']}'")
logger.info(f"Learned Prompt Position: {cfg['learned_prompt_pos']}")
logger.info(f"Optimization Objective: {'Maximize' if cfg['maximize_loss'] else 'Minimize'} Loss")
logger.info(f"Budget: {opt_cfg['budget']}")
logger.info(f"Adversarial Training (during optimization): False")
logger.info(f"Attack during Final Test: False")
if args.use_attacked_dataset:
    logger.info("--- Using Pre-Attacked PGD Dataset for Tuning/Testing ---")
    logger.info(f"Attack on TRAIN set: {args.attack_train}")
    logger.info(f"Attack on TEST set: {args.attack_test}")
    logger.info(f"Attack Ratio: {args.attack_ratio}, Epsilon: {args.pgd_eps}, Alpha: {args.pgd_alpha}, Steps: {args.pgd_steps}")


start_time = time.time()
logger.info("--- Starting Optimization Loop ---")

if args.opt in __pypop__:
    if args.task_name in __classification__:
        logger.info("Setting up text and image context for PyPop optimizer.")
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)
        res = opt.optimize()
        logger.info(f"Optimization Result (PyPop): {res}")
    else:
        logger.warning(f"PyPop optimizer path not fully defined for task {args.task_name}")

else:
    if args.task_name in __classification__:
        logger.info("Setting up text and image context for non-PyPop optimizer.")
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)

        while not opt.stop():
            solutions = opt.ask()

            prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
            prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])

            if cfg["parallel"]:
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = True
                fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list])
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False
                logger.info(f"Evaluated Parallel Pop (call {prompt_clip.num_call})")

            else:
                prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False
                fitnesses = []
                for i, p_zip in enumerate(tqdm(zip(prompt_text_list, prompt_image_list), total=len(solutions), ncols=80, desc="Eval Sequential Pop")):
                     fit = prompt_clip.eval(p_zip)
                     fitnesses.append(fit.item() if isinstance(fit, torch.Tensor) else fit)
                logger.info(f"Evaluated Sequential Pop (call {prompt_clip.num_call})")

            opt.tell(solutions, fitnesses)

            # Logging of best objective value every N generations (controlled by verbose_frequency in YAML)
            # This is independent of the testing frequency.
            if prompt_clip.num_call % (cfg["popsize"] * opt_cfg['verbose_frequency']) == 0:
                 log_loss_label = "Maximized Loss" if prompt_clip.maximize_loss else "Minimized Loss"
                 
                 # The detailed test accuracy log is now inside prompt_clip.eval(), so we only log the objective here
                 logger.info(f"Generation ~{int(prompt_clip.num_call / cfg['popsize'])}, Best Objective ({log_loss_label}): {prompt_clip.best_objective_loss_value:.4f}")

    else:
        logger.warning(f"Non-PyPop optimizer path not fully defined for task {args.task_name}")

logger.info("\n--- Optimization Finished ---")
end_time = time.time()
optimization_time = end_time - start_time
logger.info(f"Total Optimization Time: {optimization_time:.2f} seconds")

logger.info("\n--- Final Evaluation using Best Prompts ---")
final_acc_clean = prompt_clip.test()
logger.info(f"Final Clean Accuracy: {final_acc_clean:.4f}")

# Final attacked accuracy evaluation removed

pth_filename = fname_base + "_final.pth"
final_results_path = os.path.join(output_dir, pth_filename)

content = {
    "task_name": args.task_name, "opt_name": cfg["opt_name"], "backbone": cfg["backbone"],
    "k_shot": cfg["k_shot"],
    "best_accuracy": prompt_clip.best_accuracy, "acc": prompt_clip.acc,
    "best_train_accuracy": prompt_clip.best_train_accuracy, "train_acc": prompt_clip.train_acc,
    "best_prompt_text": prompt_clip.best_prompt_text, "best_prompt_image": prompt_clip.best_prompt_image,
    "best_objective_loss_value": prompt_clip.best_objective_loss_value,
    "maximize_loss_setting": prompt_clip.maximize_loss,
    "loss": prompt_clip.loss, "num_call": prompt_clip.num_call,
    "final_acc_clean": final_acc_clean.item(),
    "Linear_L": prompt_clip.linear_L.state_dict(),
    "Linear_V": prompt_clip.linear_V.state_dict(),
    "optimization_time_seconds": optimization_time,
    "config_used": cfg,
    "args_used": vars(args)
}

Analysis_Util.save_results(content, output_dir, pth_filename)
logger.info(f"Final results saved to {final_results_path}")
logger.info("--- Run Complete ---")