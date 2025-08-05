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

dim_group = parser.add_argument_group('Prompt Dimension Configuration')
dim_group.add_argument("--n_prompt_tokens_L", type=int, default=None, help="Number of learnable text prompt tokens.")
dim_group.add_argument("--n_prompt_tokens_V", type=int, default=None, help="Number of learnable visual prompt tokens.")
dim_group.add_argument("--intrinsic_dim_L", type=int, default=None, help="Intrinsic dimension for text prompts.")
dim_group.add_argument("--intrinsic_dim_V", type=int, default=None, help="Intrinsic dimension for visual prompts.")

prompt_group = parser.add_argument_group('Initial Prompt Configuration')
prompt_group.add_argument("--initial_prompt_text", type=str, default=None, help="Initial text prompt (e.g., 'a photo of a'). If None, no initial prompt is used.")
prompt_group.add_argument("--learned_prompt_pos", type=str, default="prefix", choices=["prefix", "middle", "suffix"], help="Position of the learned prompt relative to the initial prompt and class name.")

parser.add_argument("--maximize_loss", action='store_true', help='Tune prompts to maximize the loss instead of minimizing it')

surrogate_group = parser.add_argument_group('Adversarial Attack Surrogate Configuration')
surrogate_group.add_argument("--attack_surrogate_model", type=str, default=None,
                             help="Specify a different model architecture for generating attacks (e.g., 'RN50', 'ViT-L/14'). "
                                  "If None, uses the main backbone.")
surrogate_group.add_argument("--attack_surrogate_prompt_text", type=str, default="a photo of a {}",
                             help="Text prompt template to use with the surrogate model for attack generation. "
                                  "Use '{}' as a placeholder for the class name.")

# --- MODIFIED: Expanded PGD/FGSM/CW Attacked Dataset Generation Arguments ---
attack_group = parser.add_argument_group('Adversarial Attack Configuration')
attack_group.add_argument("--use_attacked_dataset", action='store_true', help="Enable generation/use of a PGD-attacked dataset.")
attack_group.add_argument("--attack_train", action='store_true', help="Apply attack to the training set.")
attack_group.add_argument("--attack_test", action='store_true', help="Apply attack to the test set.")

attack_group.add_argument("--attack_type_train", type=str, default="pgd", choices=["pgd", "fgsm", "cw"], help="Type of attack for the training set.")
attack_group.add_argument("--attack_type_test", type=str, default="pgd", choices=["pgd", "fgsm", "cw"], help="Type of attack for the test set.")

attack_group.add_argument("--attack_train_ratio", type=float, default=0.5, help="Ratio of images to attack in the training set.")
attack_group.add_argument("--attack_test_ratio", type=float, default=0.5, help="Ratio of images to attack in the test set.")

pgd_group = attack_group.add_argument_group('PGD/FGSM Parameters')
pgd_group.add_argument("--pgd_eps_train", type=float, default=8/255.0, help="PGD/FGSM attack epsilon for training set.")
pgd_group.add_argument("--pgd_alpha_train", type=float, default=2/255.0, help="PGD attack alpha (step size) for training set.")
pgd_group.add_argument("--pgd_steps_train", type=int, default=10, help="Number of PGD attack steps for training set.")
pgd_group.add_argument("--pgd_eps_test", type=float, default=8/255.0, help="PGD/FGSM attack epsilon for test set.")
pgd_group.add_argument("--pgd_alpha_test", type=float, default=2/255.0, help="PGD attack alpha (step size) for test set.")
pgd_group.add_argument("--pgd_steps_test", type=int, default=10, help="Number of PGD attack steps for test set.")

cw_group = attack_group.add_argument_group('CW Parameters')
cw_group.add_argument("--cw_c", type=float, default=1.0, help="CW attack confidence parameter (trade-off between dist and class loss).")
cw_group.add_argument("--cw_lr", type=float, default=0.01, help="CW attack optimizer learning rate.")
cw_group.add_argument("--cw_steps", type=int, default=20, help="Number of CW attack optimization steps.")
# --- END MODIFIED ---

# --- Noise Injection Arguments ---
noise_group = parser.add_argument_group('Noise Injection Configuration')
noise_group.add_argument("--noise_type_text", type=str, default="none", choices=["none", "gaussian", "uniform", "binomial"], help="Type of noise to add to text prompts during tuning.")
noise_group.add_argument("--noise_type_visual", type=str, default="none", choices=["none", "gaussian", "uniform", "binomial"], help="Type of noise to add to visual prompts during tuning.")
noise_group.add_argument("--noise_level", type=float, default=0.1, help="Magnitude/standard deviation of the injected noise.")


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

# --- MODIFIED: Add all attack and noise args to config dict ---
cfg["use_attacked_dataset"] = args.use_attacked_dataset
cfg["attack_train"] = args.attack_train
cfg["attack_test"] = args.attack_test
cfg["attack_type_train"] = args.attack_type_train
cfg["attack_type_test"] = args.attack_type_test
cfg["attack_train_ratio"] = args.attack_train_ratio
cfg["attack_test_ratio"] = args.attack_test_ratio
# PGD/FGSM
cfg["pgd_eps_train"] = args.pgd_eps_train
cfg["pgd_alpha_train"] = args.pgd_alpha_train
cfg["pgd_steps_train"] = args.pgd_steps_train
cfg["pgd_eps_test"] = args.pgd_eps_test
cfg["pgd_alpha_test"] = args.pgd_alpha_test
cfg["pgd_steps_test"] = args.pgd_steps_test
# CW
cfg["cw_c"] = args.cw_c
cfg["cw_lr"] = args.cw_lr
cfg["cw_steps"] = args.cw_steps
# Surrogate model
cfg["attack_surrogate_model"] = args.attack_surrogate_model
cfg["attack_surrogate_prompt_text"] = args.attack_surrogate_prompt_text
# Noise
cfg["noise_type_text"] = args.noise_type_text
cfg["noise_type_visual"] = args.noise_type_visual
cfg["noise_level"] = args.noise_level
# --- END MODIFIED ---

if args.task_name in cfg:
    for k,v in cfg[args.task_name].items():
        cfg[k]=v
else:
    logging.warning(f"Task '{args.task_name}' not found in config. Using default settings.")

if args.n_prompt_tokens_L is not None:
    cfg['n_prompt_tokens_L'] = args.n_prompt_tokens_L
if args.n_prompt_tokens_V is not None:
    cfg['n_prompt_tokens_V'] = args.n_prompt_tokens_V
if args.intrinsic_dim_L is not None:
    cfg['intrinsic_dim_L'] = args.intrinsic_dim_L
if args.intrinsic_dim_V is not None:
    cfg['intrinsic_dim_V'] = args.intrinsic_dim_V
    
if cfg['n_prompt_tokens_L'] == 0:
    cfg['intrinsic_dim_L'] = 0
if cfg['n_prompt_tokens_V'] == 0:
    cfg['intrinsic_dim_V'] = 0
    

output_dir = os.path.join(cfg["output_dir"], args.task_name)
Analysis_Util.mkdir_if_missing(output_dir)

initial_prompt_str_fn = f"_initPrompt" if cfg["initial_prompt_text"] is not None else ""
learned_pos_str_fn = f"_pos{cfg['learned_prompt_pos']}"

noise_str_fn = ""
if cfg['noise_type_text'] != 'none' or cfg['noise_type_visual'] != 'none':
    noise_str_fn = f"_noiseT_{cfg['noise_type_text']}_noiseV_{cfg['noise_type_visual']}_level_{cfg['noise_level']}"

fname_base = "{}{}_{}_{}_parallel{}{}_maxLoss{}{}".format(
    cfg["k_shot"], args.task_name, cfg["opt_name"], cfg["backbone"].replace("/", "-"),
    args.parallel, initial_prompt_str_fn, learned_pos_str_fn, cfg["maximize_loss"], noise_str_fn
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

surrogate_clip_model = None
surrogate_preprocess = None
if args.use_attacked_dataset and args.attack_surrogate_model:
    logger.info(f"Loading surrogate model '{args.attack_surrogate_model}' for attack generation.")
    # We load the surrogate model here and pass it into the main class.
    # This keeps the model loading logic in the main script.
    surrogate_clip_model, surrogate_preprocess = clip.load(args.attack_surrogate_model, device=device)
    
if args.task_name in __classification__:
    prompt_clip = PromptCLIP_Shallow(args.task_name, cfg,
                                    surrogate_clip_model=surrogate_clip_model,
                                    surrogate_preprocess=surrogate_preprocess)
else:
     logger.error(f"Task type for '{args.task_name}' not fully implemented.")
     exit()

def fitness_eval(prompt_zip_np):
    prompt_zip_np = np.array(prompt_zip_np)
    prompt_text_intrinsic = None
    prompt_image_intrinsic = None
    
    current_pos = 0
    if intrinsic_dim_L > 0:
        prompt_text_intrinsic = prompt_zip_np[current_pos : current_pos + intrinsic_dim_L]
        current_pos += intrinsic_dim_L
    if intrinsic_dim_V > 0:
        prompt_image_intrinsic = prompt_zip_np[current_pos : current_pos + intrinsic_dim_V]
        
    prompt_text_list = prompt_clip.generate_text_prompts([prompt_text_intrinsic] if prompt_text_intrinsic is not None else [None])
    prompt_image_list = prompt_clip.generate_visual_prompts([prompt_image_intrinsic] if prompt_image_intrinsic is not None else [None])
    
    original_parallel = prompt_clip.parallel
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = False
    fit_value = prompt_clip.eval([prompt_text_list[0], prompt_image_list[0]])
    prompt_clip.parallel = prompt_clip.text_encoder.parallel = prompt_clip.image_encoder.parallel = original_parallel

    return fit_value.item() if isinstance(fit_value, torch.Tensor) else fit_value

ndim_problem = 0
if intrinsic_dim_L > 0:
    ndim_problem += intrinsic_dim_L
if intrinsic_dim_V > 0:
    ndim_problem += intrinsic_dim_V

if ndim_problem == 0 and args.opt != "shallow_cma":
    logger.info("No prompts to tune (all dimensions are 0). Skipping optimization loop.")
    pro = {'fitness_function': fitness_eval, 'ndim_problem': 1}
else:
    pro = {'fitness_function': fitness_eval, 'ndim_problem': ndim_problem}


opt_cfg = {
    'fitness_threshold': 1e-10,
    'seed_rng': cfg.get('seed', 0),
    'budget': cfg.get('budget', 25200),
    'x': cfg.get('initial_mean', 0 * np.ones((ndim_problem,))) if ndim_problem > 0 else np.array([]),
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

if args.use_attacked_dataset:
    logger.info("--- Adversarial Attack Settings ---")
    surrogate_name = cfg['attack_surrogate_model'] if cfg['attack_surrogate_model'] else cfg['backbone']
    logger.info(f"Attack Generation Model (Surrogate): {surrogate_name}")
    logger.info(f"Attack Generation Prompt: '{cfg['attack_surrogate_prompt_text'].format('[CLASS]')}'")
    if args.attack_train:
        logger.info(f"Train Attack: Type={args.attack_type_train.upper()}, Ratio={args.attack_train_ratio}")
        if args.attack_type_train in ['pgd', 'fgsm']:
            logger.info(f"  -> PGD/FGSM Params: Epsilon={args.pgd_eps_train}, Alpha={args.pgd_alpha_train}, Steps={args.pgd_steps_train}")
        elif args.attack_type_train == 'cw':
            logger.info(f"  -> CW Params: C={args.cw_c}, LR={args.cw_lr}, Steps={args.cw_steps}")
    if args.attack_test:
        logger.info(f"Test Attack: Type={args.attack_type_test.upper()}, Ratio={args.attack_test_ratio}")
        if args.attack_type_test in ['pgd', 'fgsm']:
            logger.info(f"  -> PGD/FGSM Params: Epsilon={args.pgd_eps_test}, Alpha={args.pgd_alpha_test}, Steps={args.pgd_steps_test}")
        elif args.attack_type_test == 'cw':
            logger.info(f"  -> CW Params: C={args.cw_c}, LR={args.cw_lr}, Steps={args.cw_steps}")

if args.noise_type_text != 'none' or args.noise_type_visual != 'none':
    logger.info("--- Noise Injection Enabled During Tuning ---")
    logger.info(f"Text Noise: type={args.noise_type_text}, level={args.noise_level}")
    logger.info(f"Visual Noise: type={args.noise_type_visual}, level={args.noise_level}")


start_time = time.time()
# ... (optimization loop remains the same) ...
# ... (final evaluation and saving remain the same) ...
logger.info("--- Starting Optimization Loop ---")
if ndim_problem == 0:
    logger.info("Skipping optimization loop as ndim_problem is 0.")
    prompt_clip.best_prompt_text = None
    prompt_clip.best_prompt_image = None
else:
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
final_acc_primary = prompt_clip.test()
final_acc_clean_baseline = None

test_set_type = "Attacked" if args.attack_test else "Clean"
logger.info(f"Final {test_set_type} Accuracy: {final_acc_primary:.4f}")

if prompt_clip.test_loader_clean is not None:
    final_acc_clean_baseline = prompt_clip.test(use_clean_loader=True)
    logger.info(f"Final Clean (Baseline) Accuracy: {final_acc_clean_baseline:.4f}")


pth_filename = fname_base + "_final.pth"
final_results_path = os.path.join(output_dir, pth_filename)

content = {
    "task_name": args.task_name, "opt_name": cfg["opt_name"], "backbone": cfg["backbone"],
    "k_shot": cfg["k_shot"],
    "best_accuracy": prompt_clip.best_accuracy, "acc": prompt_clip.acc,
    "acc_clean_during_attack_run": prompt_clip.acc_clean_during_attack_run,
    "best_train_accuracy": prompt_clip.best_train_accuracy, "train_acc": prompt_clip.train_acc,
    "best_prompt_text": prompt_clip.best_prompt_text, "best_prompt_image": prompt_clip.best_prompt_image,
    "best_objective_loss_value": prompt_clip.best_objective_loss_value,
    "maximize_loss_setting": prompt_clip.maximize_loss,
    "loss": prompt_clip.loss, "num_call": prompt_clip.num_call,
    f"final_acc_{test_set_type.lower()}": final_acc_primary.item(),
    "final_acc_clean_baseline": final_acc_clean_baseline.item() if final_acc_clean_baseline is not None else "N/A",
    "Linear_L": prompt_clip.linear_L.state_dict(),
    "Linear_V": prompt_clip.linear_V.state_dict(),
    "optimization_time_seconds": optimization_time,
    "config_used": cfg,
    "args_used": vars(args)
}

Analysis_Util.save_results(content, output_dir, pth_filename)
logger.info(f"Final results saved to {final_results_path}")
logger.info("--- Run Complete ---")