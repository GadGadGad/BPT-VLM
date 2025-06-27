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

pgd_gen_group = parser.add_argument_group('Pre-Attack Dataset Generation Parameters')
pgd_gen_group.add_argument('--enable_pre_attack_gen', action='store_true',
                           help='Enable dynamic generation of a pre-attacked dataset. Task name must end with "_PGD_GEN".')
pgd_gen_group.add_argument('--pre_attack_gen_epsilon', type=float, default=8/255, help='Epsilon for pre-attack generation.')
pgd_gen_group.add_argument('--pre_attack_gen_alpha', type=float, default=2/255, help='Alpha for pre-attack generation.')
pgd_gen_group.add_argument('--pre_attack_gen_num_iter', type=int, default=10, help='Number of iterations for pre-attack generation.')
pgd_gen_group.add_argument('--pre_attack_gen_ratio', type=float, default=1.0,
                           help='Ratio of the dataset to be pre-attacked (0.0 to 1.0). 1.0 means fully attacked, 0.0 is fully clean. Used with --enable_pre_attack_gen.')


parser.add_argument("--maximize_loss", action='store_true', help='Tune prompts to maximize the loss instead of minimizing it')
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

if args.task_name in cfg:
    for k,v in cfg[args.task_name].items():
        cfg[k]=v
else:
    logging.warning(f"Task '{args.task_name}' not found in config. Using default settings.")

if 'pre_attack_gen' not in cfg:
    cfg['pre_attack_gen'] = {}
cfg['pre_attack_gen']['enabled'] = args.enable_pre_attack_gen
cfg['pre_attack_gen']['epsilon'] = args.pre_attack_gen_epsilon
cfg['pre_attack_gen']['alpha'] = args.pre_attack_gen_alpha
cfg['pre_attack_gen']['num_iter'] = args.pre_attack_gen_num_iter
cfg['pre_attack_gen']['ratio'] = args.pre_attack_gen_ratio

output_dir = os.path.join(cfg["output_dir"], args.task_name)
Analysis_Util.mkdir_if_missing(output_dir)

initial_prompt_str_fn = f"_initPrompt" if cfg["initial_prompt_text"] is not None else ""
learned_pos_str_fn = f"_pos{cfg['learned_prompt_pos']}"
pre_attack_gen_ratio_str_fn = f"_ratio{cfg['pre_attack_gen']['ratio']}" if cfg['pre_attack_gen']['enabled'] and cfg['pre_attack_gen']['ratio'] < 1.0 else ""
pre_attack_gen_str_fn = f"_preAttackGen{pre_attack_gen_ratio_str_fn}" if cfg['pre_attack_gen']['enabled'] else ""

fname_base = "{}{}_{}_{}_parallel{}{}{}_maxLoss{}".format(
    cfg["k_shot"],
    args.task_name,
    cfg["opt_name"],
    cfg["backbone"].replace("/", "-"),
    args.parallel,
    initial_prompt_str_fn,
    learned_pos_str_fn,
    pre_attack_gen_str_fn,
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

if args.task_name in __classification__ or args.task_name.endswith("_PGD_GEN"):
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
cfg['budget'] = cfg.get('budget', 25200)

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

if cfg['pre_attack_gen']['enabled']:
    gen_eps = cfg['pre_attack_gen'].get('epsilon')
    gen_alpha = cfg['pre_attack_gen'].get('alpha')
    gen_iter = cfg['pre_attack_gen'].get('num_iter')
    gen_ratio = cfg['pre_attack_gen'].get('ratio')
    logger.info(f"Pre-Attacked Dataset Generation: ENABLED (Ratio: {gen_ratio*100:.0f}% Attacked)")
    logger.info(f"  Generation Params: Epsilon={gen_eps}, Alpha={gen_alpha}, Iter={gen_iter}")
else:
    logger.info(f"Using Clean Dataset (Pre-Attack Generation is DISABLED)")


start_time = time.time()
logger.info("--- Starting Optimization Loop ---")

# The prompt_clip object is now fully initialized, so no more set_context calls are needed.
if args.opt in __pypop__:
    if args.task_name in __classification__ or args.task_name.endswith("_PGD_GEN"):
        res = opt.optimize()
        logger.info(f"Optimization Result (PyPop): {res}")
    else:
        logger.warning(f"PyPop optimizer path not fully defined for task {args.task_name}")

else:
    if args.task_name in __classification__ or args.task_name.endswith("_PGD_GEN"):
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
            if prompt_clip.num_call % (cfg["popsize"] * opt_cfg['verbose_frequency']) == 0:
                 log_loss_label = "Maximized Loss" if prompt_clip.maximize_loss else "Minimized Loss"
                 dataset_type_str = f" ({(prompt_clip.pre_attack_gen_config.get('ratio', 0)*100):.0f}% Attacked Dataset)" if prompt_clip.pre_attack_gen_config['enabled'] else " (Clean Dataset)"
                 logger.info(f"Generation ~{int(prompt_clip.num_call / cfg['popsize'])}, Best Objective ({log_loss_label}{dataset_type_str}): {prompt_clip.best_objective_loss_value:.4f}")

    else:
        logger.warning(f"Non-PyPop optimizer path not fully defined for task {args.task_name}")

logger.info("\n--- Optimization Finished ---")
end_time = time.time()
optimization_time = end_time - start_time
logger.info(f"Total Optimization Time: {optimization_time:.2f} seconds")

logger.info("\n--- Final Evaluation using Best Prompts ---")
final_accuracy = prompt_clip.test()
dataset_type_str = f"{(prompt_clip.pre_attack_gen_config.get('ratio', 0)*100):.0f}% Attacked" if prompt_clip.pre_attack_gen_config['enabled'] else "Clean"
logger.info(f"Final Accuracy on {dataset_type_str} Test Set: {final_accuracy:.4f}")

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
    "final_accuracy": final_accuracy.item(),
    "Linear_L": prompt_clip.linear_L.state_dict(),
    "Linear_V": prompt_clip.linear_V.state_dict(),
    "pre_attack_gen_config": prompt_clip.pre_attack_gen_config,
    "optimization_time_seconds": optimization_time,
    "config_used": cfg,
    "args_used": vars(args)
}

Analysis_Util.save_results(content, output_dir, pth_filename)
logger.info(f"Final results saved to {final_results_path}")
logger.info("--- Run Complete ---")