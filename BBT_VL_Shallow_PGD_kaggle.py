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
parser.add_argument("--adv_train", action='store_true', help='Enable Adversarial Training')
parser.add_argument("--test_every_n_gens", type=int, default=None, help="Run test evaluation every N generations. If not set, testing only occurs at the very end.")


prompt_group = parser.add_argument_group('Initial Prompt Configuration')
prompt_group.add_argument("--initial_prompt_text", type=str, default=None, help="Initial text prompt (e.g., 'a photo of a'). If None, no initial prompt is used.")
prompt_group.add_argument("--learned_prompt_pos", type=str, default="prefix", choices=["prefix", "middle", "suffix"], help="Position of the learned prompt relative to the initial prompt and class name.")

test_attack_group = parser.add_argument_group('Test-Time Attack Parameters')
test_attack_group.add_argument('--enable_test_attack', action='store_true', help='Enable adversarial attack during final testing.')
test_attack_group.add_argument('--test_attack_type', type=str, default='pgd', choices=['pgd', 'fgsm', 'gaussian'], help='Type of adversarial attack for testing.')
test_attack_group.add_argument('--test_attack_epsilon', type=float, default=8/255, help='Epsilon for the test attack.')
test_attack_group.add_argument('--test_attack_alpha', type=float, default=2/255, help='Alpha/step size for the test attack (used by PGD).')
test_attack_group.add_argument('--test_attack_num_iter', type=int, default=10, help='Number of iterations for the test attack (used by PGD).')


adv_train_group = parser.add_argument_group('Adversarial Training Parameters (for optimization loss)')
adv_train_group.add_argument("--adv_train_epsilon", type=float, default=4/255, help="PGD epsilon for adversarial training (e.g., 4/255 or 8/255)")
adv_train_group.add_argument("--adv_train_alpha", type=float, default=1/255, help="PGD alpha/step size for adversarial training (e.g., eps/4)")
adv_train_group.add_argument("--adv_train_num_iter", type=int, default=10, help="PGD number of iterations for adversarial training (e.g., 10)")
adv_train_group.add_argument("--adv_train_all", action='store_true', help="Use PGD Adv Tuning for whole process.")
adv_train_group.add_argument("--adv_train_attack_prompt_type", type=str, default="on-the-fly", choices=["constant", "on-the-fly", "perturbed"], help="Strategy for selecting text prompt during adversarial example generation for training loss.")
adv_train_group.add_argument("--adv_train_alpha_text_prompt", type=float, default=0.01, help="PGD alpha/step size for perturbing text prompt itself (if adv_train_attack_prompt_type is 'perturbed')")
adv_train_group.add_argument("--adv_train_sample_ratio", type=float, default=1.0, help="Ratio of few-shot samples to apply PGD to during adversarial training (0.0 to 1.0). 1.0 means all samples.")
adv_train_group.add_argument("--adv_train_attack_type", type=str, default="pgd", choices=["pgd", "fgsm", "gaussian"], help="Type of adversarial attack for training (PGD, FGSM, or Gaussian noise).")


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

if 'test_attack' not in cfg:
    cfg['test_attack'] = {}
cfg['test_attack']['enabled'] = args.enable_test_attack
cfg['test_attack']['attack_type'] = args.test_attack_type
cfg['test_attack']['epsilon'] = args.test_attack_epsilon
cfg['test_attack']['alpha'] = args.test_attack_alpha
cfg['test_attack']['num_iter'] = args.test_attack_num_iter

if 'adv_train' not in cfg:
    cfg['adv_train'] = {}
cfg['adv_train']['epsilon'] = args.adv_train_epsilon
cfg['adv_train']['alpha'] = args.adv_train_alpha
cfg['adv_train']['num_iter'] = args.adv_train_num_iter
cfg['adv_train']['enabled'] = args.adv_train
cfg['adv_train']['all_call'] = args.adv_train_all
cfg['adv_train']['attack_prompt_type'] = args.adv_train_attack_prompt_type
cfg['adv_train']['alpha_text_prompt'] = args.adv_train_alpha_text_prompt
cfg['adv_train']['sample_ratio'] = args.adv_train_sample_ratio
cfg['adv_train']['attack_type'] = args.adv_train_attack_type


output_dir = os.path.join(cfg["output_dir"], args.task_name)
Analysis_Util.mkdir_if_missing(output_dir)

adv_train_attack_type_str_fn = f"_advAttackType{cfg['adv_train']['attack_type']}" if cfg['adv_train']['enabled'] else ""
adv_train_attack_prompt_type_str_fn = f"_advPromptGen{cfg['adv_train']['attack_prompt_type']}" if cfg['adv_train']['enabled'] else ""
adv_train_sample_ratio_str_fn = f"_advSampleRatio{cfg['adv_train']['sample_ratio']}" if cfg['adv_train']['enabled'] and cfg['adv_train']['sample_ratio'] < 1.0 else ""

initial_prompt_str_fn = f"_initPrompt" if cfg["initial_prompt_text"] is not None else ""
learned_pos_str_fn = f"_pos{cfg['learned_prompt_pos']}"

test_attack_str_fn = f"_testAttack{cfg['test_attack']['attack_type']}" if cfg['test_attack']['enabled'] else "_testAttackNone"

fname_base = "{}{}_{}_{}_parallel{}_advTrain{}{}{}{}{}{}_maxLoss{}".format(
    cfg["k_shot"],
    args.task_name,
    cfg["opt_name"],
    cfg["backbone"].replace("/", "-"),
    args.parallel,
    initial_prompt_str_fn,
    learned_pos_str_fn,
    cfg["adv_train"]["enabled"],
    adv_train_attack_type_str_fn,
    adv_train_attack_prompt_type_str_fn,
    adv_train_sample_ratio_str_fn,
    test_attack_str_fn,
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
logger.info(f"Adversarial Training (during optimization): {cfg['adv_train']['enabled']}")
if cfg['adv_train']['enabled']:
    adv_eps_cfg = cfg['adv_train'].get('epsilon', "Default in PromptCLIP")
    adv_alpha_cfg = cfg['adv_train'].get('alpha', "Default in PromptCLIP")
    adv_iter_cfg = cfg['adv_train'].get('num_iter', "Default in PromptCLIP")
    adv_attack_prompt_type_cfg = cfg['adv_train'].get('attack_prompt_type', 'on-the-fly')
    adv_alpha_text_prompt_cfg = cfg['adv_train'].get('alpha_text_prompt', 'N/A' if adv_attack_prompt_type_cfg != 'perturbed' else 'Default in PromptCLIP')
    adv_sample_ratio_cfg = cfg['adv_train'].get('sample_ratio', 1.0)
    adv_attack_type_cfg = cfg['adv_train'].get('attack_type', 'pgd')
    logger.info(f"  Adv. Training Params (from cfg): Epsilon={adv_eps_cfg}, Alpha_Img={adv_alpha_cfg}, Iter={adv_iter_cfg}, SampleRatio={adv_sample_ratio_cfg}")
    logger.info(f"  Training Attack Type: {adv_attack_type_cfg.upper()}")
    logger.info(f"  Adv. Training Attack Prompt Type: {adv_attack_prompt_type_cfg}")
    if adv_attack_prompt_type_cfg == 'perturbed':
        logger.info(f"  Adv. Training Alpha for Text Prompt Perturbation: {adv_alpha_text_prompt_cfg}")
logger.info(f"Attack during Final Test: {cfg['test_attack']['enabled']}")
if cfg['test_attack']['enabled']:
    test_attack_type_cfg = cfg['test_attack'].get('attack_type', "N/A")
    test_attack_eps_cfg = cfg['test_attack'].get('epsilon', "N/A")
    test_attack_alpha_cfg = cfg['test_attack'].get('alpha', "N/A")
    test_attack_iter_cfg = cfg['test_attack'].get('num_iter', "N/A")
    logger.info(f"  Test Attack Type: {test_attack_type_cfg.upper()}")
    logger.info(f"  Test Attack Params (from cfg): Epsilon={test_attack_eps_cfg}, Alpha={test_attack_alpha_cfg}, Iter={test_attack_iter_cfg}")


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
                 adv_train_info_str = f" (AdvTrain: {prompt_clip.adv_train_config['enabled']}, AttackType: {prompt_clip.adv_train_attack_type}, AttackGen: {prompt_clip.adv_train_attack_prompt_type}"
                 if prompt_clip.adv_train_config['enabled'] and prompt_clip.adv_train_config.get('sample_ratio', 1.0) < 1.0:
                     adv_train_info_str += f", SampleRatio: {prompt_clip.adv_train_config.get('sample_ratio', 1.0)}"
                 adv_train_info_str += ")" if prompt_clip.adv_train_config['enabled'] else ""
                 
                 # The detailed test accuracy log is now inside prompt_clip.eval(), so we only log the objective here
                 logger.info(f"Generation ~{int(prompt_clip.num_call / cfg['popsize'])}, Best Objective ({log_loss_label}{adv_train_info_str}): {prompt_clip.best_objective_loss_value:.4f}")

    else:
        logger.warning(f"Non-PyPop optimizer path not fully defined for task {args.task_name}")

logger.info("\n--- Optimization Finished ---")
end_time = time.time()
optimization_time = end_time - start_time
logger.info(f"Total Optimization Time: {optimization_time:.2f} seconds")

logger.info("\n--- Final Evaluation using Best Prompts ---")
final_acc_clean = prompt_clip.test(attack_config=None)
logger.info(f"Final Clean Accuracy: {final_acc_clean:.4f}")

final_acc_attack = torch.tensor(0.0)
if cfg['test_attack']['enabled']:
    if prompt_clip.best_prompt_text is not None:
        final_acc_attack = prompt_clip.test(attack_config=prompt_clip.test_attack_config)
        attack_name = cfg['test_attack']['attack_type'].upper()
        logger.info(f"Final {attack_name} Accuracy  : {final_acc_attack:.4f}")
    else:
        logger.info("Final Attacked Accuracy  : Skipped (no best prompt available)")
        final_acc_attack = None
else:
    logger.info("Final Attacked Accuracy  : Skipped (test attack not enabled in args/config)")
    final_acc_attack = None

pth_filename = fname_base + "_final.pth"
final_results_path = os.path.join(output_dir, pth_filename)

content = {
    "task_name": args.task_name, "opt_name": cfg["opt_name"], "backbone": cfg["backbone"],
    "k_shot": cfg["k_shot"],
    "best_accuracy": prompt_clip.best_accuracy, "acc": prompt_clip.acc,
    "best_train_accuracy": prompt_clip.best_train_accuracy, "train_acc": prompt_clip.train_acc,
    "best_accuracy_attack": prompt_clip.best_accuracy_attack, "acc_attack": prompt_clip.acc_attack,
    "best_prompt_text": prompt_clip.best_prompt_text, "best_prompt_image": prompt_clip.best_prompt_image,
    "best_objective_loss_value": prompt_clip.best_objective_loss_value,
    "maximize_loss_setting": prompt_clip.maximize_loss,
    "loss": prompt_clip.loss, "num_call": prompt_clip.num_call,
    "final_acc_clean": final_acc_clean.item(),
    "final_acc_attack": final_acc_attack.item() if isinstance(final_acc_attack, torch.Tensor) else final_acc_attack,
    "Linear_L": prompt_clip.linear_L.state_dict(),
    "Linear_V": prompt_clip.linear_V.state_dict(),
    "test_attack_config": prompt_clip.test_attack_config,
    "adv_train_config": prompt_clip.adv_train_config,
    "optimization_time_seconds": optimization_time,
    "config_used": cfg,
    "args_used": vars(args)
}

Analysis_Util.save_results(content, output_dir, pth_filename)
logger.info(f"Final results saved to {final_results_path}")
logger.info("--- Run Complete ---")