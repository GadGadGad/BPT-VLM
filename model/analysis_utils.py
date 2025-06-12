import os
import torch
import errno
import json
import logging

logger = logging.getLogger(__name__)

class Analysis_Util:
    @staticmethod
    def mkdir_if_missing(dirname):
        """Create dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    @staticmethod
    def save_results(content, output_dir, fname):
        """Saves the given content dictionary to a .pth file."""
        Analysis_Util.mkdir_if_missing(output_dir)
        filename = os.path.join(output_dir, fname)
        torch.save(content, filename)

    @staticmethod
    def write_summary_to_txt(content, output_dir, pth_filename):
        """
        Loads data from a content dictionary and writes a human-readable summary to a .txt file.
        This version is robust against missing keys or non-numeric metric values.
        """
        Analysis_Util.mkdir_if_missing(output_dir)

        txt_filename = os.path.splitext(pth_filename)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)

        with open(txt_filepath, 'w') as f:
            f.write("--- Experiment Results Summary ---\n")
            f.write(f"Source: {pth_filename}\n\n")

            # Helper for formatting dictionaries nicely using json
            def format_dict_pretty(d):
                def tensor_serializer(obj):
                    if isinstance(obj, torch.Tensor):
                        return f"Tensor(shape={obj.shape}, dtype={obj.dtype})"
                    try:
                        return str(obj)
                    except TypeError:
                        return f"<Unserializable object: {type(obj).__name__}>"
                return json.dumps(d, indent=2, default=tensor_serializer)

            # --- Key Metrics (Robust Formatting) ---
            f.write("="*20 + " Key Metrics " + "="*20 + "\n")

            # Helper function for safe formatting
            def safe_format(value, specifier):
                return f"{value:{specifier}}" if isinstance(value, (int, float)) else "N/A"

            f.write(f"Final Clean Accuracy: {safe_format(content.get('final_acc_clean'), '.4f')}\n")
            f.write(f"Final PGD Accuracy: {safe_format(content.get('final_acc_pgd'), '.4f')}\n")
            f.write(f"Best Objective Loss Value: {safe_format(content.get('best_objective_loss_value'), '.6f')}\n")
            f.write(f"  - Maximizing Loss: {content.get('maximize_loss_setting', 'N/A')}\n")
            f.write(f"Best Test Clean Accuracy (during run): {safe_format(content.get('best_accuracy'), '.4f')}\n")
            f.write(f"Best Test PGD Accuracy (during run): {safe_format(content.get('best_accuracy_pgd'), '.4f')}\n")
            f.write(f"Best Train Accuracy (during run): {safe_format(content.get('best_train_accuracy'), '.4f')}\n")
            f.write(f"Optimization Time (s): {safe_format(content.get('optimization_time_seconds'), '.2f')}\n")
            f.write(f"Number of Evals: {content.get('num_call', 'N/A')}\n\n")

            # --- Configurations ---
            f.write("="*20 + " Configurations " + "="*20 + "\n")
            f.write("--- Command Line Arguments ---\n")
            f.write(format_dict_pretty(content.get('args_used', {})) + "\n\n")
            f.write("--- Full Config Used ---\n")
            f.write(format_dict_pretty(content.get('config_used', {})) + "\n\n")


            # --- Saved Tensors & Models ---
            f.write("="*20 + " Saved Tensors & Models " + "="*20 + "\n")
            for key, value in content.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"Tensor '{key}':\n  Shape: {value.shape}\n  Dtype: {value.dtype}\n")
                elif isinstance(value, dict) and key in ["Linear_L", "Linear_V"]:
                    f.write(f"State Dict '{key}':\n")
                    for sk, sv in value.items():
                        f.write(f"  - {sk}: Tensor(shape={sv.shape})\n")

            snapshot = content.get('training_dataset_snapshot')
            if snapshot and isinstance(snapshot, dict):
                img_tensor = snapshot.get('images')
                lbl_tensor = snapshot.get('labels')
                img_shape = img_tensor.shape if isinstance(img_tensor, torch.Tensor) else 'N/A'
                lbl_shape = lbl_tensor.shape if isinstance(lbl_tensor, torch.Tensor) else 'N/A'
                f.write(f"Tensor 'training_dataset_snapshot[images]':\n  Shape: {img_shape}\n")
                f.write(f"Tensor 'training_dataset_snapshot[labels]':\n  Shape: {lbl_shape}\n")
            f.write("\n")

            # --- Historical Data (Robust Formatting) ---
            f.write("="*20 + " Historical Data (Last 10 Entries) " + "="*20 + "\n")
            for key in ["acc", "acc_pgd", "train_acc"]:
                history = content.get(key)
                if isinstance(history, list) and history:
                    # Filter out non-numeric types before formatting
                    formatted_history = [f"{item:.4f}" for item in history[-10:] if isinstance(item, (int, float))]
                    f.write(f"History '{key}' (len={len(history)}): {formatted_history}\n")

            loss_history = content.get('loss')
            if isinstance(loss_history, list) and loss_history:
                 if isinstance(loss_history[0], list): # Parallel mode
                     last_entries = [f"len={len(sublist)}" for sublist in loss_history[-5:]]
                     f.write(f"History 'loss' (len={len(loss_history)}, nested): Last 5 entries' lengths: {last_entries}\n")
                 else: # Sequential mode
                     formatted_history = [f"{item:.6f}" for item in loss_history[-10:] if isinstance(item, (int, float))]
                     f.write(f"History 'loss' (len={len(loss_history)}): {formatted_history}\n")


            f.write("\n--- End of Summary ---\n")
        logger.info(f"Human-readable summary saved to {txt_filepath}")