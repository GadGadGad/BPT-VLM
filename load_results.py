import torch
import argparse
import os
import sys

# Add the project root to the Python path to allow importing 'model'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model.analysis_utils import Analysis_Util
except ImportError:
    print("Error: Could not import Analysis_Util. Make sure 'model/analysis_utils.py' exists and this script is run from the project root.")
    sys.exit(1)

def main():
    """
    Command-line tool to load a .pth result file and generate a .txt summary.
    """
    parser = argparse.ArgumentParser(description="Load a .pth result file and generate a human-readable .txt summary.")
    parser.add_argument("pth_file", type=str, help="Path to the .pth result file.")
    args = parser.parse_args()

    if not os.path.exists(args.pth_file):
        print(f"Error: File not found at {args.pth_file}")
        return

    print(f"Loading data from {args.pth_file}...")
    try:
        # map_location ensures it can be loaded on CPU-only machines
        content = torch.load(args.pth_file, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading .pth file: {e}")
        return

    output_dir = os.path.dirname(args.pth_file)
    pth_filename = os.path.basename(args.pth_file)

    print("Generating summary...")
    try:
        # Call the static method from the utility class
        Analysis_Util.write_summary_to_txt(content, output_dir, pth_filename)
        txt_filename = os.path.splitext(pth_filename)[0] + ".txt"
        print(f"Successfully created summary file: {os.path.join(output_dir, txt_filename)}")
    except Exception as e:
        print(f"An error occurred while writing the summary: {e}")

if __name__ == '__main__':
    main()