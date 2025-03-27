import os
import argparse
import torch
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip

def load_clip_to_cpu(backbone_name="ViT-B/32"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="./")
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

parser = argparse.ArgumentParser()
parser.add_argument("fpath", type=str, help="Path to the learned prompt")
parser.add_argument("topk", type=int, help="Select top-k similar words")
args = parser.parse_args()

fpath = args.fpath
topk = args.topk
assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location='cpu')
ctx = prompt_learner['best_prompt_text']
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    num_ctx_tokens = ctx.shape[0]

    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    print("\n--- Top-{topk} words for each context vector position ---")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk] 

    for m in range(num_ctx_tokens):
        idxs = sorted_idxs[m]
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"w_{m+1}: {words} {dist}")

    print("\n--- Best matching sequence (closest word per position) ---")
    best_idxs = torch.argmin(distance, dim=1) 
    best_words_sequence = [tokenizer.decoder[idx.item()] for idx in best_idxs]
    best_sentence = " ".join(best_words_sequence)
    print(best_sentence)

elif ctx.dim() == 3:
    raise NotImplementedError