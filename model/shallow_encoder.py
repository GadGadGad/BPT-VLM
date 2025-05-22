import torch
import torch.nn as nn
import numpy as np

# ... (TextEncoder class remains the same) ...
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def set_context(self,context):
        self.n_cls = context["n_cls"]
        self.n_prompt_tokens_L = context["n_prompt_tokens_L"]
        self.init_pattern_embedding = context["init_pattern_embedding"]
        self.tokenized_pattern_prompts= context["tokenized_pattern_prompts"]
        self.batch_size = context["batch_size"] # original batch size
        self.parallel = context["parallel"]
        self.pop_size = context["pop_size"]
        if self.parallel:
            self.init_pattern_embedding = self.init_pattern_embedding.repeat(self.pop_size,1,1)
            self.tokenized_pattern_prompts = self.tokenized_pattern_prompts.repeat(self.pop_size,1)

    def incorporate_prompt(self, prompt, embedding):
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_prompt_tokens_L:, :]
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0).expand(self.n_cls, -1, -1)
        x = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                prompt,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return x

    def incorporate_prompt_parallel(self,prompt,embedding):
        prefix = embedding[:, :1, :] # (n_cls * popsize, 1, dim)
        suffix = embedding[:, 1 + self.n_prompt_tokens_L:, :] # (n_cls * popsize, 1, dim)
        x = []
        for index,pt in enumerate(prompt):
            if pt.dim() == 2:
                pt = pt.unsqueeze(0).expand(self.n_cls, -1, -1)
            start = index * self.n_cls
            pfx = prefix[start:start+self.n_cls]
            sfx = suffix[start:start+self.n_cls]
            tmp_x = torch.cat(
                [
                    pfx,  # (n_cls, 1, dim)
                    pt,  # (n_cls, n_ctx, dim)
                    sfx,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            x.append(tmp_x)
        x = torch.cat(x, dim=0)
        return x

    def forward(self, prompt):
        if self.parallel:
            x = self.incorporate_prompt_parallel(prompt,self.init_pattern_embedding)
        else:
            x = self.incorporate_prompt(prompt,self.init_pattern_embedding[:self.n_cls])
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        if self.parallel:
            x = x[torch.arange(x.shape[0]), self.tokenized_pattern_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), self.tokenized_pattern_prompts[:self.n_cls].argmax(dim=-1)] @ self.text_projection
        return x


class VisionEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.input_resolution = clip_model.visual.input_resolution
        self.patch_size = clip_model.visual.patch_size
        self.prefix_len = (self.input_resolution//self.patch_size)**2+1 # Number of CLS token + patch tokens
        self.output_dim = clip_model.visual.output_dim
        self.conv1 = clip_model.visual.conv1
        self.width = clip_model.visual.width
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.tranformer = clip_model.visual.transformer # Typo: should be 'transformer'
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        if hasattr(clip_model.visual, 'transformer'):
            self.transformer_layer = clip_model.visual.transformer


    def set_context(self,context):
        self.n_prompt_tokens_V = context["n_prompt_tokens_V"]
        self.batch_size = context["batch_size"] # original batch size
        self.parallel = context["parallel"]
        self.pop_size = context["pop_size"]

    def incorporate_prompt(self, prompt, embedding):
        if prompt is None: # If no visual prompt is provided
            return embedding # Return the embeddings unmodified

        B = embedding.shape[0]
        # Original logic to append prompt tokens:
        expanded_prompt = prompt.expand(B, -1, -1) # This is now safe
        embedding = torch.cat((
            embedding[:, :self.prefix_len, :], # All original tokens (CLS + patches)
            expanded_prompt,                   # Append the visual prompt tokens
        ), dim=1)
        return embedding

    def incorporate_prompt_parallel(self, prompt_list, embedding): # Renamed 'prompt' to 'prompt_list'
        # embedding: (batch_size*popsize, num_original_tokens, width)
        # prompt_list: list of prompt tensors, one for each population member; can contain None
        B_per_member = int(embedding.shape[0] / self.pop_size)
        processed_embeddings_list = []

        for index, pt_member in enumerate(prompt_list): # pt_member is the prompt for one population member
            start_idx = index * B_per_member
            end_idx = start_idx + B_per_member
            current_member_embeddings = embedding[start_idx:end_idx, :, :]

            if pt_member is None: # If this population member has no visual prompt
                processed_embeddings_list.append(current_member_embeddings)
            else:
                # This member has a visual prompt
                expanded_pt_member = pt_member.expand(B_per_member, -1, -1) # This is now safe
                # Concatenate original tokens with this member's visual prompt tokens
                embedding_with_prompt = torch.cat((
                    current_member_embeddings[:, :self.prefix_len, :],
                    expanded_pt_member,
                ), dim=1)
                processed_embeddings_list.append(embedding_with_prompt)
        
        final_embeddings = torch.cat(processed_embeddings_list, dim=0)
        return final_embeddings

    def forward(self, x, prompt): # x is image, prompt is image_prompt (tensor or list of tensors)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # Add class token
        class_emb_expanded = self.class_embedding.to(x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat([class_emb_expanded, x], dim=1)
        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Incorporate visual prompt(s)
        if self.parallel:
            x = self.incorporate_prompt_parallel(prompt, x)
        else:
            x = self.incorporate_prompt(prompt, x)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer_layer(x) # Use the corrected attribute name
        x = x.permute(1, 0, 2) # LND -> NLD
        
        # Get CLS token output
        x = self.ln_post(x[:, 0, :]) 
        
        if self.proj is not None:
            x = x @ self.proj
        return x