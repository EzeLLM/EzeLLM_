import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding
from transformers import PretrainedConfig, PreTrainedModel, GPT2Tokenizer
import os
from typing import Optional, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import load_state_dict
import time
@dataclass
class LoadResult:
    model: "EzeLLM"
    tokenizer: "GPT2Tokenizer"

class EzeLLMConfig(PretrainedConfig):
    model_type = "ezellm"
    def __init__(
        self,
        block_size=2048,  # max sequence length
        vocab_size=50_304,  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
        hidden_count=20,  # number of layers
        head_count=16,  # number of heads
        embed_size=1024,  # embedding dimension
        batch_rows=3,
        n_kv_heads=8,
        **kwargs
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.hidden_count = hidden_count
        self.head_count = head_count
        self.embed_size = embed_size
        self.batch_rows = batch_rows
        self.n_kv_heads = n_kv_heads
        super().__init__(**kwargs)


class _attn_(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.head_count == 0
        
        self.n_heads = config.head_count
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else self.n_heads
        self.head_dim = config.embed_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Replication factor for key/value heads
        
        self.qkv_gen = nn.Linear(config.embed_size, self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim)
        
        self.projection_attn1 = nn.Linear(config.embed_size, config.embed_size)
        self.projection_attn1.SCALE_INIT = 1


        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        B, T, _ = x.size()
        
        # Generate q, k, v
        qkv = self.qkv_gen(x)
        
        # Split into q, k, v
        q, k, v = torch.split(qkv, [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # Replicate k and v heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # y = flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True)
        # y = y.to(dtype=torch.float32)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # Apply projection and SwiGLU activation
        y = self.projection_attn1(y)

        
        return y

class SwiGLU(torch.nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x):
        return F.silu(x) * F.sigmoid(x)

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ba_dense = nn.Linear(config.embed_size , int(5 * config.embed_size))
        self.swiglu = SwiGLU()
        self.aa_proj = nn.Linear(int(5 * config.embed_size), config.embed_size)
        self.aa_proj.SCALE_INIT = 1
    def forward(self,x):
        x = self.ba_dense(x)        
        x = self.swiglu(x)
        x = self.aa_proj(x)

        return x
    
class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.embed_size)
        self.attn = _attn_(config)
        self.ln2 = nn.RMSNorm(config.embed_size)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class EzeLLM(nn.Module):
    base_model_prefix = "ezellm"
    config_class = EzeLLMConfig
    def __init__(self,config=config_class()):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_size),
            hiddens = nn.ModuleList([Block(config) for _ in range(config.hidden_count)]),
            ln_f = nn.RMSNorm(config.embed_size),
        ))
        self.lm_head = nn.Sequential(
                    nn.Linear(config.embed_size,int(config.embed_size),bias=True),
                    nn.RMSNorm(int(config.embed_size)),
                    SwiGLU(),
                    nn.Linear(config.embed_size,int(config.embed_size),bias=True),
                    nn.RMSNorm(int(config.embed_size)),
                    SwiGLU(),
                    nn.Linear(int(config.embed_size),config.vocab_size,bias=False)
        )
        self.transformer.wte.weight = self.lm_head[-1].weight
        self.apply(self._init_weights)
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'SCALE_INIT'):
                std *= (2*self.config.hidden_count)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, max seq is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx)
        x = tok_emb 
        for block in self.transformer.hiddens:
            x = block(x)
        x = self.transformer.ln_f(x)
        for head in self.lm_head:
            logits = head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, os.PathLike],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> LoadResult:
        """
        Load a pretrained model and tokenizer from a directory.
        
        Args:
            pretrained_model_path: Path to the pretrained model directory
            device: Device to load the model on
            
        Returns:
            LoadResult containing the loaded model and tokenizer
        """
        # Load config
        config = EzeLLMConfig.from_pretrained(pretrained_model_path)
        
        # Initialize model with config
        model = cls(config)
        
        # Load state dict
        state_dict = torch.load(
            os.path.join(pretrained_model_path, "pytorch_model.bin"),
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.to(device)
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_path)
        
        return LoadResult(model=model, tokenizer=tokenizer)

    def generate(
        self,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.2,
        num_return_sequences: int = 1,
        top_k: int = 50,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> list[str]:
        """
        Generate text using the model.
        
        Args:
            tokenizer: Tokenizer to use
            prompt: Input prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            num_return_sequences: Number of sequences to generate
            top_k: Number of highest probability tokens to keep for top-k sampling
            device: Device to run inference on
            
        Returns:
            List of generated sequences
        """
        self.to(device)
        self.eval()
        
        # Encode input prompt
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        # Repeat for multiple sequences
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x = tokens.to(device)
        
        # Generate
        with torch.no_grad():
            t0 = time.time()
            while x.size(1) < max_length:
                # Get logits from model
                logits, _ = self(x)
                logits = logits[:, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                
                # Sample from top-k
                ix = torch.multinomial(topk_probs, num_samples=1)
                
                # Get selected tokens
                next_tokens = torch.gather(topk_indices, -1, ix)
                
                # Concatenate with input
                x = torch.cat((x, next_tokens), dim=1)
            t1 = time.time()
            print(f"Generated {max_length - len(tokenizer.encode(prompt))} tokens in {t1 - t0:.3f}s")
        # Decode all sequences
        generated_sequences = []
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            generated_sequences.append(decoded)
        
        return generated_sequences

path_to_ezellm = '/Users/ezelbayraktar/Documents/dev/tosafe/pusher/EzeLLM'

# Example usage:
result = EzeLLM.from_pretrained(path_to_ezellm)
model, tokenizer = result.model, result.tokenizer

outputs = model.generate(
    tokenizer=tokenizer,
    prompt="Researchers at EzeLLM labs are working on a revolutionary new Language Model (LM) that",
    max_length=100,
    temperature=0.7,
    num_return_sequences=2,
    top_k=100
)

for output in outputs:
    print("\n>>>", output)


path_to_ezellm = '/Users/ezelbayraktar/Documents/dev/tosafe/pusher/EzeLLM'



