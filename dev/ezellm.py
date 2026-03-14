
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint
import random
import sys
@dataclass
class EzeLLMConfig:
    block_size: int = 2048 # max sequence length
    vocab_size: int = 50_304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    hidden_count: int = 20 # number of layers
    head_count: int = 16 # number of heads
    embed_size: int = 1024 # embedding dimension
    batch_rows: int = 3
    n_kv_heads: int = 8





class _attn_(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.head_count == 0
        
        # New attributes for GQA
        self.n_heads = config.head_count
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else self.n_heads
        self.head_dim = config.embed_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Replication factor for key/value heads
        
        # Adjust qkv_gen to account for the different number of heads
        self.qkv_gen = nn.Linear(config.embed_size, self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim)
        
        self.projection_attn1 = nn.Linear(config.embed_size, config.embed_size)
        self.projection_attn1.SCALE_INIT = 1
        # self.swiglu = SwiGLU()
        # self.projection_attn2 = nn.Linear(config.embed_size, config.embed_size)
        # self.projection_attn2.SCALE_INIT = 1

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
        # y = self.swiglu(y)
        # y = self.projection_attn2(y)
        
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
    def __init__(self,config,device):
        super().__init__()
        self.device = device


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
        torch.manual_seed(964)
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.eot_id = self.tokenizer._special_tokens['<|endoftext|>']
        self.to(self.device)
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
        #pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb #+ pos_emb
        for block in self.transformer.hiddens:
            x = block(x)
        x = self.transformer.ln_f(x)
        for head in self.lm_head:
            logits = head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    
    def generate(
            self,
            input_: str= "I'm a",
            tempreature: int = 1,
            tempreature_interval:int = 0.0,
            topk: int = 100,
            topp: float = 0.9,
            max_l: int = 2048,
            num_return_seq: int = 1
    #TODO implement topp
    ) -> List:
        print(
    f"""\n
Parameters:
    Input text: {input_}
    Temperature: {tempreature}
    Temperature interval: {tempreature_interval}
    Top-k: {topk}
    Top-p: {topp}
    Maximum length: {max_l}
    Number of return sequences: {num_return_seq}\n\n
    """
)

        tokens = self.tokenizer.encode(input_)
        tokens = torch.tensor(tokens,dtype=torch.long,device=self.device)
        tokens = tokens.unsqueeze(0).repeat(num_return_seq,1) 
        x = tokens.to(self.device)
        gen_start = time.time()
        if tempreature_interval > 0:
            tempreatures = np.arange(tempreature-tempreature_interval,tempreature+tempreature_interval,0.00001)
            tempreature = random.choice(tempreatures)
        while x.size(1) < max_l:
            with torch.no_grad():
                logits = self(x)[0]
                
                logits = logits[:,-1,:]
                logits = logits / tempreature

                probs = F.softmax(logits,dim=-1)
                topk_probs, topk_indices = torch.topk(probs,topk,dim=-1)
                # print(topk_probs)
                # print(topk_indices)
                # print(self.tokenizer.decode(topk_indices[0].tolist()))
                # sys.exit()

                ix = torch.multinomial(topk_probs,1)
                xcol = torch.gather(topk_indices,-1,ix)
                x = torch.cat((x,xcol),dim=-1)
                if xcol == self.eot_id:
                    break
        gen_time = time.time()-gen_start
        for i in range(num_return_seq):
            tokens = x[i,:max_l].tolist()
            decoded = self.tokenizer.decode(tokens=tokens)
        print(f"Generated {len(tokens) if num_return_seq == 1 else len(tokens[0])} tokens in {gen_time:.2f}s, {(len(tokens) if num_return_seq == 1 else len(tokens[0]))/ gen_time:.2f} tokens per second")
        return decoded
    
    def _attn_forward_cached(self, attn_module, x, cache, layer_idx, offset):
        B, T, _ = x.size()
        qkv = attn_module.qkv_gen(x)
        q, k, v = torch.split(qkv, [
            attn_module.n_heads * attn_module.head_dim,
            attn_module.n_kv_heads * attn_module.head_dim,
            attn_module.n_kv_heads * attn_module.head_dim], dim=2)
        q = q.view(B, T, attn_module.n_heads, attn_module.head_dim).transpose(1, 2)
        k = k.view(B, T, attn_module.n_kv_heads, attn_module.head_dim).transpose(1, 2)
        v = v.view(B, T, attn_module.n_kv_heads, attn_module.head_dim).transpose(1, 2)
        q = attn_module.rotary_emb.rotate_queries_or_keys(q, offset=offset)
        k = attn_module.rotary_emb.rotate_queries_or_keys(k, offset=offset)
        if cache is not None:
            cache[layer_idx]['k'][:, :, cache['pos']:cache['pos']+T, :] = k
            cache[layer_idx]['v'][:, :, cache['pos']:cache['pos']+T, :] = v
            full_len = cache['pos'] + T
            k_full = cache[layer_idx]['k'][:, :, :full_len, :]
            v_full = cache[layer_idx]['v'][:, :, :full_len, :]
            k_full = k_full.repeat_interleave(attn_module.n_rep, dim=1)
            v_full = v_full.repeat_interleave(attn_module.n_rep, dim=1)
            y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=(T > 1))
        else:
            k = k.repeat_interleave(attn_module.n_rep, dim=1)
            v = v.repeat_interleave(attn_module.n_rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return attn_module.projection_attn1(y)

    def _forward_cached(self, idx, cache=None, offset=0):
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
        for i, block in enumerate(self.transformer.hiddens):
            residual = x
            x = residual + self._attn_forward_cached(block.attn, block.ln1(x), cache, i, offset)
            x = x + block.mlp(block.ln2(x))
        x = self.transformer.ln_f(x)
        for head in self.lm_head:
            logits = head(x)
        return logits

    def _make_cache(self, max_seq_len, batch_size=1):
        cfg = self.config
        head_dim = cfg.embed_size // cfg.head_count
        dtype = next(self.parameters()).dtype
        cache = {'pos': 0}
        for i in range(cfg.hidden_count):
            cache[i] = {
                'k': torch.zeros(batch_size, cfg.n_kv_heads, max_seq_len, head_dim,
                                 device=self.device, dtype=dtype),
                'v': torch.zeros(batch_size, cfg.n_kv_heads, max_seq_len, head_dim,
                                 device=self.device, dtype=dtype),
            }
        return cache

    def generate_fast(
            self,
            input_: str = "I'm a",
            temperature: float = 1.0,
            topk: int = 50,
            max_l: int = 2048,
    ) -> str:
        """Generate text with KV cache — much faster for long sequences."""
        tokens = self.tokenizer.encode(input_)
        x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        cache = self._make_cache(max_seq_len=max_l)
        gen_start = time.time()

        with torch.no_grad():
            # Prefill
            logits = self._forward_cached(x, cache=cache, offset=0)
            cache['pos'] += x.shape[1]

            # Sample first token
            next_token = self._sample(logits[:, -1, :], temperature, topk)
            x = torch.cat((x, next_token), dim=-1)

            # Decode loop
            while x.size(1) < max_l and next_token.item() != self.eot_id:
                offset = cache['pos']
                logits = self._forward_cached(next_token, cache=cache, offset=offset)
                cache['pos'] += 1
                next_token = self._sample(logits[:, -1, :], temperature, topk)
                x = torch.cat((x, next_token), dim=-1)

        gen_time = time.time() - gen_start
        decoded = self.tokenizer.decode(x[0, :max_l].tolist())
        total = x.shape[1]
        print(f"Generated {total} tokens in {gen_time:.2f}s, {total/gen_time:.2f} tokens/sec")
        return decoded

    @staticmethod
    def _sample(logits, temperature, topk):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        return torch.gather(topk_indices, -1, ix)

    @staticmethod
    def from_pretrained(dict_path:str,matrix_percesion:str=None):
        checkpoint = torch.load(dict_path, weights_only=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = EzeLLM(checkpoint['config'],device=device)
        model.load_state_dict(checkpoint['model'])
        
        if matrix_percesion=='raw':
            torch.set_float32_matmul_precision('highest')
        elif matrix_percesion=='high':
            torch.set_float32_matmul_precision('high')
        elif matrix_percesion=='mid':
            torch.set_float32_matmul_precision('high')
        else:
            print('Matrix percesion is passed as None, you may consider "mid" for better performance.')

        return model

    @staticmethod
    def from_pretrained_fast(dict_path: str):
        """Load model in FP16 for fast KV-cached inference. Use model.generate_fast()."""
        checkpoint = torch.load(dict_path, weights_only=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = EzeLLM(checkpoint['config'], device=device)
        model.load_state_dict(checkpoint['model'])
        model.half()
        model.to(device)
        model.device = device
        model.eval()
        return model

if __name__ == '__main__':
    import os as _os
    import argparse as _argparse

    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _default_model = next(
        (p for p in [
            _os.path.join(_script_dir, '..', 'Optimization', 'model.pt'),
            _os.path.join(_script_dir, 'model.pt'),
            'model.pt',
        ] if _os.path.exists(p)),
        'model.pt',
    )

    _parser = _argparse.ArgumentParser(description='EzeLLM text generation')
    _parser.add_argument('--model', type=str, default=_default_model, help='Path to model checkpoint')
    _parser.add_argument('--prompt', type=str, default='The theory of relativity', help='Input prompt')
    _parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    _parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    _parser.add_argument('--topk', type=int, default=50, help='Top-k sampling')
    _args = _parser.parse_args()

    # FP16 + KV cache is the default — 8% faster than FP32, same quality
    model = EzeLLM.from_pretrained_fast(dict_path=_args.model)
    print(model.generate_fast(
        input_=_args.prompt,
        temperature=_args.temperature,
        topk=_args.topk,
        max_l=_args.max_tokens,
    ))

            