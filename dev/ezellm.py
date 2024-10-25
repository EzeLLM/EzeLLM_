
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
import time
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint
from typing import List
import random

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
    def __init__(self,config):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.set_float32_matmul_precision('high')
        else:
            torch.set_float32_matmul_precision('medium')

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_size),
            #wpe = nn.Embedding(config.block_size,config.embed_size),
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
            tempreature: int = 0.7,
            tempreature_interval:int = 0.1,
            topk: int = 50,
            topp: float = 0.9,
            max_l: int = 200,
            num_return_seq: int = 1

    ) -> List:
        num_return_seq = 1
        max_l = 200
        tokens = self.tokenizer.encode(input_)
        tokens = torch.tensor(tokens,dtype=torch.long)
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
    
    @staticmethod
    def from_pretrained(dict_path:str):
        checkpoint = torch.load(dict_path,map_location=torch.device('cpu'))
        model = EzeLLM(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model
    

if __name__ == '__main__':
    path_to_pt = 'critical/model.pt'
    model = EzeLLM.from_pretrained(dict_path=path_to_pt)
    print(model.generate())




            