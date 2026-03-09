use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::cache::KVCache;
use crate::config::EzeLLMConfig;
use crate::rope::RotaryEmbedding;

/// Repeat KV heads to match query heads for GQA.
/// x: (batch, n_kv_heads, seq, head_dim) → (batch, n_heads, seq, head_dim)
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, hd))?
        .reshape((b, n_kv * n_rep, seq, hd))
        .map_err(Into::into)
}

pub struct Attention {
    qkv: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rep: usize,
}

impl Attention {
    pub fn load(vb: VarBuilder, config: &EzeLLMConfig) -> Result<Self> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();
        let hidden_size = config.hidden_size;

        let qkv_out = n_heads * head_dim + 2 * n_kv_heads * head_dim;
        let qkv = candle_nn::linear(hidden_size, qkv_out, vb.pp("qkv"))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            qkv,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rep: config.n_rep(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope: &RotaryEmbedding,
        cache: &mut KVCache,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;

        // QKV projection
        let qkv = self.qkv.forward(x)?;

        // Split into Q, K, V
        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape to (B, heads, T, head_dim)
        let q = q
            .reshape((b, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE with correct position offset
        let q = rope.apply(&q, start_pos)?;
        let k = rope.apply(&k, start_pos)?;

        // Update KV cache and get full K, V
        let (k_full, v_full) = cache.update(&k, &v)?;

        // Repeat KV heads for GQA
        let k_full = repeat_kv(&k_full, self.n_rep)?;
        let v_full = repeat_kv(&v_full, self.n_rep)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_full.transpose(2, 3)?)? * scale)?;

        // Apply causal mask during prefill (seq_len > 1)
        let attn_weights = if seq_len > 1 {
            let total_len = k_full.dim(2)?;
            let mask = create_causal_mask(seq_len, total_len, x.device())?;
            let mask = mask.to_dtype(attn_weights.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            // Single token decode — no mask needed
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let y = attn_weights.matmul(&v_full)?;

        // Concatenate heads and project
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, ()))?;
        let y = self.o_proj.forward(&y)?;

        Ok(y)
    }
}

/// Create a causal attention mask.
/// Returns a (1, 1, seq_len, total_len) tensor where masked positions are -inf.
fn create_causal_mask(seq_len: usize, total_len: usize, device: &Device) -> Result<Tensor> {
    // For prefill: query positions are [total_len - seq_len .. total_len]
    // Each query at position i can attend to positions [0..=i]
    let mut mask_data = vec![f32::NEG_INFINITY; seq_len * total_len];
    let offset = total_len - seq_len;
    for i in 0..seq_len {
        // Query i corresponds to absolute position (offset + i)
        // It can attend to positions 0..=(offset + i)
        for j in 0..=(offset + i) {
            mask_data[i * total_len + j] = 0.0;
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, total_len), device)?;
    Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
}
