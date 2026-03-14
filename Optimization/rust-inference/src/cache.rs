use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Pre-allocated KV cache for a single transformer layer.
///
/// Allocates a fixed-size buffer upfront and writes into slices,
/// avoiding per-step tensor allocations during decode.
pub struct KVCache {
    k: Tensor, // (batch, n_kv_heads, max_seq_len, head_dim)
    v: Tensor, // (batch, n_kv_heads, max_seq_len, head_dim)
    seq_len: usize,
}

impl KVCache {
    /// Create a pre-allocated cache.
    pub fn new(
        batch_size: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (batch_size, n_kv_heads, max_seq_len, head_dim);
        let k = Tensor::zeros(shape, dtype, device)?;
        let v = Tensor::zeros(shape, dtype, device)?;
        Ok(Self { k, v, seq_len: 0 })
    }

    /// Current number of cached tokens.
    #[allow(dead_code)]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Write new K/V into the cache and return the full cached K/V
    /// (up to and including the new tokens).
    /// k_new, v_new: (batch, n_kv_heads, new_tokens, head_dim)
    pub fn update(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let new_tokens = k_new.dim(2)?;
        let end = self.seq_len + new_tokens;

        // Write new values into the pre-allocated buffer via slice_set
        // Ensure inputs are contiguous (they may not be after transpose/reshape)
        let k_new = k_new.contiguous()?;
        let v_new = v_new.contiguous()?;
        self.k.slice_set(&k_new, 2, self.seq_len)?;
        self.v.slice_set(&v_new, 2, self.seq_len)?;

        // Return a view of the filled portion
        let k_full = self.k.narrow(2, 0, end)?;
        let v_full = self.v.narrow(2, 0, end)?;

        self.seq_len = end;

        Ok((k_full, v_full))
    }

    /// Reset the cache for a new conversation.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}
