use anyhow::Result;
use candle_core::Tensor;

/// KV cache for a single transformer layer.
pub struct KVCache {
    k: Option<Tensor>, // (batch, n_kv_heads, cached_seq_len, head_dim)
    v: Option<Tensor>, // (batch, n_kv_heads, cached_seq_len, head_dim)
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Current number of cached tokens.
    #[allow(dead_code)]
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    /// Append new K/V tensors and return the full cached K/V.
    /// k_new, v_new: (batch, n_kv_heads, new_seq_len, head_dim)
    pub fn update(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (k_full, v_full) = match (&self.k, &self.v) {
            (Some(k_old), Some(v_old)) => {
                let k_full = Tensor::cat(&[k_old, k_new], 2)?;
                let v_full = Tensor::cat(&[v_old, v_new], 2)?;
                (k_full, v_full)
            }
            _ => (k_new.clone(), v_new.clone()),
        };

        self.k = Some(k_full.clone());
        self.v = Some(v_full.clone());

        Ok((k_full, v_full))
    }

    /// Reset the cache for a new conversation.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}
