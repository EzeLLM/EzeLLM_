use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder};

use crate::attention::Attention;
use crate::cache::KVCache;
use crate::config::EzeLLMConfig;
use crate::lm_head::LMHead;
use crate::mlp::MLP;
use crate::rope::RotaryEmbedding;

/// A single transformer block: pre-norm attention + pre-norm MLP.
struct TransformerBlock {
    ln1: RmsNorm,
    attn: Attention,
    ln2: RmsNorm,
    mlp: MLP,
}

impl TransformerBlock {
    fn load(vb: VarBuilder, config: &EzeLLMConfig) -> Result<Self> {
        let eps = config.rms_norm_eps;
        let ln1 = candle_nn::rms_norm(config.hidden_size, eps, vb.pp("ln1"))?;
        let attn = Attention::load(vb.pp("attn"), config)?;
        let ln2 = candle_nn::rms_norm(config.hidden_size, eps, vb.pp("ln2"))?;
        let mlp = MLP::load(vb.pp("mlp"), config.hidden_size, config.intermediate_size)?;
        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope: &RotaryEmbedding,
        cache: &mut KVCache,
    ) -> Result<Tensor> {
        // Pre-norm attention with residual
        let residual = x;
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h, start_pos, rope, cache)?;
        let x = (residual + h)?;

        // Pre-norm MLP with residual
        let residual = &x;
        let h = self.ln2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let x = (residual + h)?;

        Ok(x)
    }
}

/// Full EzeLLM model.
pub struct EzeLLM {
    wte: Embedding,
    layers: Vec<TransformerBlock>,
    ln_f: RmsNorm,
    lm_head: LMHead,
    rope: RotaryEmbedding,
    pub config: EzeLLMConfig,
}

impl EzeLLM {
    pub fn load(vb: VarBuilder, config: &EzeLLMConfig) -> Result<Self> {
        let device = vb.device();
        let eps = config.rms_norm_eps;

        // Load embedding
        let wte = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("wte"))?;

        // Load transformer blocks
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(vb.pp(format!("layers.{i}")), config)?;
            layers.push(block);
        }

        // Final layer norm
        let ln_f = candle_nn::rms_norm(config.hidden_size, eps, vb.pp("ln_f"))?;

        // LM head — weight tied to embedding
        let wte_weight = wte.embeddings();
        let lm_head = LMHead::load(vb.pp("lm_head"), config, wte_weight)?;

        // Precompute RoPE tables (pre-cast to model dtype)
        let dtype = vb.dtype();
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            wte,
            layers,
            ln_f,
            lm_head,
            rope,
            config: config.clone(),
        })
    }

    /// Forward pass.
    /// token_ids: (batch, seq_len) u32
    /// start_pos: position offset for RoPE (= total tokens generated so far)
    /// caches: mutable reference to per-layer KV caches
    pub fn forward(
        &self,
        token_ids: &Tensor,
        start_pos: usize,
        caches: &mut [KVCache],
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(token_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, start_pos, &self.rope, &mut caches[i])?;
        }

        x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        Ok(logits)
    }

    /// Create pre-allocated KV caches for all layers.
    pub fn create_caches(&self, max_seq_len: usize, dtype: DType, device: &Device) -> Result<Vec<KVCache>> {
        (0..self.config.num_hidden_layers)
            .map(|_| {
                KVCache::new(
                    1, // batch_size
                    self.config.num_key_value_heads,
                    max_seq_len,
                    self.config.head_dim(),
                    dtype,
                    device,
                )
            })
            .collect()
    }
}
