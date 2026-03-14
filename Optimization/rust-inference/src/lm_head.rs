use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::EzeLLMConfig;

/// LM output head.
///
/// The Python model defines a 7-layer Sequential lm_head but iterates it with
/// `for head in self.lm_head: logits = head(x)` — each sub-layer receives the
/// original `x`, not the chained output. Only the last layer's result (the tied
/// embedding linear) is kept. We match this trained behavior.
pub struct LMHead {
    output: Linear,
}

impl LMHead {
    pub fn load(_vb: VarBuilder, _config: &EzeLLMConfig, wte_weight: &Tensor) -> Result<Self> {
        // lm_head.6: Linear(hidden_size, vocab_size) — weight tied to wte
        let output = Linear::new(wte_weight.clone(), None);
        Ok(Self { output })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.output.forward(x)?;
        Ok(x)
    }
}
