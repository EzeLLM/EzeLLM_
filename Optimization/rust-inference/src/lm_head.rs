use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::config::EzeLLMConfig;

/// SwiGLU activation (same as in mlp.rs)
fn swiglu(x: &Tensor) -> Result<Tensor> {
    let silu = candle_nn::ops::silu(x)?;
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    Ok((silu * sigmoid)?)
}

/// The 3-layer MLP output head:
///   Linear(1024, 1024) → RMSNorm → SwiGLU →
///   Linear(1024, 1024) → RMSNorm → SwiGLU →
///   Linear(1024, 50304)  [weight tied to embedding]
pub struct LMHead {
    linear1: Linear,
    norm1: RmsNorm,
    linear2: Linear,
    norm2: RmsNorm,
    output: Linear,
}

impl LMHead {
    pub fn load(vb: VarBuilder, config: &EzeLLMConfig, wte_weight: &Tensor) -> Result<Self> {
        let hs = config.hidden_size;
        let eps = config.rms_norm_eps;

        // lm_head.0: Linear(1024, 1024)
        let linear1 = candle_nn::linear(hs, hs, vb.pp("0"))?;

        // lm_head.1: RMSNorm(1024)
        let norm1 = candle_nn::rms_norm(hs, eps, vb.pp("1"))?;

        // lm_head.2: SwiGLU (no params)

        // lm_head.3: Linear(1024, 1024)
        let linear2 = candle_nn::linear(hs, hs, vb.pp("3"))?;

        // lm_head.4: RMSNorm(1024)
        let norm2 = candle_nn::rms_norm(hs, eps, vb.pp("4"))?;

        // lm_head.5: SwiGLU (no params)

        // lm_head.6: Linear(1024, 50304) — weight tied to wte
        // Build this linear manually using the embedding weight
        let output = Linear::new(wte_weight.clone(), None);

        Ok(Self {
            linear1,
            norm1,
            linear2,
            norm2,
            output,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.norm1.forward(&x)?;
        let x = swiglu(&x)?;
        let x = self.linear2.forward(&x)?;
        let x = self.norm2.forward(&x)?;
        let x = swiglu(&x)?;
        let x = self.output.forward(&x)?;
        Ok(x)
    }
}
