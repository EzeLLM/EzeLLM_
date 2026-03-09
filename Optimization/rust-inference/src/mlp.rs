use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

/// SwiGLU activation: silu(x) * sigmoid(x)
/// Note: EzeLLM uses a non-standard SwiGLU where both silu and sigmoid
/// are applied to the same input (not a gated variant).
fn swiglu(x: &Tensor) -> Result<Tensor> {
    let silu = candle_nn::ops::silu(x)?;
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    Ok((silu * sigmoid)?)
}

pub struct MLP {
    up: Linear,   // hidden_size → intermediate_size (1024 → 5120)
    down: Linear, // intermediate_size → hidden_size (5120 → 1024)
}

impl MLP {
    pub fn load(vb: VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let up = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up"))?;
        let down = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down"))?;
        Ok(Self { up, down })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.up.forward(x)?;
        let h = swiglu(&h)?;
        let h = self.down.forward(&h)?;
        Ok(h)
    }
}
