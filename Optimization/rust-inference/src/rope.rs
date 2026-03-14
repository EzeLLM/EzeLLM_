use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Precomputed sin/cos tables for Rotary Position Embeddings.
/// Uses the **interleaved** convention to match rotary_embedding_torch:
///   rotate_half pairs adjacent elements (x0,x1), (x2,x3), ...
///   NOT split-half (x0,x32), (x1,x33), ...
pub struct RotaryEmbedding {
    cos: Tensor, // (max_seq_len, head_dim) in model dtype
    sin: Tensor, // (max_seq_len, head_dim) in model dtype
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // inv_freq for each pair: shape (half_dim,)
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;

        // freqs: (max_seq_len, half_dim)
        let freqs = positions.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // Interleave to (max_seq_len, head_dim): [f0,f0, f1,f1, f2,f2, ...]
        // This matches rotary_embedding_torch's `repeat(freqs, '... n -> ... (n r)', r=2)`
        let freqs = freqs
            .unsqueeze(2)?                              // (seq, half_dim, 1)
            .expand((max_seq_len, half_dim, 2))?         // (seq, half_dim, 2)
            .reshape((max_seq_len, head_dim))?;          // (seq, head_dim)

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Apply rotary embedding to x: (batch, heads, seq_len, head_dim)
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;

        // Slice for relevant positions, already correct dtype
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        // Reshape for broadcasting: (1, 1, seq_len, head_dim)
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // rotate_half (interleaved): [x0,x1,x2,x3,...] → [-x1,x0,-x3,x2,...]
        let x_rotated = rotate_half_interleaved(x)?;

        // x * cos + rotate_half(x) * sin
        let result = (x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
        Ok(result)
    }
}

/// Interleaved rotate_half matching rotary_embedding_torch:
///   input:  [..., x0, x1, x2, x3, ...]
///   output: [..., -x1, x0, -x3, x2, ...]
fn rotate_half_interleaved(x: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    let half_dim = head_dim / 2;

    // Reshape last dim: (..., head_dim) → (..., half_dim, 2)
    let shape = x.shape().dims().to_vec();
    let new_shape: Vec<usize> = shape[..3].iter().copied()
        .chain([half_dim, 2])
        .collect();
    let x = x.reshape(&new_shape[..])?;

    // Split the pairs
    let x1 = x.narrow(4, 0, 1)?; // even indices
    let x2 = x.narrow(4, 1, 1)?; // odd indices

    // Stack as (-x2, x1) and flatten back
    let neg_x2 = x2.neg()?;
    let rotated = Tensor::cat(&[&neg_x2, &x1], 4)?; // (..., half_dim, 2)
    let out_shape: Vec<usize> = shape.to_vec();
    rotated.reshape(&out_shape[..]).map_err(Into::into)
}
