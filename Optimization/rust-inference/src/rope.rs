use anyhow::Result;
use candle_core::{Device, Tensor};

/// Precomputed sin/cos tables for Rotary Position Embeddings.
pub struct RotaryEmbedding {
    cos: Tensor, // (max_seq_len, head_dim)
    sin: Tensor, // (max_seq_len, head_dim)
}

impl RotaryEmbedding {
    /// Precompute the frequency table for the full max sequence length.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute inv_freq: 1.0 / (theta ^ (2i / head_dim)) for i in 0..half_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // (half_dim,)

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?; // (max_seq_len,)

        // Outer product: (max_seq_len, half_dim)
        let freqs = positions.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // Duplicate for full head_dim: [freq, freq] so cos/sin have shape (max_seq_len, head_dim)
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    /// Apply rotary embedding to a tensor x of shape (batch, heads, seq_len, head_dim)
    /// starting at position `start_pos`.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let head_dim = x.dim(3)?;
        let half_dim = head_dim / 2;

        // Slice cos/sin for the relevant positions: [start_pos .. start_pos + seq_len]
        let cos = self.cos.narrow(0, start_pos, seq_len)?; // (seq_len, head_dim)
        let sin = self.sin.narrow(0, start_pos, seq_len)?; // (seq_len, head_dim)

        // Reshape for broadcasting: (1, 1, seq_len, head_dim)
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Split x into first half and second half
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // rotate_half: [-x2, x1]
        let cos1 = cos.narrow(3, 0, half_dim)?;
        let sin1 = sin.narrow(3, 0, half_dim)?;

        // x_rotated[..., :half] = x[..., :half] * cos - x[..., half:] * sin
        // x_rotated[..., half:] = x[..., half:] * cos + x[..., :half] * sin
        // Note: cos and sin are duplicated so cos[:half] == cos[half:], sin[:half] == sin[half:]
        let out1 = (x1.broadcast_mul(&cos1)? - x2.broadcast_mul(&sin1)?)?;
        let out2 = (x2.broadcast_mul(&cos1)? + x1.broadcast_mul(&sin1)?)?;

        let result = Tensor::cat(&[&out1, &out2], 3)?;
        Ok(result)
    }
}
