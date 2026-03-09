use anyhow::Result;
use candle_core::{DType, Tensor, D};
use rand::Rng;

/// Sample a token from logits using top-k + temperature sampling.
pub fn sample_top_k(logits: &Tensor, temperature: f64, top_k: usize) -> Result<u32> {
    // logits: (vocab_size,) — 1D tensor

    // Apply temperature
    let logits = if temperature != 1.0 {
        (logits / temperature)?
    } else {
        logits.clone()
    };

    // Greedy decoding for very low temperature
    if temperature < 0.01 {
        let token = logits
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        return Ok(token);
    }

    // Get top-k values and indices
    let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    let vocab_size = logits_vec.len();
    let k = top_k.min(vocab_size);

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    // Softmax over top-k values
    let max_val = top_k_items[0].1;
    let exp_vals: Vec<f32> = top_k_items.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|v| v / sum).collect();

    // Multinomial sampling
    let idx = multinomial_sample(&probs);
    Ok(top_k_items[idx].0 as u32)
}

/// Sample from a discrete distribution using the inverse CDF method.
fn multinomial_sample(probs: &[f32]) -> usize {
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
