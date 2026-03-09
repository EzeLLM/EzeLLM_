use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct EzeLLMConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl EzeLLMConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn n_rep(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: EzeLLMConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }
}
