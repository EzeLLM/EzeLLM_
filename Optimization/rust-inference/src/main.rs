mod attention;
mod cache;
mod config;
mod lm_head;
mod mlp;
mod model;
mod rope;
mod sampling;

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;

use crate::config::EzeLLMConfig;
use crate::model::EzeLLM;
use crate::sampling::sample_top_k;

// Embed model files directly into the binary
const MODEL_BYTES: &[u8] = include_bytes!("../../exported/model.safetensors");
const CONFIG_BYTES: &[u8] = include_bytes!("../../exported/config.json");
const TOKENIZER_BYTES: &[u8] = include_bytes!("../../exported/tokenizer.json");

#[derive(Parser)]
#[command(name = "ezellm", about = "EzeLLM inference engine (model embedded)")]
struct Args {
    /// Input prompt
    #[arg(short, long, default_value = "The theory of relativity")]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-k for sampling
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        eprintln!("Using CPU");
        Device::Cpu
    } else if cfg!(feature = "cuda") {
        match Device::new_cuda(0) {
            Ok(d) => { eprintln!("Using CUDA"); d }
            Err(e) => { eprintln!("CUDA init failed ({}), falling back to CPU", e); Device::Cpu }
        }
    } else if cfg!(feature = "metal") {
        match Device::new_metal(0) {
            Ok(d) => { eprintln!("Using Metal"); d }
            Err(e) => { eprintln!("Metal init failed ({}), falling back to CPU", e); Device::Cpu }
        }
    } else {
        eprintln!("Using CPU");
        Device::Cpu
    };

    let dtype = if device.is_cuda() || device.is_metal() {
        DType::F16
    } else {
        DType::F32
    };

    // Load config from embedded bytes
    let config = EzeLLMConfig::from_bytes(CONFIG_BYTES)?;
    eprintln!("Model config: {:?}", config);

    // Load model weights from embedded safetensors
    eprintln!("Loading model...");
    let load_start = Instant::now();
    let vb = VarBuilder::from_slice_safetensors(MODEL_BYTES, dtype, &device)?;
    let model = EzeLLM::load(vb, &config)?;
    eprintln!("Model loaded in {:.2?}", load_start.elapsed());

    // Load tokenizer from embedded bytes
    let tokenizer = tokenizers::Tokenizer::from_bytes(TOKENIZER_BYTES)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // EOT token ID for GPT-2
    let eot_id: u32 = 50256;

    // Tokenize prompt
    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_ids.len();
    let mut tokens = prompt_ids.clone();

    // Initialize pre-allocated KV caches
    let max_seq_len = (prompt_len + args.max_tokens).min(config.max_position_embeddings);
    let mut caches = model.create_caches(max_seq_len, dtype, &device)?;

    // Print the prompt
    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    // === Prefill phase ===
    let prefill_start = Instant::now();
    let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0, &mut caches)?;

    // Sample first new token from last position
    let seq_len = logits.dim(1)?;
    let last_logits = logits
        .narrow(1, seq_len - 1, 1)?
        .squeeze(0)?
        .squeeze(0)?;
    let next_token = sample_top_k(&last_logits, args.temperature, args.top_k)?;
    tokens.push(next_token);

    let prefill_time = prefill_start.elapsed();

    // Print first generated token
    if next_token != eot_id {
        if let Ok(decoded) = tokenizer.decode(&[next_token], false) {
            print!("{}", decoded);
            std::io::stdout().flush()?;
        }
    }

    // === Decode phase ===
    let decode_start = Instant::now();
    let mut generated = 1usize; // already generated one token

    if next_token != eot_id {
        for _ in 1..args.max_tokens {
            let pos = tokens.len() - 1;
            if pos >= config.max_position_embeddings - 1 {
                break;
            }
            let input =
                Tensor::new(&[*tokens.last().unwrap()], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos, &mut caches)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;

            let next_token = sample_top_k(&logits, args.temperature, args.top_k)?;

            if next_token == eot_id {
                break;
            }

            tokens.push(next_token);
            generated += 1;

            // Stream output
            if let Ok(decoded) = tokenizer.decode(&[next_token], false) {
                print!("{}", decoded);
                std::io::stdout().flush()?;
            }
        }
    }

    let decode_time = decode_start.elapsed();

    println!();
    println!();
    println!("--- Stats ---");
    println!(
        "Prefill: {:.2?} ({} tokens, {:.1} tok/s)",
        prefill_time,
        prompt_len,
        prompt_len as f64 / prefill_time.as_secs_f64()
    );
    println!(
        "Decode:  {:.2?} ({} tokens, {:.1} tok/s)",
        decode_time,
        generated,
        generated as f64 / decode_time.as_secs_f64()
    );
    println!(
        "Total:   {:.2?} ({} tokens generated)",
        prefill_time + decode_time,
        generated
    );

    Ok(())
}
