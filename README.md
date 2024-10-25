# EzeLLM

EzeLLM is a custom large language model (LLM) architecture, designed for efficient text generation. It features a llama-like model structure and is optimized for experimentation, finetuning, and research. This repository provides the necessary code to load a pretrained model and run a demo to showcase its capabilities.

## NEWS
- An easy-to-use Python library is coming very soon.
- Quantized models will be available soon, improving mobility over the current fp32 model.
- A finetuning script is also on the way.

## Notes
- The library is tested on cpu only at the moment. 
## Motivation

EzeLLM was developed as a platform for optimization and finetuning research, rather than competing with state-of-the-art LLMs (e.g., sonnet 3.5). The challenge was to surpass GPT-3's performance (356M parameters) using a single NVIDIA 4090, which was achieved.

## Features

- Custom LLM architecture inspired by LLaMA.
- Efficient model loading and generation pipeline.
- Configuration through TOML files.
- Pretrained model download from a specified URL.
- Rotary, multi-head attention for enhanced performance.

## Source Tree

```
EzeLLM_
├── config
│   ├── config.toml
│   └── memory.toml
├── README.md
└── dev
    ├── __pycache__
    ├── pipeline.py
    ├── get_model.py
    └── ezellm.py
```

## Getting Started

### Prerequisites

- PyTorch
- Python packages: `requests`, `tqdm`, `toml`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/EzeLLM/EzeLLM.git
   cd EzeLLM_
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The configuration files (`config.toml` and `memory.toml`) contain settings for model URLs and memory information.

- **config.toml**: Contains the URL for downloading the pretrained model.
  
  ```toml
  model_url="https://huggingface.co/TerminatorPower/EzeLLM-base-text-fp32/resolve/main/model.pt"
  ```

- **memory.toml**: Stores model paths after downloading for future use.

### Usage

To run the model, execute the pipeline script from the terminal:

```bash
python dev/pipeline.py
```

Once the script is running, you can input text and the model will generate a continuation. 

**Note**: The model is trained on a 100 billion token dataset focused on English educational content. It is most appropriate to use the model with this in mind, as it performs optimally on educational or informational text inputs.


## File Structure

- **config/config.toml**: Configuration for the model URL.
- **dev/pipeline.py**: Contains the logic for setting up the pipeline, downloading, and loading the model.
- **dev/get_model.py**: Helper functions for downloading files with a progress bar.
- **dev/ezellm.py**: Definition of the EzeLLM model architecture, including attention and generation logic.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for new features, bug fixes, or improvements.
Please report bugs and problems.