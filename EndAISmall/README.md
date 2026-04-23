## EndAI‑small.py (TinyLlama)

### Purpose

To quickly finetune a model and easily run and use it with the EndAI server!
This model is specifically designed by me!

### Usage

```bash
python EndAI-small.py
```

### What it does

1. Downloads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. Loads 3% of `HuggingFaceH4/ultrachat_200k` (train_sft)
3. Applies a system prompt and formats.
4. Tokenises with max length 512
5. Trains a LoRA adapter (`r=8`, targets `q_proj`, `v_proj`) for 2 epochs
6. Saves the merged model to `./endai-small-final/`

Manually convert to gguf using the llama.cpp build from the non-small version of this.
