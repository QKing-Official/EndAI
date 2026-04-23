## EndAI.py (Mistral‑7B)

### Purpose

Fine‑tune a **7B** model on a rich mixture of instruction, reasoning, and chat data, then export directly to **GGUF** (Q4_K_M) for use with the EndAI Server.

### Usage

```bash
python EndAI.py
```

### What it does (step‑by‑step)

1. **Builds dataset** (if `data.jsonl` does not exist) by downloading and mixing:
   - OpenHermes‑2.5 (3k samples)
   - Dolly (3k)
   - GSM8K (2k)
   - UltraChat (2k, flattened)
   - C4 (2k, converted to weak QA)
   → Total ~12k samples.

2. **Loads** `mistralai/Mistral-7B-v0.1` in **4‑bit** (QLoRA) using `bitsandbytes`.

3. **Applies LoRA** (`r=16`, `lora_alpha=32`, targets `q_proj`, `v_proj`).

4. **Trains** with AdamW (lr=2e‑4) for one epoch over the dataset.  
   - Press `q` + `Enter` to stop gracefully.
   - Saves checkpoints every 500 steps to `lora_out/`.

5. **Merges** LoRA weights into the base model → saved to `merged_model/`.

6. **Tests** the merged model with a simple prompt.

7. **Exports to GGUF** (Q4_K_M) using `llama.cpp/convert-hf-to-gguf.py` → output `model.gguf`.

### Outputs

- `lora_out/` – intermediate LoRA adapters
- `merged_model/` – full merged HF model
- `model.gguf` – final quantised model ready for EndAI Server

### Notes

- Requires **`llama.cpp`** cloned in the same directory (or adjust path).
- The GGUF conversion may take 10‑20 minutes on a CPU, and let it run even when it freezes.
- For very large datasets, increase `max_length`.

## Using the Trained Models with EndAI Server

1. Place `model.gguf` (from `EndAI.py`) inside the `models/` folder of the EndAI Server.
2. Start the server: `python server.py`
3. In the web UI, load `model.gguf` – the server will auto‑detect the template (Mistral) and offload layers.

```bash
python llama.cpp/convert-hf-to-gguf.py ./endai-small-final --outfile tinyllama-endai.gguf --outtype q4_k_m
```

## Customisation

- **Change model** – edit `MODEL_NAME` at the top of each script.
- **Adjust dataset size** – modify the slice (e.g., `[:5000]`) in the download loops.
- **Change LoRA rank** – modify `r` and `lora_alpha` in `LoraConfig`.
- **Disable GGUF export** – comment out `export_gguf()` in `main()`.


## License & Acknowledgements

These scripts use:
- **TinyLlama** (Apache 2.0)
- **Mistral‑7B** (Apache 2.0)
- Datasets: OpenHermes (MIT‑licensed), Dolly (CC‑BY‑SA), GSM8K (MIT), UltraChat (CC‑BY‑NC‑4.0), C4 (ODC‑BY).

Fine‑tuned models inherit the base model’s license.