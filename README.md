# EndAI Server

EndAI is a local AI chat server built with Flask that runs GGUF models using `llama.cpp`. It supports model loading, GPU/CPU acceleration, streaming chat, session memory, and automatic hardware detection.

Designed as a self‑hosted AI backend with a web interface and full API control.


## Features

- **Local GGUF inference** – run any GGUF‑format model via `llama.cpp`  
- **Automatic hardware detection** – CUDA (NVIDIA), ROCm (AMD), Metal (Apple), CPU fallback  
- **Dynamic GPU offloading** – configurable or auto‑estimated GPU layers  
- **Streaming chat** – real‑time token generation with Server‑Sent Events (SSE)  
- **Multi‑session storage** – JSON‑based chat history (easily migratable to SQLite)  
- **Model download** – built‑in downloader with progress tracking  
- **Prompt templates** – auto‑detection for ChatML, Llama 2/3, Mistral, Alpaca  
- **Context trimming** – prevents token overflow by trimming oldest messages  
- **Model hot‑swapping** – load/unload models without restarting the server  
- **Generation presets** – balanced, creative, precise, concise, storyteller  


## System Requirements

- **Python** 3.10 or higher  
- **RAM** – 8 GB minimum (16 GB+ recommended)  
- **Storage** – enough for your GGUF models (e.g., 4–20 GB per model)  
- **llama.cpp backend** – required (see installation)  

### Optional GPU Support

| GPU type   | Requirement                                |
|------------|--------------------------------------------|
| NVIDIA     | CUDA drivers + `nvidia-smi`               |
| AMD        | ROCm drivers + `rocm-smi`                 |
| Apple      | macOS with Metal (automatic)               |


## Installation


You can use this simple script to install it on Linux or macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/QKing-Official/EndAI/main/setup.sh | bash
```

For Windows:

```bash
irm https://raw.githubusercontent.com/QKing-Official/EndAI/main/setup.ps1 | iex
```

Alternatively, you can follow the manual steps below.


### 1. Clone the repository

```bash
git clone https://github.com/QKing-Official/EndAI.git
cd endai-server
```

### 2. Install Python dependencies

**Basic (CPU only):**

```bash
pip install flask llama-cpp-python
```

**Optional performance packages:**

```bash
pip install numpy psutil
```

### 3. Important: Install `llama-cpp-python` with the correct backend

EndAI **will not work** without proper `llama-cpp-python` support. Choose the variant matching your hardware:

#### CPU only (simplest)
```bash
pip install llama-cpp-python
```

#### NVIDIA GPU (CUDA 12.1)
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```
Verify with `nvidia-smi`.

#### AMD GPU (ROCm)
```bash
pip install llama-cpp-python
```
Requires ROCm drivers – check with `rocm-smi`.

#### Apple Silicon (Metal, macOS)
```bash
pip install llama-cpp-python
```
Metal support is used automatically on macOS.

### 4. Prepare the models directory

Create a `models/` folder and place your `.gguf` files there:

```bash
mkdir models
# Example: download a model
wget -O models/mistral-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## Running the Server

```bash
python server.py
```

Then open your browser at:  
**http://localhost:5000**

The web interface allows you to:
- Load/unload models
- Adjust generation parameters (temperature, top_p, max tokens)
- Select prompt templates
- Create/rename/delete chat sessions
- Download models directly from a URL

## Configuration & Internals

### Automatic Hardware Detection

The server automatically detects:

- **NVIDIA** – reads `nvidia-smi` → VRAM → estimates offload layers (~85 MB/layer, 80% of VRAM)  
- **AMD** – uses `rocm-smi` → offloads all layers (`-1`)  
- **Apple Metal** – checks `system_profiler` → offloads all layers  
- **CPU** – fallback with 0 GPU layers

### Prompt Templates & Auto‑detection

Based on the model filename, a template is suggested:

| Contains                    | Template  |
|-----------------------------|-----------|
| `llama-3`, `llama3`, `meta-llama-3` | llama3 |
| `llama-2`, `llama2`          | llama2    |
| `mistral`, `mixtral`         | mistral   |
| `alpaca`                     | alpaca    |
| otherwise                    | chatml    |

### Context Trimming

To prevent exceeding the context window, the server trims the conversation history, keeping system messages and the most recent messages that fit into `max_ctx - reserved_tokens`.

### Session Storage

Sessions are stored in `sessions.json` (JSON format). The code contains a **TODO** to migrate to SQLite – contributions welcome.


## Troubleshooting

### `llama-cpp-python` not found / model fails to load

- Make sure you installed the correct variant (CUDA, ROCm, Metal, or CPU).
- On Linux, verify GPU drivers are working (`nvidia-smi` / `rocm-smi`).
- On macOS, ensure you have a Metal‑compatible device.

### Model loads but no GPU offloading

- Check `/api/hardware` – if `n_gpu_layers_auto` is 0, the server fell back to CPU.  
- Manually specify `n_gpu_layers` in the load request.

### Download fails / hangs

- The download runs in a background thread. Check `/api/download/progress` for errors.
- Ensure the URL is directly downloadable and the server has write permission to `models/`.

### Streaming stops unexpectedly

- Some models may hit the token limit and attempt to continue (max 10 rounds).  
- Adjust `max_tokens` or use a larger context window.


## License

This project is provided as‑is. The underlying `llama.cpp` and `llama-cpp-python` have their own licenses (MIT). Please comply with model‑specific licenses when using GGUF files.

## Contributing

Issues and pull requests are welcome. Priorities include:

- Migrating session storage to SQLite  
- Adding API authentication (API keys)  
- Supporting more prompt templates  
- Improving token counting accuracy


## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) – GGUF inference engine  
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) – Python bindings  
- Flask – lightweight web framework  

**EndAI Server – run your own AI chat, completely offline!**

---

# EndAI Training Scripts

These scripts fine‑tune open‑source LLMs on conversational / instruction data and export them for use with the **EndAI Server** (GGUF format via `llama.cpp`).

Two versions are provided:

- **`EndAI‑small.py`** – lightweight fine‑tuning of `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on 3% of UltraChat.  
  *Runs on modest hardware (e.g., 8 GB RAM, 4 GB VRAM).*

- **`EndAI.py`** – full‑scale LoRA fine‑tuning of `mistralai/Mistral-7B-v0.1` on a mixture of datasets (OpenHermes, Dolly, GSM8K, UltraChat, C4). Includes automatic dataset building, 4‑bit QLoRA training, and direct GGUF export.

You dont have to do all steps if you want to just download and use the model.
I uploaded them to huggingface.

EndAI: https://huggingface.co/QKing-Official/EndAI
EndAI-Small: https://huggingface.co/QKing-Official/EndAI-Small

Download the desired GGUF file from them and put them in the models directory.
Preferably dont use the f16 variants. Those are bigger, slower and dont have advantages for running them.
Load the model in the webui and go chat with it!
Be careful, the smaller model can be scary, it won't do anything and that is just a mistake in my training process.
Enjoy the model!

If you want to build the models yourself and you got powerful hardware, follow the steps below.
Make sure for the full model you atleast have 4070 GPU.

## Requirements (Both Scripts)

- Python 3.10+
- CUDA GPU recommended (NVIDIA) – 8+ GB VRAM for Mistral, 4+ GB for TinyLlama
- At least 16 GB system RAM for Mistral (8 GB for TinyLlama)
- `llama.cpp` repository cloned (for GGUF conversion in `EndAI.py`)

### Install common dependencies

```bash
# CPU only or Apple Metal
pip install torch transformers datasets peft accelerate bitsandbytes rich

# CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate bitsandbytes rich

## AMD ROCm (Linux only)
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
pip install transformers datasets peft accelerate bitsandbytes rich
```

For **`EndAI.py`** you also need:

```bash
pip install huggingface_hub sentencepiece protobuf
```

And **`llama.cpp`** (for GGUF conversion):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release -j $(nproc)   # For Linux/macOS
# On Windows: cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

# The script expects convert-hf-to-gguf.py inside llama.cpp/
# That is why it needs llama.cpp for it.
```

See more about how to use it in the EndAI and EndAISmall directory

Respect the licenses of the datasets and programs used in the files itself, they vary per script please


Cya onces again! I'm going for lockinweek 4!
Good luck with it and feel free to contact me when you have issues!
qking@freemchosting.com or make a github issue please.
I will try to answer it as fast and clear as possible.