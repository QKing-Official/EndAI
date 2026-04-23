import os
import json
import time
import threading
import subprocess
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from rich.live import Live
from rich.table import Table
from rich.console import Console

console = Console()

# The configuration

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

DATA_FILE = "data.jsonl"
OUT_DIR = "lora_out"
MERGED_DIR = "merged_model"
GGUF_FILE = "model.gguf"

STOP = False

# Stop the key listener

def listen():
    global STOP
    while True:
        cmd = input()
        if cmd.strip().lower() == "q":
            STOP = True
            break

# Build the dataset
# Also download all datasets from the internet
# I cant make my own dataset sadly so you have to bear with this
def build_dataset():
    import json
    from datasets import load_dataset

    data = []

    def add(user, assistant):
        if not user or not assistant:
            return
        data.append({
            "text": f"User: {user.strip()}\nAssistant: {assistant.strip()}"
        })

    print("Downloading OpenHermes (high quality instructions)...")

    hermes = load_dataset(
        "teknium/OpenHermes-2.5",
        split="train[:3000]"
    )

    for r in hermes:
        add(r["instruction"], r["response"])

    print("Downloading Dolly (light reasoning + QA)...")

    dolly = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train[:3000]"
    )

    for r in dolly:
        user = r["instruction"]
        if r.get("context"):
            user += "\nContext: " + r["context"]
        add(user, r["response"])

    print("Downloading GSM8K (reasoning boost)...")

    gsm = load_dataset("gsm8k", "main", split="train[:2000]")

    for r in gsm:
        add(r["question"], r["answer"])

    print("Downloading UltraChat (chat behavior)...")

    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]")

    for r in chat:
        # UltraChat already multi-turn so flatten it out to be functional
        conv = r["messages"]
        for i in range(len(conv) - 1):
            if conv[i]["role"] == "user" and conv[i+1]["role"] == "assistant":
                add(conv[i]["content"], conv[i+1]["content"])

    print("Downloading C4 (knowledge base)...")

    c4 = load_dataset("allenai/c4", "en", split="train[:2000]")

    for r in c4:
        text = r["text"]
        # convert into weak QA style for fun
        # Nah, no fun at 1 am so this was needed
        if len(text) > 200:
            add(
                "Summarize and explain this text:\n" + text[:1000],
                text[:800]
            )

    print(f"Total samples: {len(data)}")

    with open("data.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# LOAD The model to actually be able to finetune it. You dumb thing
# The model can be a bit dumb tho. It repeats the question when it doesnt know what to answer.
# I had to improve this but no time

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Why tf did I import it here??
import torch # That is a question for never

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    from transformers import BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    return model, tok

# Setup and apply the lora and its configuration

def apply_lora(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)

# Load the actual data and tokenise it
# I am talking about the dataset indeed!

# Hello mister sleep!
# Wait, you dont exist
# I know I dont, so go lock tf in!

def load_data(tokenizer):
    ds = load_dataset("json", data_files=DATA_FILE)["train"]

    def tok(x):
        return tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    return ds.map(tok)

# This is the actual training loop to train the model on the dataset
def train(model, dataset):
    global STOP

    optimizer = AdamW(model.parameters(), lr=2e-4)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()

    start = time.time()

    def ui(step, loss):
        elapsed = time.time() - start
        speed = step / elapsed if elapsed else 0
        eta = (len(dataset) - step) / speed if speed else 0

        table = Table(title="Training")

        table.add_column("Step")
        table.add_column("Loss")
        table.add_column("Speed")
        table.add_column("ETA")

        table.add_row(
            str(step),
            f"{loss:.4f}",
            f"{speed:.2f}/s",
            f"{int(eta)}s"
        )
        return table

    model.train()

    with Live(ui(0, 0.0), refresh_per_second=2) as live:

        for i, item in enumerate(dataset):

            if STOP:
                console.print("[red]Stopping... saving checkpoint[/red]")
                break

            input_ids = torch.tensor([item["input_ids"]]).to(model.device)
            labels = input_ids.clone()

            out = model(input_ids, labels=labels)
            loss = out.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                live.update(ui(i, loss.item()))

            if i % 500 == 0 and i > 0:
                model.save_pretrained(OUT_DIR)

    model.save_pretrained(OUT_DIR)

## Merge the model

def merge():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base, OUT_DIR)
    model = model.merge_and_unload()

    model.save_pretrained(MERGED_DIR)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.save_pretrained(MERGED_DIR)

# This part has a chance to export the model to gguf, but might fail since the build location may differ

def export_gguf():
    console.print("[cyan]Converting to GGUF...[/cyan]")

    subprocess.run([
        "python",
        "llama.cpp/convert-hf-to-gguf.py",
        MERGED_DIR,
        "--outfile",
        GGUF_FILE,
        "--outtype",
        "q4_k_m"
    ])

# A quick test right from the transformers

def test_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print("\n[green]Testing model...[/green]\n")

    tok = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )

    prompt = "User: Hello, who are you?\nAssistant:"

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True
    )

    print(tok.decode(out[0], skip_special_tokens=True))

# THE MAIN FUNCTION! OH HOLY FUNCTION!

def main():

    if not os.path.exists(DATA_FILE):
        build_dataset()

    console.print("[cyan]Loading model...[/cyan]")
    model, tok = load_model()

    model = apply_lora(model)

    dataset = load_data(tok)

    console.print("[green]Training (press q + enter to stop)[/green]")
    train(model, dataset)

    console.print("[cyan]Merging...[/cyan]")
    merge()

    console.print("[cyan]Testing...[/cyan]")
    test_model()

    console.print("[cyan]Exporting GGUF...[/cyan]")
    export_gguf()

    console.print("\n[bold green]DONE → model.gguf ready[/bold green]")

if __name__ == "__main__":
    main()