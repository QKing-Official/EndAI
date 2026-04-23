#!/usr/bin/env python3

import os
import sys
import json
import time
import uuid
import subprocess
import threading
from pathlib import Path
from urllib import request as urlrequest
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SESSIONS_FILE = Path("sessions.json")

app = Flask(__name__)

# States of all various shit. Too much too comment all here in this little line of misery

_model        = None
_loaded_name  = None
_load_status  = "idle"   # idle | loading | ready | error ## Used for clarity
_load_error   = None
_model_lock   = threading.Lock()
_downloads    = {}        # filename → {downloaded, total, speed, done, error, eta}
_sessions     = {}        # session_id → {name, messages, created_at, updated_at}

# Hardware detection, detect if using gpu or cpu
# Also detect what brand and technology should be used to run the models
# We support cpu, AMD, NVIDIA, Apple
# It will also automatically set the layers needed and that will work the best with your gpu
# This will ensure smooth user experience

def detect_hw():
    info = {
        "cuda":     False,
        "rocm":     False,
        "metal":    False,
        "cpu":      True,
        "gpu_name": None,
        "vram_mb":  None,
        "n_gpu_layers_auto": 0,
    }

    # NVIDIA CUDA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            vram  = int(parts[1].strip()) if len(parts) > 1 else None
            info.update({
                "cuda":     True,
                "gpu_name": parts[0].strip(),
                "vram_mb":  vram,
                # Heuristic: push as many layers as VRAM allows.
                # ~100 MB per layer for 7B Q4 models; conservative at 80 %.
                "n_gpu_layers_auto": _estimate_gpu_layers(vram),
            })
            return info
    except Exception:
        pass

    # AMD ROCm
    try:
        r = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "GPU" in r.stdout:
            # ROCm doesn't expose VRAM as easily; fall back to -1 (offload all layers)
            gpu_line = [l for l in r.stdout.splitlines() if "GPU" in l]
            name = gpu_line[0].split(",")[-1].strip() if gpu_line else "AMD GPU"
            info.update({
                "rocm":     True,
                "gpu_name": name,
                "n_gpu_layers_auto": -1,   # -1 = offload everything
            })
            return info
    except Exception:
        pass

    # Apple Metal (macOS) for the apple fanboys
    # If you use this im going to cry tbh
    # I understand cpu more than this
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and "Metal" in r.stdout:
                # Extract first GPU name line to just throw it back to the user!
                for line in r.stdout.splitlines():
                    if "Chipset Model" in line or "Chip" in line:
                        info.update({
                            "metal":    True,
                            "gpu_name": line.split(":")[-1].strip(),
                            "n_gpu_layers_auto": -1,
                        })
                        break
        except Exception:
            pass

    return info

# This is the gpu layer estimation
# This is used to make sure the running is stable and will work smoothly
# This is just state-of-the-art in my opinion
# Yes I enjoy commenting
# Glad you asked!
# I really like that
def _estimate_gpu_layers(vram_mb: int | None) -> int:
    # Just some raw estimation of gpu layers
    # Not accurate at all!
    if vram_mb is None:
        return 0
    if vram_mb >= 10_000:
        return -1          # plenty of VRAM → offload all layers
    usable   = int(vram_mb * 0.80)
    per_layer = 85          # MB, conservative estimate for Q4 7-B
    return max(0, usable // per_layer)


HW = detect_hw()

# Model presets for easy of use
# This greatly improves the user experience I think

PRESETS = {
    "balanced": {
        "label": "Balanced",
        "description": "Good for most tasks",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
    },
    "creative": {
        "label": "Creative",
        "description": "More varied, imaginative output",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 768,
    },
    "precise": {
        "label": "Precise",
        "description": "Focused, deterministic answers",
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 512,
    },
    "concise": {
        "label": "Concise",
        "description": "Short, direct replies",
        "temperature": 0.5,
        "top_p": 0.85,
        "max_tokens": 256,
    },
    # Insanely high token count so it can actually do shit!
    # Before it couldnt do anything properly
    "storyteller": {
        "label": "Storyteller",
        "description": "Long, expressive creative writing",
        "temperature": 1.1,
        "top_p": 0.98,
        "max_tokens": 2048,
    },
}

# Prompt templates cuz its needed
# Each model has its own formats
# That is why we include a couple of them here
# Project templates, those are to make sure each model gets its reponse in the format they need


# The template for chatml like models
def _chatml(messages):
    out = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out

# The template for models tha are llama2-lik
def _llama2(messages):
    sys_prompt = ""
    chat = []

    for m in messages:
        if m.get("role") == "system":
            sys_prompt = m.get("content", "")
        else:
            chat.append(m)

    out = "<s>"

    for i, m in enumerate(chat):
        role = m.get("role")
        content = m.get("content", "")

        if role == "user":
            if i == 0 and sys_prompt:
                out += f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{content} [/INST]"
            else:
                out += f"[INST] {content} [/INST]"

        elif role == "assistant":
            # IMPORTANT: keep </s> ONLY inside template, not as stop token
            out += f" {content} </s><s>"

    return out

# The template for the common llama3 model series and other llms that are like it
def _llama3(messages):
    out = "<|begin_of_text|>"

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out

# The template for mistral ai-like models and ofc mistral based models itself
def _mistral(messages):
    out = ""

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        if role == "system":
            # Mistral doesn't really support system role well so I dont care at this point
            out += f"{content}\n\n"

        elif role == "user":
            out += f"[INST] {content} [/INST]"

        elif role == "assistant":
            out += f"{content}</s>"

    return out

# alpaca-like models. Havent seen much of them but I added it to be sure people can run everything they want
def _alpaca(messages):
    out = ""

    sys_prompt = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"),
        ""
    )

    if sys_prompt:
        out += f"{sys_prompt}\n\n"

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        if role == "user":
            out += f"### Instruction:\n{content}\n\n"
        elif role == "assistant":
            out += f"### Response:\n{content}\n\n"

    out += "### Response:\n"
    return out


# The template registry. Here all all the templates stored and mapped to their information and function.
# Without this the functions are useless since they get called through the readers of this list

TEMPLATES = {
    "chatml": {
        "label": "ChatML",
        "fn": _chatml,
        "stop": ["<|im_end|>"],
    },

    "llama2": {
        "label": "Llama 2",
        "fn": _llama2,
        # DO NOT include </s> here since it causes the model to fuck itself (early stop)
        "stop": ["[INST]"],
    },

    "llama3": {
        "label": "Llama 3",
        "fn": _llama3,
        "stop": ["<|eot_id|>"],
    },

    "mistral": {
        "label": "Mistral",
        "fn": _mistral,
        # Be safe and dont stop a lot
        "stop": ["[INST]"],
    },

    "alpaca": {
        "label": "Alpaca",
        "fn": _alpaca,
        "stop": ["### Instruction:"],
    },
}


# Guesss the template! This is based on the filename of the model, lol. Its genius

# This will do name checking in filenames to find if it has this chars inside to auto figure it out
# Somehow this doesnt work good
# Not somehow, but yeah
def guess_template(filename: str) -> str:
    fn = filename.lower()

    if any(x in fn for x in ["llama-3", "llama3", "meta-llama-3"]):
        return "llama3"

    if any(x in fn for x in ["llama-2", "llama2"]):
        return "llama2"

    if any(x in fn for x in ["mistral", "mixtral"]):
        return "mistral"

    if "alpaca" in fn:
        return "alpaca"

    return "chatml"


# The token counter that estimates the tokens that a message will cost


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _messages_token_count(messages: list) -> int:
    return sum(_rough_token_count(m.get("content", "")) + 4 for m in messages)


# Context trimming, tries to get previous messages and answers but trims it whenit exceeds the context window
# We need to make sure it fits inside the context window at all times

def trim_messages(messages: list, max_context: int = 4096, max_tokens_reserved: int = 512) -> list:
    budget = max_context - max_tokens_reserved

    tokenizer_fn = _rough_token_count

    if _model is not None:
        try:
            def tokenizer_fn(text):
                return len(_model.tokenize(text.encode()))
        except Exception:
            pass

    def msg_tokens(m):
        return tokenizer_fn(m.get("content", "")) + 4

    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]

    # Always keep system messages
    total_tokens = sum(msg_tokens(m) for m in system_msgs)

    kept = []
    remaining = budget - total_tokens

    # Walk backwards (most recent first)
    # Imagine the micheal jackson sfx
    for m in reversed(conv_msgs):
        t = msg_tokens(m)

        if t <= remaining:
            kept.insert(0, m)
            remaining -= t
        else:
            break

    return system_msgs + kept

# Import llamacpp safely
# We use llamacpp for actually running the ai models

def _import_llama_cpp():
    from llama_cpp import Llama
    return Llama

# Load and unload the model since we need to get it to memory to offload it and use it in the chat!

def load_model(filename: str, max_seq_len: int = 4096, n_gpu_layers: int | None = None):
    global _model, _loaded_name, _load_status, _load_error

    path = MODELS_DIR / filename
    if not path.exists():
        _load_status = "error"
        _load_error = f"Model file not found: {filename}"
        return

    _load_status = "loading"
    _load_error  = None

    # Resolve how many layers to offload (prefer most) How more offloaded to gpu how more gpu usage, how more speed increases and how much the other resources are actually used.
    if n_gpu_layers is None:
        n_gpu_layers = HW.get("n_gpu_layers_auto", 0)

    use_gpu = n_gpu_layers != 0

    try:
        Llama = _import_llama_cpp()
        with _model_lock:
            if _model is not None:
                del _model
                _model = None

            cpu_count = os.cpu_count() or 4
            kwargs = dict(
                model_path      = str(path),
                n_ctx           = max_seq_len,
                n_gpu_layers    = n_gpu_layers,
                n_threads       = max(1, cpu_count - 1) if not use_gpu else 4,
                n_threads_batch = max(1, cpu_count - 1) if not use_gpu else 4,
                verbose         = True,
            )

            _model       = Llama(**kwargs)
            _loaded_name = filename
            _load_status = "ready"

    except Exception as e:
        _load_status = "error"
        _load_error  = str(e)
        _loaded_name = None


def unload_model():
    global _model, _loaded_name, _load_status, _load_error
    with _model_lock:
        if _model is not None:
            del _model
            _model = None
        _loaded_name = None
        _load_status = "idle"
        _load_error  = None

# Sessions with persistence. Saves them in sessions.json. Please see this later since I cant live with json data storage
# TODO: migrate to SQLite
# Sadly I have no time to migrate to sqlite... this is sad, but we really shouldnt get it broken before the ship
# Its 00.00 and I want to sleep, but I need 2 hours more.... That is sooo sad
# SLEEP SLEEP SLEEP
# I need to sleep but I cant :sob:

def _save_sessions():
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(_sessions, f, indent=2)
    except Exception:
        pass

def _load_sessions():
    global _sessions
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE) as f:
                _sessions = json.load(f)
        except Exception:
            _sessions = {}

_load_sessions()

def new_session(name: str = None) -> dict:
    sid = str(uuid.uuid4())[:8]
    now = time.time()
    session = {
        "id":         sid,
        "name":       name or f"Chat {len(_sessions) + 1}",
        "messages":   [],
        "created_at": now,
        "updated_at": now,
    }
    _sessions[sid] = session
    _save_sessions()
    return session

def get_session(sid: str) -> dict | None:
    return _sessions.get(sid)

def update_session(sid: str, messages: list, name: str = None):
    if sid not in _sessions:
        return
    _sessions[sid]["messages"]   = messages
    _sessions[sid]["updated_at"] = time.time()
    if name:
        _sessions[sid]["name"] = name
    _save_sessions()

def delete_session(sid: str):
    _sessions.pop(sid, None)
    _save_sessions()

# The download working in a seperate thread
# this ensures that the speed is still high and the webserver will survive it since it has its own thread (almost)

def _download_worker(url: str, filename: str):
    dest = MODELS_DIR / filename
    key  = filename
    _downloads[key] = {"downloaded": 0, "total": 0, "speed": 0,
                       "eta": None, "done": False, "error": None}
    try:
        req = urlrequest.Request(url, headers={"User-Agent": "gguf-chat/1.0"})
        with urlrequest.urlopen(req, timeout=30) as resp:
            total      = int(resp.headers.get("Content-Length") or 0)
            _downloads[key]["total"] = total
            downloaded = 0
            t0 = time.time()
            with open(dest, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - t0 or 1e-9
                    speed   = downloaded / elapsed
                    eta     = int((total - downloaded) / speed) if speed and total else None
                    _downloads[key].update({"downloaded": downloaded,
                                            "speed": int(speed), "eta": eta})
        _downloads[key]["done"] = True
    except Exception as e:
        _downloads[key]["error"] = str(e)
        if dest.exists():
            dest.unlink()

# The main routes used to communicate with the frontend
# This is the worst thing ever since I did not add a API key, but better than nothing

# This is the main route, it will serve the index.html to the user. The actual chat is there
# This is the only frontend route, all the other routes are backend or internal only
@app.route("/")
def index():
    return render_template("index.html")

# This route displays your hardware information
@app.route("/api/hardware")
def api_hardware():
    return jsonify(HW)

# This route lists all existing presets
@app.route("/api/presets")
def api_presets():
    return jsonify(PRESETS)

# This route lists all templates that exist.
@app.route("/api/templates")
def api_templates():
    return jsonify({k: {"label": v["label"]} for k, v in TEMPLATES.items()})

# This route lists all models that are downloaded and info about them
@app.route("/api/models")
def api_models():
    models = []
    for f in sorted(MODELS_DIR.glob("*.gguf")):
        size_mb = round(f.stat().st_size / (1024 ** 2), 1)
        models.append({
            "name":               f.name,
            "size_mb":            size_mb,
            "loaded":             f.name == _loaded_name,
            "suggested_template": guess_template(f.name),
        })
    return jsonify({"models": models, "loaded": _loaded_name})

# This route calls the load function and wil make sure the selected model is loaded and sliced up to be ran
@app.route("/api/models/load", methods=["POST"])
def api_load():
    d           = request.json or {}
    filename    = d.get("filename")
    if not filename:
        return jsonify({"error": "filename required"}), 400

    max_seq_len  = int(d.get("max_seq_len", 4096))

    # Accept an explicit n_gpu_layers override; otherwise auto-detect
    n_gpu_layers = d.get("n_gpu_layers")        # may be None
    if n_gpu_layers is not None:
        n_gpu_layers = int(n_gpu_layers)

    t = threading.Thread(
        target=load_model,
        args=(filename,),
        kwargs={"max_seq_len": max_seq_len, "n_gpu_layers": n_gpu_layers},
        daemon=True,
    )
    t.start()
    return jsonify({
        "status":       "loading",
        "model":        filename,
        "n_gpu_layers": n_gpu_layers if n_gpu_layers is not None else HW.get("n_gpu_layers_auto", 0),
    })

# This will give the status of the model, various states indicate in what state the model is.
# Anything other than loaded wont work with chatting
@app.route("/api/models/status")
def api_model_status():
    return jsonify({
        "loaded":  _loaded_name,
        "ready":   _model is not None,
        "status":  _load_status,
        "error":   _load_error,
    })

# This route will unload the models, so that other models can be loaded again
# This also frees up a lot of vram and other precious resoures
@app.route("/api/models/unload", methods=["POST"])
def api_unload():
    unload_model()
    return jsonify({"status": "ok"})

# This route deleted a model, like delete the file
# Not the best idea ever, but this is what I came up with
# Since its open source and no one will ever read this. I am so scared to confess my love to someone. The only ones who understand are probably the ones that read this (no one)
# But actually, you can just modify the function

@app.route("/api/models/delete", methods=["POST"])
def api_delete():
    d = request.json or {}
    filename = d.get("filename")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    path = MODELS_DIR / filename
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    if _loaded_name == filename:
        unload_model()
    try:
        path.unlink()
    except OSError as e:
        return jsonify({"error": f"Could not delete file: {e}"}), 500
    return jsonify({"status": "ok"})

# This is the download url used for downloading
# Yeah, sounds stupid, but yeah
# This works pretty fast and Im proud of it!
# This process starts in a seperate thread
# This ensures safety and speed for all users
@app.route("/api/download", methods=["POST"])
def api_download():
    d        = request.json or {}
    url      = d.get("url", "").strip()
    filename = d.get("filename", "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400
    if not filename:
        filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith(".gguf"):
        filename += ".gguf"
    if filename in _downloads and not _downloads[filename].get("done") \
            and not _downloads[filename].get("error"):
        return jsonify({"error": "already downloading"}), 400
    threading.Thread(target=_download_worker, args=(url, filename), daemon=True).start()
    return jsonify({"status": "started", "filename": filename})

@app.route("/api/download/progress")
def api_download_progress():
    return jsonify(_downloads)

# Session related routes! DO SQLITE PLSSSS (YES im talking to myself)
# Please add sqlite
# This can be used later for various types of things
# LALALALALALLALLAL ITS MIDNIGHT AND IM STILLLLL AWAAAAAAAAAAAAAAAAAAAAAAAKE
## Update day 2: 2 hours left to code and 7 hours timeframe. Its midnight (00.22) and I really want to sleep
## This is probably not good, but I aint failing, so bear with my annoying comments! Fuck you if you hate it!
## And crunchwrap is on spotify playing atm, so this is fire!

# This route lists the sessions
@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    sessions = sorted(_sessions.values(), key=lambda s: s["updated_at"], reverse=True)
    return jsonify(sessions)

## This route can create sessions
@app.route("/api/sessions", methods=["POST"])
def api_new_session():
    d = request.json or {}
    return jsonify(new_session(d.get("name")))

# This route is able to fetch all information about a single sessions
@app.route("/api/sessions/<sid>", methods=["GET"])
def api_get_session(sid):
    s = get_session(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s)

## This performs actions on a specific session
@app.route("/api/sessions/<sid>", methods=["PUT"])
def api_update_session(sid):
    d = request.json or {}
    if sid not in _sessions:
        return jsonify({"error": "not found"}), 404
    update_session(sid, d.get("messages", _sessions[sid]["messages"]), d.get("name"))
    return jsonify(_sessions[sid])

## DELETE A SESSION!
@app.route("/api/sessions/<sid>", methods=["DELETE"])
def api_delete_session(sid):
    delete_session(sid)
    return jsonify({"status": "ok"})

## REname a session. 
# I have to tell you all something, I am typing and deletion // everytime. I always forget its # in python
# I have coded way to much in C last time
# I am back in python just for this
# This was in C first but I had to rewrite it in python since it had so many issues
@app.route("/api/sessions/<sid>/rename", methods=["POST"])
def api_rename_session(sid):
    d    = request.json or {}
    name = d.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if sid not in _sessions:
        return jsonify({"error": "not found"}), 404
    _sessions[sid]["name"] = name
    _save_sessions()
    return jsonify(_sessions[sid])

# The all mighty token counter! Hail the token counter! The allmighty token counter!

# This will just count the tokens so it makes sure it wont exceed the limit
# At this point im too tired to write funny comments
# This is just for safety and prevents overloading
@app.route("/api/tokens/count", methods=["POST"])
def api_count_tokens():
    d        = request.json or {}
    messages = d.get("messages", [])
    text     = " ".join(m.get("content", "") for m in messages)
    rough    = _rough_token_count(text)
    precise  = None
    if _model is not None:
        try:
            precise = len(_model.tokenize(text.encode()))
        except Exception:
            pass
    return jsonify({"rough": rough, "precise": precise})

# The chat stream, makes sure that everything is real-time and the user knows its doing something

# This is the chat stream route
# Should never be visited in browser,
# frontend calls it and it sends text while the model is generating
@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    d             = request.json or {}
    messages      = d.get("messages", [])
    max_tokens    = int(d.get("max_tokens", 512))
    temperature   = float(d.get("temperature", 0.7))
    top_p         = float(d.get("top_p", 0.9))
    template_key  = d.get("template", "chatml")
    max_ctx       = int(d.get("max_ctx", 4096))

    if _model is None:
        return jsonify({"error": "No model loaded"}), 400

    template = TEMPLATES.get(template_key, TEMPLATES["chatml"])

    trimmed = trim_messages(
        messages,
        max_context=max_ctx,
        max_tokens_reserved=max_tokens + 256
    )

    prompt = template["fn"](trimmed)

    # Just disable stop in some situations
    stop = template["stop"] if template["stop"] else None

    def generate():
        full_output = ""
        current_prompt = prompt
        max_rounds = 10   # safety limit to avoid infinite loops of doom 
        # Spoiler, doesnt work
        # My stupid model EndAI Small has a thing where it does it and this cant stop it sadly

        try:
            for round_i in range(max_rounds):

                stream = _model(
                    current_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    stop=stop,
                    echo=False,
                )

                chunk_text = ""

                for chunk in stream:
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        chunk_text += token
                        full_output += token
                        yield f"data: {json.dumps({'token': token})}\n\n"

                # Check if the model is actually done, if not just continue.
                # If its done stop.
                finish_reason = chunk["choices"][0].get("finish_reason", "")

                # If model stopped naturally → stop loop
                # We have to stop it when it stopped if it has a good finish reason
                if finish_reason == "stop":
                    break

                # If hit token limit → CONTINUE! BE A SLAVE TO MY SYSTEM!
                ## Yeah im crazy, that was me yesterday at 1 am
                ## This actually checks finishing reason
                if finish_reason == "length":
                    # Append what we got and ask it to continue
                    continuation_prompt = full_output.strip() + "\n\nContinue."

                    current_prompt = template["fn"](
                        trimmed + [{"role": "assistant", "content": continuation_prompt}]
                    )

                    continue

                # fallback for safety of us all!
                # And so the thing doesnt freak out and crash
                break

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# The main entry point of the code
# Listening to rebot rock rnnnnnn! Insanely nice when coding! tu tututututu tu tu tu tututututu 
# W song!
# This prints various information about the system and its useful for understanding if it detected your hardware
if __name__ == "__main__":
    gpu_info = "CPU only"
    if HW["cuda"]:
        vram = f"{HW['vram_mb']} MB VRAM" if HW["vram_mb"] else "VRAM unknown"
        layers = HW["n_gpu_layers_auto"]
        layer_str = "all layers" if layers == -1 else f"{layers} layers"
        gpu_info = f"CUDA · {HW['gpu_name']} · {vram} → offloading {layer_str}"
    elif HW["rocm"]:
        gpu_info = f"ROCm · {HW['gpu_name']} → offloading all layers"
    elif HW["metal"]:
        gpu_info = f"Metal · {HW['gpu_name']} → offloading all layers"

    print("=" * 65)
    print(f"  Hardware : {gpu_info}")
    print(f"  Threads  : {os.cpu_count()} logical cores ({max(1, os.cpu_count()-1)} used)")
    print(f"  Models   : {MODELS_DIR.resolve()}")
    print(f"  Sessions : {SESSIONS_FILE.resolve()}")
    print("  Serving  : http://localhost:5000")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
