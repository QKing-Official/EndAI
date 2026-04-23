#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  EndAI Universal Setup
#  Linux · macOS · Windows (Git Bash / WSL)
#
#  One-line install:
#    curl -fsSL https://raw.githubusercontent.com/QKing-Official/EndAI/main/setup.sh | bash
#    wget -qO- https://raw.githubusercontent.com/QKing-Official/EndAI/main/setup.sh | bash
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
R="\033[0m" BOLD="\033[1m" DIM="\033[2m"
RED="\033[91m" GRN="\033[92m" YLW="\033[93m" CYN="\033[96m" MAG="\033[95m"

log()  { echo -e "  ${CYN}→${R} $*"; }
ok()   { echo -e "  ${GRN}✔${R} $*"; }
warn() { echo -e "  ${YLW}⚠${R} $*"; }
err()  { echo -e "  ${RED}✖${R} $*"; exit 1; }
step() { echo -e "  ${MAG}●${R} $*"; }
ask()  {
  local prompt="$1" default="${2:-y}" hint
  [[ "$default" == "y" ]] && hint="${GRN}Y${R}/n" || hint="y/${GRN}N${R}"
  echo -ne "  ${YLW}?${R} $prompt [$hint]: "
  read -r ans 2>/dev/null || ans="$default"
  [[ "${ans:-$default}" =~ ^[Yy]$ ]]
}

banner() {
  echo -e "${CYN}${BOLD}"
  echo '  ███████╗███╗   ██╗██████╗      █████╗ ██╗'
  echo '  ██╔════╝████╗  ██║██╔══██╗    ██╔══██╗██║'
  echo '  █████╗  ██╔██╗ ██║██║  ██║    ███████║██║'
  echo '  ██╔══╝  ██║╚██╗██║██║  ██║    ██╔══██║██║'
  echo '  ███████╗██║ ╚████║██████╔╝    ██║  ██║██║'
  echo '  ╚══════╝╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚═╝'
  echo -e "${R}${DIM}  Universal Setup — github.com/QKing-Official/EndAI${R}"
  echo
}

# ── OS / Arch Detection ───────────────────────────────────────────────────────
detect_os() {
  case "$(uname -s 2>/dev/null)" in
    Linux*)
      if grep -qi microsoft /proc/version 2>/dev/null; then OS="wsl"
      else OS="linux"; fi ;;
    Darwin*) OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS="gitbash" ;;
    *) OS="unknown" ;;
  esac

  ARCH="$(uname -m 2>/dev/null || echo x86_64)"
  case "$ARCH" in arm64|aarch64) ARCH="arm64" ;; *) ARCH="x86_64" ;; esac
}

# ── Package Manager Detection (Linux) ────────────────────────────────────────
detect_pkg_manager() {
  if   command -v apt-get &>/dev/null; then PKG="apt"
  elif command -v dnf     &>/dev/null; then PKG="dnf"
  elif command -v pacman  &>/dev/null; then PKG="pacman"
  elif command -v zypper  &>/dev/null; then PKG="zypper"
  else PKG="unknown"; fi
}

pkg_install() {
  case "$PKG" in
    apt)    sudo apt-get install -y -qq "$@" ;;
    dnf)    sudo dnf install -y -q "$@" ;;
    pacman) sudo pacman -Sy --noconfirm "$@" ;;
    zypper) sudo zypper install -y "$@" ;;
    *)      warn "Unknown package manager — install $* manually"; return 1 ;;
  esac
}

# ── GPU Detection ─────────────────────────────────────────────────────────────
detect_gpu() {
  GPU_TYPE="cpu"; GPU_NAME="CPU only"; CUDA_TAG="cu121"

  if command -v nvidia-smi &>/dev/null; then
    local name driver major
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "${name:-}" ]]; then
      GPU_TYPE="nvidia"; GPU_NAME="$name"
      driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
      major="${driver%%.*}"
      (( major >= 545 )) && CUDA_TAG="cu121" || CUDA_TAG="cu118"
      return
    fi
  fi

  if [[ "$OS" == "linux" || "$OS" == "wsl" ]] && command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname &>/dev/null && { GPU_TYPE="amd"; GPU_NAME="ROCm GPU"; return; }
  fi

  if [[ "$OS" == "macos" ]]; then
    local chip
    chip=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2; exit}')
    GPU_TYPE="metal"; GPU_NAME="${chip:-Apple Metal}"
    return
  fi
}

# ── Python Detection & Install ────────────────────────────────────────────────
find_python() {
  for cmd in python3.12 python3.11 python3.10 python3 python; do
    command -v "$cmd" &>/dev/null || continue
    local major minor
    major=$("$cmd" -c "import sys;print(sys.version_info.major)" 2>/dev/null) || continue
    minor=$("$cmd" -c "import sys;print(sys.version_info.minor)" 2>/dev/null) || continue
    [[ "$major" -eq 3 && "$minor" -ge 10 ]] || continue
    PYTHON="$cmd"
    PYTHON_VER=$("$cmd" --version 2>&1 | awk '{print $2}')
    return 0
  done
  return 1
}

install_python() {
  warn "Python 3.10+ not found — attempting auto-install…"
  case "$OS" in
    linux|wsl)
      detect_pkg_manager
      case "$PKG" in
        apt)
          sudo apt-get update -qq
          pkg_install python3 python3-pip python3-venv ;;
        dnf)    pkg_install python3 python3-pip ;;
        pacman) pkg_install python python-pip ;;
        zypper) pkg_install python3 python3-pip ;;
        *) err "Cannot auto-install Python. Install 3.10+ from https://python.org" ;;
      esac ;;
    macos)
      if command -v brew &>/dev/null; then
        step "Installing Python 3.11 via Homebrew…"
        brew install python@3.11
      else
        err "Install Homebrew first: https://brew.sh  or Python: https://python.org"
      fi ;;
    gitbash)
      err "Install Python 3.10+ from https://python.org (tick 'Add to PATH') then re-run." ;;
    *)
      err "Cannot auto-install Python on this platform. Install 3.10+ from https://python.org" ;;
  esac
}

# ── Git Check ─────────────────────────────────────────────────────────────────
ensure_git() {
  command -v git &>/dev/null && return 0
  warn "git not found — installing…"
  case "$OS" in
    linux|wsl)
      detect_pkg_manager
      case "$PKG" in
        apt)    sudo apt-get update -qq; pkg_install git ;;
        dnf)    pkg_install git ;;
        pacman) pkg_install git ;;
        zypper) pkg_install git ;;
      esac ;;
    macos)
      command -v brew &>/dev/null && brew install git || \
        err "Run: xcode-select --install  to get git" ;;
    gitbash)
      err "Git Bash should already have git. Try reinstalling Git for Windows." ;;
  esac
}

# ── venv ──────────────────────────────────────────────────────────────────────
ensure_venv_deps() {
  # Debian/Ubuntu ships python3 without venv module — install it if missing
  if [[ "$OS" == "linux" || "$OS" == "wsl" ]]; then
    if ! "$PYTHON" -c "import venv" &>/dev/null; then
      warn "python3-venv not found — installing…"
      detect_pkg_manager
      case "$PKG" in
        apt)
          sudo apt-get update -qq
          # install the version-specific venv package (e.g. python3.11-venv)
          local pyver; pyver=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
          pkg_install "python${pyver}-venv" || pkg_install python3-venv ;;
        dnf)    pkg_install python3 ;;
        pacman) : ;; # venv is bundled on Arch
        zypper) pkg_install python3 ;;
      esac
    fi
  fi
}

setup_venv() {
  VENV_PATH="$REPO_PATH/.venv"
  if [[ -d "$VENV_PATH" ]]; then
    ok "Existing venv found at $VENV_PATH"
  else
    step "Creating virtual environment at $VENV_PATH…"
    "$PYTHON" -m venv "$VENV_PATH"
    ok "venv created"
  fi
  # Point PYTHON at the venv binary for all subsequent installs
  if [[ "$OS" == "gitbash" || "$OS" == "wsl" ]]; then
    PYTHON="$VENV_PATH/Scripts/python" 2>/dev/null || PYTHON="$VENV_PATH/bin/python"
  else
    PYTHON="$VENV_PATH/bin/python"
  fi
  [[ -x "$PYTHON" ]] || PYTHON="$VENV_PATH/bin/python3"
  ok "Using venv Python: $PYTHON"
}

# ── pip helper (always uses venv Python after setup_venv) ─────────────────────
pip_install() {
  "$PYTHON" -m pip install "$@" --quiet 2>&1
}

# ── Steps ─────────────────────────────────────────────────────────────────────

step_os_info() {
  echo -e "\n${BOLD}[1/7] Environment${R}"
  case "$OS" in
    linux)   ok "Linux ($(uname -r))" ;;
    wsl)     ok "Windows Subsystem for Linux" ;;
    macos)   ok "macOS $(sw_vers -productVersion 2>/dev/null || uname -r) ($ARCH)" ;;
    gitbash) ok "Windows — Git Bash" ;;
    *)       warn "Unknown OS: $(uname -s)" ;;
  esac
  ok "GPU: $GPU_NAME ($GPU_TYPE)"
}

step_python() {
  echo -e "\n${BOLD}[2/7] Python${R}"
  if ! find_python; then
    install_python
    find_python || err "Python 3.10+ still not found after install. Install manually: https://python.org"
  fi
  ok "Using $PYTHON ($PYTHON_VER)"
}

step_clone() {
  echo -e "\n${BOLD}[3/7] Repository${R}"
  if [[ -f "server.py" ]]; then
    REPO_PATH="$(pwd)"; ok "Already inside EndAI repo: $REPO_PATH"; return
  fi
  if [[ -d "EndAI" ]]; then
    REPO_PATH="$(pwd)/EndAI"; ok "Found existing EndAI/ directory"; return
  fi
  ensure_git
  step "Cloning EndAI…"
  git clone https://github.com/QKing-Official/EndAI.git
  REPO_PATH="$(pwd)/EndAI"
  ok "Cloned to $REPO_PATH"
}

step_venv() {
  echo -e "\n${BOLD}[4/7] Virtual Environment${R}"
  ensure_venv_deps
  setup_venv
}

step_base_deps() {
  echo -e "\n${BOLD}[5/7] Base Dependencies${R}"
  step "Upgrading pip…"
  "$PYTHON" -m pip install --upgrade pip --quiet && ok "pip upgraded" || warn "pip upgrade failed (non-fatal)"
  for pkg in flask numpy psutil; do
    step "Installing $pkg…"
    pip_install "$pkg" && ok "$pkg" || warn "Failed: $pkg"
  done
}

step_llama_cpp() {
  echo -e "\n${BOLD}[6/7] llama-cpp-python ($GPU_TYPE)${R}"
  case "$GPU_TYPE" in
    nvidia)
      log "NVIDIA — using $CUDA_TAG wheel"
      step "Downloading CUDA build (may take a few minutes)…"
      pip_install llama-cpp-python \
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${CUDA_TAG}" \
        && ok "llama-cpp-python (CUDA $CUDA_TAG)" \
        || { warn "CUDA wheel failed — falling back to CPU"
             pip_install llama-cpp-python && ok "llama-cpp-python (CPU fallback)" || warn "Install failed"; }
      ;;
    amd)
      log "AMD ROCm"
      pip_install llama-cpp-python && ok "llama-cpp-python (ROCm)" || warn "Install failed"
      ;;
    metal)
      log "Apple Metal"
      CMAKE_ARGS="-DLLAMA_METAL=on" pip_install llama-cpp-python \
        && ok "llama-cpp-python (Metal)" \
        || { warn "Metal build failed — trying plain"
             pip_install llama-cpp-python && ok "llama-cpp-python" || warn "Install failed"; }
      ;;
    *)
      log "CPU-only"
      pip_install llama-cpp-python && ok "llama-cpp-python (CPU)" || warn "Install failed"
      ;;
  esac
}

step_models() {
  echo -e "\n${BOLD}[7/7] Models${R}"
  mkdir -p "$REPO_PATH/models"
  ok "models/ ready at $REPO_PATH/models"

  if ! ls "$REPO_PATH/models/"*.gguf &>/dev/null; then
    warn "No .gguf models found."
    echo -e "
  ${DIM}Grab a model from HuggingFace:${R}
    ${CYN}Full : https://huggingface.co/QKing-Official/EndAI${R}
    ${CYN}Small: https://huggingface.co/QKing-Official/EndAI-Small${R}
  Place the .gguf in: ${BOLD}$REPO_PATH/models/${R}
"
    if ask "Download EndAI-Small Q4_K_M (~700 MB) now?" "n"; then
      local url="https://huggingface.co/QKing-Official/EndAI-Small/resolve/main/EndAI-Small-Q4_K_M.gguf"
      local dest="$REPO_PATH/models/EndAI-Small-Q4_K_M.gguf"
      step "Downloading to $dest…"
      if command -v curl &>/dev/null; then
        curl -L --progress-bar "$url" -o "$dest" && ok "Download complete"
      elif command -v wget &>/dev/null; then
        wget -q --show-progress "$url" -O "$dest" && ok "Download complete"
      else
        warn "Install curl or wget to download automatically."
      fi
    fi
  else
    for f in "$REPO_PATH/models/"*.gguf; do ok "Found: $(basename "$f")"; done
  fi
}

summary_and_launch() {
  echo -e "
${BOLD}${GRN}  ══════════════════════════════════${R}
${BOLD}${GRN}   Setup Complete!${R}
${BOLD}${GRN}  ══════════════════════════════════${R}

  ${DIM}OS    :${R} $OS ($ARCH)
  ${DIM}GPU   :${R} $GPU_NAME  (${CYN}${GPU_TYPE}${R})
  ${DIM}Python:${R} $PYTHON_VER
  ${DIM}Repo  :${R} $REPO_PATH
  ${DIM}Venv  :${R} $VENV_PATH

  ${BOLD}Start EndAI:${R}
    cd \"$REPO_PATH\"
    source .venv/bin/activate
    python server.py

  Then open ${CYN}http://localhost:5000${R}
"
  if ask "Launch EndAI server now?" "y"; then
    [[ -f "$REPO_PATH/server.py" ]] || { err "server.py not found."; }
    step "Starting EndAI… (Ctrl+C to stop)"
    cd "$REPO_PATH"
    "$PYTHON" server.py
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  banner
  detect_os
  detect_gpu
  log "Detected: OS=$OS  ARCH=$ARCH  GPU=$GPU_TYPE ($GPU_NAME)"
  echo
  ask "Proceed with installation?" "y" || { echo "  Aborted."; exit 0; }

  step_os_info
  step_python
  step_clone
  step_venv
  step_base_deps
  step_llama_cpp
  step_models
  summary_and_launch
}

main