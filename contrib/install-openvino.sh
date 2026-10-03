#!/usr/bin/env bash
# Speech to text on an Intel NPU or integrated GPU, through OpenVINO.
#
# Sets up contrib/openvino_whisper_server.py as a drop-in for whisper-server:
# a Python venv with openvino-genai, the OpenVINO export of a Whisper model, and
# a `dikte-openvino-whisper` command. Then, in Dikte: Settings → API and models →
# local whisper-server binary = ~/.local/bin/dikte-openvino-whisper.
#
#   ./contrib/install-openvino.sh [hf-model]     default: OpenVINO/whisper-large-v3-turbo-int8-ov
#
# VENV=... picks another venv (an existing one with openvino-genai is reused).
# For the NPU, the distribution's NPU user-mode driver and compiler are needed
# too (Fedora: intel-npu-driver intel-npu-compiler); without them it uses the GPU.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=${1:-OpenVINO/whisper-large-v3-turbo-int8-ov}
data=${XDG_DATA_HOME:-$HOME/.local/share}/dikte
venv=${VENV:-$data/openvino-venv}
models=$data/models/openvino
model=$models/${repo##*/}
config=${XDG_CONFIG_HOME:-$HOME/.config}/dikte/openvino.json
wrapper=$HOME/.local/bin/dikte-openvino-whisper

command -v ffmpeg >/dev/null || echo "note: ffmpeg is needed for audio that is not 16 kHz mono WAV"

if [[ ! -x $venv/bin/python ]] || ! "$venv/bin/python" -c 'import openvino_genai' 2>/dev/null; then
    echo "Creating $venv"
    python3 -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet openvino-genai numpy huggingface_hub
fi

if [[ ! -f $model/openvino_encoder_model.xml ]]; then
    echo "Downloading $repo"
    "$venv/bin/pip" install --quiet huggingface_hub
    "$venv/bin/python" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' "$repo" "$model"
fi

mkdir -p "$(dirname "$config")" "$(dirname "$wrapper")"
if [[ ! -f $config ]]; then
    printf '{"model": "%s", "devices": ["GPU", "NPU", "CPU"]}\n' "$model" >"$config"
fi
cat >"$wrapper" <<WRAP
#!/bin/sh
exec "$venv/bin/python" "$here/openvino_whisper_server.py" "\$@"
WRAP
chmod +x "$wrapper"

echo "Devices OpenVINO sees: $("$venv/bin/python" -c 'import openvino as ov; print(", ".join(ov.Core().available_devices))')"
echo "Done. In Dikte set the local whisper-server binary to: $wrapper"
