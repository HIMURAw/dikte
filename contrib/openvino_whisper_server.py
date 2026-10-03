"""A drop-in for whisper-server that runs Whisper through OpenVINO.

Dikte starts its local speech-to-text server as

    <binary> -m <model> --inference-path /v1/audio/transcriptions -l auto ... --host H --port P

and treats the port opening as "ready". This script takes the same arguments,
answers the same OpenAI-shaped requests (json and verbose_json), and runs the
model on the integrated GPU, Intel's NPU or the CPU, whichever loads first in
the order given. Set Settings → API and models → whisper-server binary to the
`dikte-openvino-whisper` wrapper that install-openvino.sh writes.

The ggml model Dikte passes with -m is ignored: OpenVINO needs its own export
of the model, named in ~/.config/dikte/openvino.json:

    {"model": "~/.local/share/dikte/models/openvino/whisper-large-v3-turbo-int8-ov",
     "devices": ["GPU", "NPU", "CPU"]}

Needs openvino-genai and numpy (install-openvino.sh puts them in a venv), and
ffmpeg for audio that is not 16 kHz mono WAV.
"""

import argparse
import email.parser
import email.policy
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import openvino_genai as og

RATE = 16000
CONFIG = pathlib.Path(os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")) / "dikte" / "openvino.json"
DEFAULT_MODEL = "~/.local/share/dikte/models/openvino/whisper-large-v3-turbo-int8-ov"
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "dikte" / "openvino"


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def load_config(no_gpu):
    conf = {}
    if CONFIG.is_file():
        conf = json.loads(CONFIG.read_text())
    model = pathlib.Path(os.path.expanduser(conf.get("model", DEFAULT_MODEL)))
    devices = conf.get("devices") or ["GPU", "NPU", "CPU"]
    if no_gpu:
        devices = [d for d in devices if d != "GPU"]
    return model, devices


def open_pipeline(model, devices):
    CACHE.mkdir(parents=True, exist_ok=True)
    errors = []
    for device in devices:
        try:
            started = time.monotonic()
            pipe = og.WhisperPipeline(str(model), device, CACHE_DIR=str(CACHE))
            # The first run compiles kernels; pay for it now rather than on the
            # first dictation.
            pipe.generate(np.zeros(RATE, dtype=np.float32))
            log(f"loaded {model.name} on {device} in {time.monotonic() - started:.1f}s")
            return pipe, device
        except Exception as exc:  # noqa: BLE001 - any failure means try the next device
            first_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
            log(f"{device} failed: {first_line}")
            errors.append(f"{device}: {first_line}")
    sys.exit("no device could load the model: " + "; ".join(errors))


def read_audio(data):
    """16 kHz mono float32 samples from any audio file ffmpeg can read."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    except (wave.Error, EOFError):
        pass
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32)


def parse_form(headers, body):
    """Fields and the uploaded file of a multipart/form-data body."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + headers["Content-Type"].encode() + b"\r\n\r\n" + body
    )
    fields, upload = {}, None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if part.get_filename() is not None or name == "file":
            upload = payload
        elif name:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, upload


class Server:
    def __init__(self, pipe, default_language):
        self.pipe = pipe
        self.default_language = default_language
        self.lock = threading.Lock()

    def transcribe(self, audio, language, prompt, timestamps):
        config = self.pipe.get_generation_config()
        language = language or self.default_language
        config.language = f"<|{language}|>" if language and language != "auto" else None
        config.task = "transcribe"
        config.return_timestamps = timestamps
        if prompt and hasattr(config, "initial_prompt"):
            config.initial_prompt = prompt
        with self.lock:
            return self.pipe.generate(audio, config)


def handler_for(server, inference_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log("http", fmt % args)

        def _send(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok"})

        def do_POST(self):
            if self.path.rstrip("/") != inference_path.rstrip("/"):
                self._send(404, {"error": {"message": f"no such path {self.path}"}})
                return
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fields, upload = parse_form(self.headers, body)
                if not upload:
                    self._send(400, {"error": {"message": "no audio file in the request"}})
                    return
                audio = read_audio(upload)
                verbose = fields.get("response_format") == "verbose_json"
                started = time.monotonic()
                result = server.transcribe(audio, fields.get("language"), fields.get("prompt"), verbose)
                text = "".join(result.texts) if result.texts else ""
                log(f"{len(audio) / RATE:.1f}s of audio in {time.monotonic() - started:.2f}s")
                if not verbose:
                    self._send(200, {"text": text})
                    return
                segments = [
                    {"id": i, "start": float(c.start_ts), "end": float(max(c.end_ts, c.start_ts)), "text": c.text}
                    for i, c in enumerate(result.chunks or [])
                ]
                self._send(200, {"text": text, "segments": segments, "duration": len(audio) / RATE})
            except Exception as exc:  # noqa: BLE001 - the client gets the reason
                log("error:", exc)
                self._send(500, {"error": {"message": str(exc)}})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-m", "--model")  # the ggml file Dikte passes; not used
    parser.add_argument("--inference-path", default="/inference")
    parser.add_argument("-l", "--language", default="auto")
    parser.add_argument("-ng", "--no-gpu", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    # whisper-server flags that mean nothing here (-sns, -nlp, -t N, ...).
    args, ignored = parser.parse_known_args()
    if ignored:
        log("ignoring", " ".join(ignored))

    model, devices = load_config(args.no_gpu)
    if not (model / "openvino_encoder_model.xml").is_file():
        sys.exit(f"no OpenVINO whisper model at {model} (see {CONFIG})")
    pipe, device = open_pipeline(model, devices)
    server = Server(pipe, args.language)
    httpd = ThreadingHTTPServer((args.host, args.port), handler_for(server, args.inference_path))
    log(f"listening on {args.host}:{args.port}{args.inference_path} ({device})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
