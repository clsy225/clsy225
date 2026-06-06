import argparse
import os
import sys
import threading
import time
import traceback
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def wav_bytes(sample_rate, audio, gain=1.0):
    audio = np.asarray(audio)
    if audio.size == 0:
        pcm = np.zeros(0, dtype=np.int16)
    else:
        audio_f = audio.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(audio_f)))
        if np.issubdtype(audio.dtype, np.floating) and max_abs <= 1.5:
            pcm_f = audio_f * 32767.0
        else:
            pcm_f = audio_f
        pcm = np.clip(pcm_f * float(gain), -32768.0, 32767.0).astype(np.int16)

    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return bio.getvalue()


def first_value(data, *names, default=None):
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return default


parser = argparse.ArgumentParser(description="GPT-SoVITS RKNN keepalive API")
parser.add_argument("-g", "--gpt", required=True, help="GPT/Text2Semantic ckpt path")
parser.add_argument("-s", "--sovits", required=True, help="SoVITS pth path")
parser.add_argument("-a", "--bind_addr", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9880)
parser.add_argument("-dr", "--default_refer", required=True, help="default reference wav")
parser.add_argument("-dt", "--default_refer_text", required=True, help="default prompt text")
parser.add_argument("-dl", "--default_refer_lang", required=True, help="default prompt language")
parser.add_argument("--repo", default="/home/linaro/GPTSoVITS")
parser.add_argument("--rknn", default="/userdata/rknn_voice_test/models/vits_no_split.rknn")
parser.add_argument("--gain", type=float, default=0.35, help="WAV output gain, default avoids clipping on this board")
args = parser.parse_args()

os.environ.pop("TORCH_CPP_LOG_LEVEL", None)

overlay_dir = Path(__file__).resolve().parent
repo_dir = Path(args.repo).resolve()
sys.path.insert(0, str(overlay_dir))
sys.path.insert(1, str(repo_dir))
sys.path.insert(2, str(repo_dir / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS

config = {
    "custom": {
        "device": "cpu",
        "is_half": False,
        "version": "v2Pro",
        "t2s_weights_path": args.gpt,
        "vits_weights_path": args.sovits,
        "bert_base_path": str(repo_dir / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(repo_dir / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        "vits_rknn_path": args.rknn,
        "vits_rknn_target": "",
    }
}

print("Loading GPT-SoVITS RKNN keepalive API...")
tts = TTS(config)
lock = threading.Lock()
app = FastAPI()


def normalize_request(data):
    text = first_value(data, "text", default="")
    text_lang = first_value(data, "text_language", "text_lang", default="zh")
    ref_audio_path = first_value(data, "refer_wav_path", "ref_audio_path", default=args.default_refer)
    prompt_text = first_value(data, "prompt_text", default=args.default_refer_text)
    prompt_lang = first_value(data, "prompt_language", "prompt_lang", default=args.default_refer_lang)
    speed = float(first_value(data, "speed", "speed_factor", default=1.0))
    gain = float(first_value(data, "gain", default=args.gain))

    return {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": data.get("inp_refs") or data.get("aux_ref_audio_paths") or [],
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": int(first_value(data, "top_k", default=5)),
        "top_p": float(first_value(data, "top_p", default=1)),
        "temperature": float(first_value(data, "temperature", default=1)),
        "text_split_method": first_value(data, "text_split_method", default="cut5"),
        "batch_size": int(first_value(data, "batch_size", default=1)),
        "batch_threshold": float(first_value(data, "batch_threshold", default=0.75)),
        "split_bucket": False,
        "speed_factor": speed,
        "fragment_interval": float(first_value(data, "fragment_interval", default=0.3)),
        "seed": int(first_value(data, "seed", default=-1)),
        "parallel_infer": False,
        "repetition_penalty": float(first_value(data, "repetition_penalty", default=1.35)),
        "sample_steps": int(first_value(data, "sample_steps", default=32)),
        "super_sampling": False,
        "streaming_mode": False,
        "_gain": gain,
    }


async def collect_data(request: Request):
    data = dict(request.query_params)
    if request.method == "POST":
        try:
            body = await request.json()
            if isinstance(body, dict):
                data.update(body)
        except Exception:
            pass
    return data


async def synthesize(request: Request):
    data = await collect_data(request)
    req = normalize_request(data)
    if not req["text"]:
        return JSONResponse(status_code=400, content={"message": "text is required"})

    gain = req.pop("_gain")
    t0 = time.perf_counter()
    try:
        with lock:
            sr, audio = next(tts.run(req))
        elapsed = time.perf_counter() - t0
        content = wav_bytes(sr, audio, gain=gain)
        return Response(
            content=content,
            media_type="audio/wav",
            headers={
                "X-Elapsed-Seconds": f"{elapsed:.3f}",
                "X-Audio-Sample-Rate": str(sr),
                "X-Audio-Samples": str(len(audio)),
            },
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"message": str(exc), "type": type(exc).__name__})


@app.get("/")
async def get_root(request: Request):
    return await synthesize(request)


@app.post("/")
async def post_root(request: Request):
    return await synthesize(request)


@app.get("/tts")
async def get_tts(request: Request):
    return await synthesize(request)


@app.post("/tts")
async def post_tts(request: Request):
    return await synthesize(request)


@app.get("/health")
async def health():
    return {"ok": True, "rknn": args.rknn, "default_refer": args.default_refer}


if __name__ == "__main__":
    uvicorn.run(app, host=args.bind_addr, port=args.port, log_config=None)
