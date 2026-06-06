#!/usr/bin/env python3
"""
AI Engine 核心模块 - 本地/云端双模式AI推理

支持:
  - local: 通过RK3588 NPU推理 (llm_ask)
  - cloud: 通过OpenAI兼容API推理

用法:
  from ai_engine import AIEngine
  engine = AIEngine()
  result = engine.ask("你好")
  print(result["answer"])
"""

import os
import json
import time
import subprocess
import logging
import re
import urllib.request
import urllib.error

logger = logging.getLogger("AIEngine")

# 路径
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/userdata/models/Qwen.rkllm"
LLM_BINARY = os.path.join(WORKSPACE, "llm_ask")
CONFIG_FILE = os.path.join(WORKSPACE, "ai_config.json")

DEFAULT_CONFIG = {
    "mode": "local",
    "local": {
        "max_tokens": 512,
        "max_context": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "timeout": 120,
    },
    "cloud": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key": "",
        "model": "gpt-4o-mini",
        "max_tokens": 1024,
        "temperature": 0.7,
        "timeout": 30,
    }
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg)
            return merged
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def clean_output(output):
    lines = output.split('\n')
    parts = []
    for line in lines:
        s = line.strip()
        if s.startswith("I rkllm:") or s.startswith("W rkllm:"):
            continue
        if s in ("", "robot:", "rkllm init start", "rkllm init success",
                 "rkllm init failed", "rkllm destroy done"):
            continue
        if "rkllm destroy" in s or "程序即将退出" in s:
            continue
        parts.append(line)
    return "\n".join(parts).strip()


def parse_perf(output):
    perf = {}
    for line in output.split('\n'):
        m = re.search(r'Peak Memory Usage.*?([\d.]+)\s*GB', line)
        if m: perf['peak_memory_gb'] = float(m.group(1))
        m = re.search(r'Model init time.*?([\d.]+)', line)
        if m: perf['init_time_ms'] = float(m.group(1))
        if 'Prefill' in line and 'Stage' not in line:
            nums = re.findall(r'([\d.]+)', line)
            if len(nums) >= 4:
                perf['generate_tokens'] = int(float(nums[1]))
                perf['generate_speed_tps'] = float(nums[3])
        if 'Generate' in line and 'Stage' not in line and ':' in line:
            nums = re.findall(r'([\d.]+)', line)
            if len(nums) >= 4:
                perf['generate_tokens'] = int(float(nums[1]))
                perf['generate_speed_tps'] = float(nums[3])
    return perf


class LocalEngine:
    def __init__(self, config):
        self.config = config

    def ask(self, prompt):
        cfg = self.config["local"]
        env = os.environ.copy()
        env["RKLLM_LOG_LEVEL"] = "1"
        env["LD_LIBRARY_PATH"] = f"{WORKSPACE}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [LLM_BINARY, prompt]
        start = time.time()
        try:
            result = subprocess.run(cmd, env=env, capture_output=True,
                                    text=True, timeout=cfg["timeout"])
            elapsed = time.time() - start
            raw = result.stdout
            return {"success": True, "answer": clean_output(raw),
                    "elapsed": round(elapsed, 2), "perf": parse_perf(raw),
                    "mode": "local"}
        except subprocess.TimeoutExpired:
            return {"success": False, "answer": f"⏰ 推理超时 ({cfg['timeout']}s)",
                    "elapsed": cfg["timeout"], "perf": {}, "mode": "local"}
        except Exception as e:
            return {"success": False, "answer": f"❌ 错误: {e}",
                    "elapsed": round(time.time()-start, 2), "perf": {}, "mode": "local"}

    def available(self):
        return os.path.exists(MODEL_PATH) and os.path.exists(LLM_BINARY)


class CloudEngine:
    def __init__(self, config):
        self.config = config

    def ask(self, prompt):
        cfg = self.config["cloud"]
        if not cfg.get("api_key"):
            return {"success": False, "answer": "❌ 未配置API密钥",
                    "elapsed": 0, "perf": {}, "mode": "cloud"}

        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {cfg['api_key']}"}
        payload = json.dumps({
            "model": cfg.get("model", "gpt-4o-mini"),
            "messages": [{"role": "system", "content": "你是一个有用的AI助手。"},
                         {"role": "user", "content": prompt}],
            "max_tokens": cfg.get("max_tokens", 1024),
            "temperature": cfg.get("temperature", 0.7),
        }).encode()

        start = time.time()
        try:
            req = urllib.request.Request(cfg["api_url"], data=payload,
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=cfg.get("timeout", 30)) as resp:
                data = json.loads(resp.read().decode())
                elapsed = time.time() - start
                answer = ""
                choices = data.get("choices", [])
                if choices:
                    answer = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                perf = {"prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)}
                return {"success": True, "answer": answer,
                        "elapsed": round(elapsed, 2), "perf": perf, "mode": "cloud"}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            return {"success": False, "answer": f"🌐 API错误 ({e.code}): {body}",
                    "elapsed": round(time.time()-start, 2), "perf": {}, "mode": "cloud"}
        except urllib.error.URLError as e:
            return {"success": False, "answer": f"🌐 网络错误: {e.reason}",
                    "elapsed": round(time.time()-start, 2), "perf": {}, "mode": "cloud"}
        except Exception as e:
            return {"success": False, "answer": f"❌ 错误: {e}",
                    "elapsed": round(time.time()-start, 2), "perf": {}, "mode": "cloud"}

    def available(self):
        return bool(self.config["cloud"].get("api_key"))


class AIEngine:
    def __init__(self, config_file=CONFIG_FILE):
        self.config_file = config_file
        self.config = load_config()
        self.local = LocalEngine(self.config)
        self.cloud = CloudEngine(self.config)

    def ask(self, prompt, mode=None):
        if not prompt or not prompt.strip():
            return {"success": False, "answer": "请输入问题",
                    "elapsed": 0, "perf": {}, "mode": "none"}
        mode = mode or self.config.get("mode", "local")
        eng = {"local": self.local, "cloud": self.cloud}.get(mode)
        if not eng:
            return {"success": False, "answer": f"未知模式: {mode}"}
        return eng.ask(prompt)

    def ask_auto(self, prompt):
        if self.cloud.available():
            r = self.cloud.ask(prompt)
            if r["success"]:
                return r
        if self.local.available():
            return self.local.ask(prompt)
        return {"success": False,
                "answer": "❌ 本地和云端都不可用。请检查模型或API配置。",
                "elapsed": 0, "perf": {}, "mode": "none"}

    def switch_mode(self, mode):
        if mode not in ("local", "cloud"):
            return False
        self.config["mode"] = mode
        save_config(self.config)
        return True

    def update_config(self, updates):
        def deep(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep(d[k], v)
                else:
                    d[k] = v
        deep(self.config, updates)
        save_config(self.config)
        self.local = LocalEngine(self.config)
        self.cloud = CloudEngine(self.config)

    def get_status(self):
        return {"mode": self.config["mode"],
                "local_available": self.local.available(),
                "cloud_available": self.cloud.available(),
                "local_model": "Qwen.rkllm" if self.local.available() else None,
                "cloud_model": self.config["cloud"].get("model", "")}

    def get_config_safe(self):
        cfg = json.loads(json.dumps(self.config))
        key = cfg["cloud"].get("api_key", "")
        if key:
            cfg["cloud"]["api_key"] = key[:8] + "****" if len(key) > 8 else "****"
        return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[AI] %(message)s")
    import sys
    engine = AIEngine()
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-i", "--interactive"):
            print("交互模式，输入exit退出")
            while True:
                q = input(">>> ").strip()
                if q.lower() in ("exit", "quit"):
                    break
                if q:
                    r = engine.ask(q)
                    mode = "🖥️本地" if r["mode"] == "local" else "🌐云端"
                    status = "✅" if r["success"] else "❌"
                    print(f"  {status}[{mode}] ({r['elapsed']}s)")
                    print(f"  {r['answer']}\n")
        else:
            r = engine.ask(" ".join(sys.argv[1:]))
            print(r["answer"])
