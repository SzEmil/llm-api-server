import os
import uuid
import time
from typing import List
import gc
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()


# ---------- helpers ----------
def env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes", "y", "on")


def env_dtype(key: str, default: str = "bfloat16"):
    v = os.getenv(key, default).lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    return torch.bfloat16


# ---------- ENV config ----------
MODEL_NAME = os.getenv("MODEL_NAME", "speakleash/Bielik-4.5B-v3.0-Instruct")
LOAD_IN_4BIT = env_bool("LOAD_IN_4BIT", True)
BNB_4BIT_TYPE = os.getenv("BNB_4BIT_TYPE", "nf4")
BNB_COMPUTE_DTYPE = env_dtype("BNB_4BIT_COMPUTE", "bfloat16")
USE_FAST_TOKENIZER = env_bool("USE_FAST_TOKENIZER", True)
MAX_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", "1024"))
TRUST_REMOTE_CODE = env_bool("TRUST_REMOTE_CODE", True)

DO_SAMPLE = env_bool("DO_SAMPLE", True)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
TOP_K = int(os.getenv("TOP_K", "50"))

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT", "Jesteś pomocnym asystentem. Odpowiadaj po polsku."
)

ATTN_IMPL = os.getenv("ATTN_IMPL", "auto")  # auto|fa3|sdpa|eager
CACHE_IMPL = os.getenv("CACHE_IMPL", "dynamic")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "16384"))
CHUNK_PREFILL = int(os.getenv("CHUNK_PREFILL", "0"))  # np. 4096; 0=off
FREE_CUDA_CACHE_AFTER_REQUEST = env_bool("FREE_CUDA_CACHE_AFTER_REQUEST", True)


# ---------- API models ----------
class Statistics(BaseModel):
    processingTimeMs: int


class Message(BaseModel):
    id: str
    content: str


class PromptRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    message_id: str
    messages: List[Message]
    statistics: Statistics


# ---------- App ----------
model_pipeline: dict = {}
app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[BOOT] MODEL_NAME={MODEL_NAME}")
    print(f"[BOOT] LOAD_IN_4BIT={LOAD_IN_4BIT}")

    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    cuda_ok = torch.cuda.is_available()
    cc = None
    if cuda_ok:
        cc = torch.cuda.get_device_capability(0)
        print(f"[GPU] name={torch.cuda.get_device_name(0)} CC={cc}")

    # 1) tokenizer
    print("[BOOT] Ładowanie tokenizera...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=USE_FAST_TOKENIZER,
            token=hf_token,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
    except Exception as e:
        print(f"[WARN] Fast tokenizer nie działa ({e}). Fallback na slow.")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=False,
            token=hf_token,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

    # 2) config — sprawdźmy MXFP4
    wants_mxfp4 = False
    try:
        cfg = AutoConfig.from_pretrained(
            MODEL_NAME, token=hf_token, trust_remote_code=TRUST_REMOTE_CODE
        )
        qcfg = getattr(cfg, "quantization_config", None)
        wants_mxfp4 = (
            isinstance(qcfg, dict)
            and str(qcfg.get("quant_method", "")).lower() == "mxfp4"
        )
        print(f"[DEBUG] model.quantization_config={qcfg}")
    except Exception as e:
        print(f"[WARN] Nie udało się odczytać AutoConfig: {e}")

    supports_mxfp4 = bool(cuda_ok and cc and cc[0] >= 9)
    print(f"[DEBUG] supports_mxfp4={supports_mxfp4} wants_mxfp4={wants_mxfp4}")

    # 3) ładowanie modelu
    model = None

    if wants_mxfp4 and supports_mxfp4:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                attn_implementation=ATTN_IMPL,
                token=hf_token,
                trust_remote_code=TRUST_REMOTE_CODE,
            )
            print("[OK] Załadowano w trybie MXFP4.")
        except Exception as e:
            print(f"[WARN] MXFP4 nie działa ({e}). Próba alternatyw.")

    if model is None and LOAD_IN_4BIT and cuda_ok:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=BNB_4BIT_TYPE,
                bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                quantization_config=quant_cfg,
                attn_implementation=ATTN_IMPL,
                token=hf_token,
                trust_remote_code=TRUST_REMOTE_CODE,
            )
            print("[OK] Załadowano w trybie BitsAndBytes 4-bit.")
        except Exception as e:
            print(f"[WARN] BnB 4-bit nie działa ({e}). Próba BF16/FP32.")

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=(torch.bfloat16 if cuda_ok else torch.float32),
            attn_implementation=ATTN_IMPL,
            token=hf_token,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        print("[OK] Załadowano w trybie BF16/FP32.")

    try:
        print("[DEBUG] runtime dtype:", next(model.parameters()).dtype)
        print("[DEBUG] hf_device_map:", getattr(model, "hf_device_map", None))
    except Exception:
        pass

    model_pipeline["tokenizer"] = tokenizer
    model_pipeline["model"] = model
    print("[READY] Model gotowy.")
    yield

    print("[SHUTDOWN] Sprzątanie...")
    model_pipeline.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


@app.get("/healthz")
async def healthz():
    ok = ("model" in model_pipeline) and ("tokenizer" in model_pipeline)
    return {"ok": ok, "model": MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
async def generate_response(req: PromptRequest):
    if "model" not in model_pipeline or "tokenizer" not in model_pipeline:
        raise HTTPException(status_code=503, detail="Model nie gotowy.")

    start = time.perf_counter()
    tok = model_pipeline["tokenizer"]
    mdl = model_pipeline["model"]

    user_text = (req.prompt or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Pusty prompt.")

    # ------------------- MESSAGES -------------------
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    # deepseek wymaga tylko usera
    if MODEL_NAME.lower().startswith("deepseek-ai/deepseek-r1-distill-"):
        messages = [{"role": "user", "content": user_text}]

    # openai/gpt-oss / pllumm / nemotron → Harmony-format
    use_harmony = any(
        key in MODEL_NAME.lower()
        for key in [
            "openai/gpt-oss",
            "nvidia/llama-3_3-nemotron",
            "cyfragovpl/llama-pllum",
        ]
    )

    # ------------------- RENDER -------------------
    try:
        if use_harmony:
            rendered = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            rendered = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
    except Exception:
        rendered = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    # ------------------- TOKENIZACJA -------------------
    enc = tok(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_ids = enc["input_ids"].to(mdl.device)
    attention_mask = enc["attention_mask"].to(mdl.device)

    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|eot_id|>")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    pad_id = tok.pad_token_id

    max_new = int(MAX_TOKENS) if MAX_TOKENS > 0 else 512
    USE_CACHE_ENV = os.getenv("USE_CACHE", "1").lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )

    gen_kwargs = dict(
        max_new_tokens=max_new,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        do_sample=DO_SAMPLE,
        use_cache=USE_CACHE_ENV,
    )
    if DO_SAMPLE:
        gen_kwargs.update(dict(temperature=TEMPERATURE, top_k=TOP_K))

    # ------------------- CHUNK PREFILL -------------------
    past_key_values = None
    pre_len = input_ids.shape[1]

    if CHUNK_PREFILL > 0 and pre_len > CHUNK_PREFILL:
        with torch.inference_mode():
            for i in range(0, pre_len, CHUNK_PREFILL):
                chunk = input_ids[:, i : i + CHUNK_PREFILL]
                attn = torch.ones_like(chunk, device=mdl.device)
                out = mdl(
                    chunk,
                    attention_mask=attn,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values

        with torch.inference_mode():
            out_ids = mdl.generate(
                input_ids[:, -CHUNK_PREFILL:],  # ostatni fragment
                attention_mask=attention_mask[:, -CHUNK_PREFILL:],
                past_key_values=past_key_values,
                **gen_kwargs,
            )
    else:
        with torch.inference_mode():
            out_ids = mdl.generate(
                input_ids, attention_mask=attention_mask, **gen_kwargs
            )

    new_tokens = out_ids[0, pre_len:]
    response_text = tok.decode(new_tokens, skip_special_tokens=True).strip()

    ms = round((time.perf_counter() - start) * 1000)

    if FREE_CUDA_CACHE_AFTER_REQUEST and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "message_id": str(uuid.uuid4()),
        "messages": [{"id": str(uuid.uuid4()), "content": response_text}],
        "statistics": {"processingTimeMs": ms},
    }
