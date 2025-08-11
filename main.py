import os
import uuid
import time
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
BNB_4BIT_TYPE = os.getenv("BNB_4BIT_TYPE", "nf4")  # nf4 | fp4
BNB_COMPUTE_DTYPE = env_dtype("BNB_4BIT_COMPUTE", "bfloat16")
USE_FAST_TOKENIZER = env_bool("USE_FAST_TOKENIZER", True)
MAX_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", "1024"))
TRUST_REMOTE_CODE = env_bool("TRUST_REMOTE_CODE", True)

DO_SAMPLE = env_bool("DO_SAMPLE", True)                  # np. DO_SAMPLE=false dla ekstrakcji
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))     # np. TEMPERATURE=0.0
TOP_K = int(os.getenv("TOP_K", "50"))  

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT", "Jesteś pomocnym asystentem. Odpowiadaj po polsku."
)


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

    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    quant_cfg = None

    # 4-bit config (z automatycznym guardem na brak GPU)
    if LOAD_IN_4BIT:
        if not torch.cuda.is_available():
            print("[WARN] Brak CUDA -> 4-bit wyłączony (fallback na FP/BF16).")
            LOAD = "fp"  # lokalna flaga
        else:
            LOAD = "4bit"
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=BNB_4BIT_TYPE,
                bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
            )
    else:
        LOAD = "fp"

    # tokenizer (fast -> slow fallback)
    print("[BOOT] Ładowanie tokenizera...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=USE_FAST_TOKENIZER, token=token, trust_remote_code=TRUST_REMOTE_CODE
        )
    except Exception as e:
        print(f"[WARN] Fast tokenizer nie działa ({e}). Fallback na slow.")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=False, token=token, trust_remote_code=TRUST_REMOTE_CODE
        )

    # model: najpierw spróbuj 4-bit, potem FP/BF16 (fallback)
    print("[BOOT] Ładowanie modelu (to może potrwać)...")
    model = None
    if LOAD == "4bit":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                quantization_config=quant_cfg,
                token=token,
                trust_remote_code=TRUST_REMOTE_CODE,
            )
        except Exception as e:
            print(f"[WARN] 4-bit nie działa ({e}). Fallback na FP/BF16.")
            LOAD = "fp"

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            token=token,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

    model_pipeline["tokenizer"] = tokenizer
    model_pipeline["model"] = model

    print(
        f"[OK] Model załadowany ({'4-bit' if LOAD == '4bit' else 'FP/BF16'}) i gotowy."
    )
    yield

    print("[SHUTDOWN] Sprzątanie...")
    model_pipeline.clear()


app.router.lifespan_context = lifespan


@app.get("/healthz")
async def healthz():
    ok = ("model" in model_pipeline) and ("tokenizer" in model_pipeline)
    return {"ok": ok, "model": MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
async def generate_response(req: PromptRequest):
    # 0) sanity-check: czy model/tokenizer są załadowane
    if "model" not in model_pipeline or "tokenizer" not in model_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Model nie jest gotowy lub wystąpił błąd podczas ładowania.",
        )

    start = time.perf_counter()
    tok = model_pipeline["tokenizer"]
    mdl = model_pipeline["model"]

    # 1) Zbuduj wiadomości (DeepSeek-R1 bez system promptu)
    user_text = (req.prompt or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Pusty prompt.")

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    if MODEL_NAME.lower().startswith("deepseek-ai/deepseek-r1-distill-"):
        messages = [{"role": "user", "content": user_text}]

    # 2) Render czatu -> preferuj chat_template; fallback: ręczny prompt
    input_ids = None
    try:
        # apply_chat_template zwraca TENSOR (gdy return_tensors="pt")
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(mdl.device)
    except Exception as e:
        print(f"[WARN] Brak/nieudany chat_template ({e}). Fallback na prosty prompt.")
        # Minimalny, neutralny format zgodny z większością nowszych modeli
        prompt_text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        enc = tok(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(mdl.device)  # WYCIĄGNIJ TENSOR

    # 3) Ustal eos/pad
    # Spróbuj użyć <|eot_id|> jeśli istnieje i nie mapuje się na unk
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and tok.unk_token_id is not None and eot_id != tok.unk_token_id:
        eos_id = eot_id
    else:
        eos_id = tok.eos_token_id

    # Ustaw pad_token jeśli brak — najsensowniej na EOS
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    pad_id = tok.pad_token_id

    max_new = int(MAX_TOKENS) if MAX_TOKENS and int(MAX_TOKENS) > 0 else 512

    # 4) Generacja (bez attention_mask też zadziała, ale można dodać jeśli masz w enc)
    pre_len = input_ids.shape[1]
    with torch.no_grad():
        out_ids = mdl.generate(
            input_ids,
            max_new_tokens=max_new,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_k=TOP_K,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    new_tokens = out_ids[0, pre_len:]
    response_text = tok.decode(new_tokens, skip_special_tokens=True).strip()

    ms = round((time.perf_counter() - start) * 1000)
    return {
        "message_id": str(uuid.uuid4()),
        "messages": [{"id": str(uuid.uuid4()), "content": response_text}],
        "statistics": {"processingTimeMs": ms},
    }