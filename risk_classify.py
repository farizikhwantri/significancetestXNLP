#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def read_prompt_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()

def format_prompt(system_prompt: str, title: str, abstract: str, tokenizer=None) -> str:
    # Fill placeholders
    filled = system_prompt.replace("{{title}}", title.strip()).replace("{{abstract}}", abstract.strip())

    # If the tokenizer has a chat template, wrap as system+user for better adherence
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": filled.split("Input:")[0].strip()},
            {"role": "user", "content": f"Input:\n- Title: {title.strip()}\n- Abstract: {abstract.strip()}\n\nReturn only the Output JSON."},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Fallback: plain prompt
    return f"{filled}\n\nReturn only the Output JSON."

def extract_first_json(text: str) -> str:
    # Try strict JSON block via stack-based extraction
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found.")
    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start : i + 1]
    raise ValueError("No balanced JSON object found.")

def main():
    parser = argparse.ArgumentParser(description="Classify a paper using a local Hugging Face model.")
    parser.add_argument("--model", required=True, help="Local model path or model ID (downloaded beforehand).")
    parser.add_argument("--title", required=True, help="Paper title.")
    parser.add_argument("--abstract", required=True, help="Paper abstract.")
    parser.add_argument("--prompt-file", default=str(Path(__file__).resolve().parents[1] / "config" / "risk_prompt.txt"),
                        help="Path to risk_prompt.txt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization (bitsandbytes).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for custom models.")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash_attention_2 if available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    # Device and dtype
    device_map = "auto"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Attn impl
    # attn_impl = None if args.no-flash_attn else "flash_attention_2"
    # if attn_impl == "flash_attention_2":
    #     try:
    #         import flash_attn  # noqa: F401
    #     except Exception:
    #         attn_impl = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["torch_dtype"] = dtype

    # if attn_impl is not None:
    #     model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    prompt_template = read_prompt_file(Path(args.prompt_file))
    prompt = format_prompt(prompt_template, args.title, args.abstract, tokenizer)

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**input_ids, **gen_kwargs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant continuation if chat template included the prompt
    # Heuristic: get the substring after the prompt
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()

    try:
        json_text = extract_first_json(decoded)
        result = json.loads(json_text)
    except Exception:
        # Fallback: try to coerce common trailing commas/JSON5-like artifacts
        # Remove control prefix/suffix lines and re-attempt naive curly extraction
        cleaned = re.sub(r"^[^\{\[]*", "", decoded, flags=re.DOTALL).strip()
        json_text = extract_first_json(cleaned)
        result = json.loads(json_text)

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()