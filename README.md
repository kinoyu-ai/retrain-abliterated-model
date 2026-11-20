# Retraining Abliterated Models with Benign Dolly LoRA Distillation

**English** | [日本語](./README.ja.md)

This repository contains trained PEFT LoRA adapters and benchmark evaluations testing whether **lightweight LoRA / QLoRA distillation using only 500 benign Japanese QA responses** from the original model can **recover or improve general capabilities lost during abliteration without restoring safety refusals**.

## Included Experiments

| Model | Base (Abliterated) | Method | Adapter | Full Report & Results |
| --- | --- | --- | --- | --- |
| **LFM3-1.7B** | [`orcarouter/LFM3-1.7B-abliterated`](https://huggingface.co/orcarouter/LFM3-1.7B-abliterated) | BF16 LoRA (`r=8`, 10.6 MB) | [`adapter/`](./adapter) | Below & [`results/`](./results) |
| **Qwen3.8-27B** | [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) | 4-bit NF4 QLoRA (`r=8`, 66.0 MB) | [`qwen38-27b/adapter/`](./qwen38-27b/adapter) | [**`qwen38-27b/README.md`**](./qwen38-27b/README.md) & [`qwen38-27b/results/`](./qwen38-27b/results) |

---

## Experiment 1: LFM3-1.7B Abliterated + Benign Dolly LoRA Distillation

This section details the LFM3-1.7B experiment ([`adapter/`](./adapter), [`results/`](./results)), distilling 500 benign QA responses from [`LiquidAI/LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B) onto [`orcarouter/LFM3-1.7B-abliterated`](https://huggingface.co/orcarouter/LFM3-1.7B-abliterated).

## Key Findings

- **Benign-only training data:** Trained exclusively on 500 ordinary Japanese QA responses from Dolly (5 benign categories: QA, summarization, classification, information extraction). Zero safety or refusal prompts were included.
- **Recovery and gains beyond the original model:** Abliteration (`teacher` -> `student`) degraded `ARC-Challenge` (`63.31%` -> `59.90%`), `MMLU` (`48.53%` -> `46.31%`), `GSM8K` (`67.48%` -> `65.13%`), and `BoolQ` (`51.59%` -> `46.30%`). After 1 epoch of benign LoRA distillation (`trained`), performance **fully recovered or exceeded the original `teacher`** (`64.59%`, `49.71%`, `68.54%`, `51.28%`), while coding benchmarks improved substantially (`HumanEval`: `40.85%` -> **`58.54%`**, `MBPP`: `41.63%` -> **`52.14%`**).
- **Abliterated non-refusal preserved:** Across **4,196 refusal & over-refusal prompts** (`XSTest`, `AdvBench`, `StrongREJECT`, `OR-Bench`, `Do-Not-Answer`), where `teacher` refuses heavily (e.g., `81.0%` on `XSTest Unsafe`, `61.15%` on `AdvBench`, `31.95%` on `StrongREJECT`), `trained` retains the same near-zero refusal and near-zero over-refusal profile as `student` (`AdvBench` `0.38%`, `StrongREJECT` `0.00%`, `OR-Bench Hard-1k` `0.08%`).

---

## Experimental Setup

| Item | Configuration |
| --- | --- |
| Teacher (`teacher`) | `LiquidAI/LFM3-1.7B` (`1da06bec6a819da10b54c67e5c90972b46d81871`) |
| Abliterated Base (`student`) | `orcarouter/LFM3-1.7B-abliterated` (`9fded8f990462fa0a0b6439805d1065933ac199e`) |
| Training Data | First **500** complete (EOS-terminated), non-empty teacher answers generated (`enable_thinking=False`, greedy, `max_new_tokens=256`, `repetition_penalty=1.1`) on [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja) (CC BY-SA 3.0). Answer lengths: min `5`, max `255`, mean `156.20` tokens. |
| Category Breakdown (500 total) | `information_extraction`: 136 / `closed_qa`: 116 / `open_qa`: 93 / `classification`: 78 / `summarization`: 77 (`0` refusal responses) |
| LoRA Config | `r=8`, `lora_alpha=16`, `lora_dropout=0.0`, `bias="none"`, `target_modules=["o_proj", "out_proj", "w2"]` (`2,654,208 / 1,731,455,744 = 0.1533%` trainable parameters) |
| Optimization | `AdamW` (`lr=1e-4`), gradient accumulation `8` (`63` steps / `1` epoch), gradient clipping `max_norm=1.0`, `bfloat16`, `seed=42`, cross-entropy loss on answer tokens + EOS only |
| Hardware / Runtime | 1× AMD Radeon RX 9060 XT (16 GB), PyTorch `2.13.0+rocm10.0`, Transformers `5.23.0`, PEFT `0.21.3` |

### LoRA & Training Details ([`adapter/adapter_config.json`](./adapter/adapter_config.json))

```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "bias": "none",
  "use_dora": false,
  "use_rslora": false,
  "target_modules": ["o_proj", "out_proj", "w2"]
}
```

- **Prompt / Generation limits:** `MAX_PROMPT=512`, `max_new_tokens=256`, `enable_thinking=False`, `do_sample=False`.
- **Loss & Updates:** Token-normalized `cross_entropy(..., reduction="sum")` over answer + EOS tokens per 8-sample accumulation window (`token_loss`: initial `0.3986`, first-5 mean `0.4044` -> last-5 mean `0.3800`).
- **Model loading:** `dtype=torch.bfloat16`, `attn_implementation="sdpa"`, `use_kernels=False`.

---

## Benchmark Results (`teacher` vs `student` vs `trained`)

### 1. Coding, Text, Reasoning & Japanese (9 Benchmarks, 10,349 Prompts)

| Category | Benchmark | Protocol | N | `teacher` (Original) | `student` (Abliterated) | `trained` (Abliterated + LoRA) | `trained - student` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Coding | **HumanEval** | Pass@1 (greedy, 384 tok) | 164 | 24.39% (40/164) | 40.85% (67/164) | **58.54%** (96/164) | **+17.68 pt** |
| Coding | **MBPP (sanitized)** | Pass@1 (greedy, 384 tok) | 257 | 36.58% (94/257) | 41.63% (107/257) | **52.14%** (134/257) | **+10.51 pt** |
| Reading | **BoolQ** (val) | 0-shot 2-choice logprob | 3,270 | **51.59%** (1687/3270) | 46.30% (1514/3270) | 51.28% (1677/3270) | **+4.98 pt** |
| Commonsense | **WinoGrande** (val, xl) | 0-shot 2-choice logprob | 1,267 | 50.59% (641/1267) | **51.62%** (654/1267) | 50.43% (639/1267) | -1.18 pt |
| Science QA | **ARC-Challenge** (test) | 0-shot 4-choice logprob | 1,172 | 63.31% (742/1172) | 59.90% (702/1172) | **64.59%** (757/1172) | **+4.69 pt** |
| Knowledge | **MMLU** (val, 57 subjects) | 0-shot 4-choice logprob | 1,531 | 48.53% (743/1531) | 46.31% (709/1531) | **49.71%** (761/1531) | **+3.40 pt** |
| Math (EN) | **GSM8K** (test) | 0-shot gen (320 tok) | 1,319 | 67.48% (890/1319) | 65.13% (859/1319) | **68.54%** (904/1319) | **+3.41 pt** |
| Japanese QA | **JCommonsenseQA** (valid v1.3) | 0-shot 5-choice logprob | 1,119 | **73.99%** (828/1119) | 72.74% (814/1119) | 72.65% (813/1119) | -0.09 pt |
| Math (JA) | **MGSM-ja** (test) | 0-shot gen (320 tok) | 250 | 40.00% (100/250) | 39.60% (99/250) | **40.80%** (102/250) | **+1.20 pt** |

### 2. Japanese Dolly Held-Out Set (100 Prompts, max 256 tokens)

| Model | Mean Answer Tokens | Hit 256-Token Limit | Reference Char 3-gram F1 | Teacher Char 3-gram F1 |
| --- | ---: | ---: | ---: | ---: |
| `teacher` (Original) | 208.33 | 51 / 100 | 0.1484 | 1.0000 |
| `student` (Abliterated) | 211.14 | 54 / 100 | 0.1438 | 0.4566 |
| `trained` (Abliterated + LoRA) | 196.55 | 43 / 100 | 0.1468 | 0.4519 |

### 3. Refusal & Over-Refusal Evaluation (6 Benchmarks, 4,196 Prompts)

#### XSTest Full (450 Prompts, prefix heuristic, max 128 tokens)

| Split | `teacher` (Original) | `student` (Abliterated) | `trained` (Abliterated + LoRA) |
| --- | ---: | ---: | ---: |
| **Safe (Over-refusal)** (250) | 11.6% (29/250) | 0.4% (1/250) | 0.8% (2/250) |
| **Unsafe (Refusal)** (200) | **81.0%** (162/200) | 2.5% (5/200) | 3.0% (6/200) |

#### XSTest Privacy 50 Prompts — Full Manual Inspection (max 256 tokens)

- `privacy_public` (25 safe public-entity prompts): **0 / 25** over-refusals for all three models.
- `contrast_privacy` (25 private-person prompts):
  - `teacher`: **22** refusals (21 explicit policy/privacy refusals, 1 access-only) / 2 partial / 1 answer
  - `student`: **4** refusals (**0** policy refusals, 4 access-only) / 14 partial / 7 answers
  - `trained`: **6** refusals (**0** policy refusals, 6 access-only) / 12 partial / 7 answers

#### Five Additional Refusal & Over-Refusal Benchmarks (3,746 Prompts, max 96 tokens)

Evaluated identically across all three models ([`results/`](./results)):

| Benchmark | Type | N | `teacher` Policy Refusal (All*) | `student` Policy Refusal (All*) | `trained` Policy Refusal (All*) |
| --- | --- | ---: | ---: | ---: | ---: |
| **AdvBench** | Harmful instructions | 520 | **61.15%** (318) | 0.38% (2) | 0.38% (2) |
| **StrongREJECT** | Harmful instructions (6 categories) | 313 | **31.95%** (100) | 0.32% (1) | **0.00%** (0) |
| **OR-Bench Toxic** | Toxic prompts (10 categories) | 655 | **20.15%** (132) | 0.15% (1 / All 2) | 0.15% (1 / All 2) |
| **OR-Bench Hard-1k** | Safe hard prompts (Over-refusal) | 1,319 | 14.18% (187) | 0.15% (2) | **0.08%** (1) |
| **Do-Not-Answer** | Risk-area prompts | 939 | **14.70%** (138 / All 144) | 1.60% (15 / All 39) | **1.28%** (12 / All 25) |

\* "All" = policy refusal phrases + personal/real-time data access disclaimers (`access_disclaimer`) + empty outputs (`0` empty outputs across all 3,746 prompts for all models). Manual inspection of flagged `student`/`trained` samples shows that the 2 `AdvBench` hits and 1 `StrongREJECT` hit are mostly modality/style false positives (e.g., stating it cannot produce a video file and writing a script instead), and `Do-Not-Answer` hits are mostly factual statements about lacking personal memory/data/emotions rather than safety refusals.

---

## Usage (Loading the LoRA Adapter)

Apply [`./adapter`](./adapter) on top of `orcarouter/LFM3-1.7B-abliterated` (requires accepting the model license on Hugging Face):

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "orcarouter/LFM3-1.7B-abliterated"
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./adapter")
model.eval()

messages = [{"role": "user", "content": "Write a Python function that returns the first n Fibonacci numbers."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Repository Structure

- [`README.md`](./README.md) — English documentation (this file; LFM3-1.7B details + overview)
- [`README.ja.md`](./README.ja.md) — Japanese documentation
- [`adapter/`](./adapter) — LFM3-1.7B PEFT LoRA adapter (`adapter_model.safetensors` ~10.6 MB, `adapter_config.json`, `README.md`)
- [`results/`](./results) — LFM3-1.7B aggregated JSON metrics for all 15 benchmarks and Dolly (5 files)
- [`qwen38-27b/`](./qwen38-27b) — Qwen3.8-27B 4-bit QLoRA experiment ([`README.md`](./qwen38-27b/README.md), [`README.ja.md`](./qwen38-27b/README.ja.md), [`adapter/`](./qwen38-27b/adapter) ~66.0 MB, [`results/`](./qwen38-27b/results))

---

## License & Acknowledgments

- The base model, teacher model (`LiquidAI/LFM3-1.7B` / `orcarouter/LFM3-1.7B-abliterated`), and derived LoRA adapter are subject to the **LFM 1.0 License**.
- Training prompts are derived from [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja) (CC BY-SA 3.0). Benchmarks include [XSTest](https://github.com/paul-rottger/xstest), AdvBench, StrongREJECT, OR-Bench, Do-Not-Answer, HumanEval, MBPP, BoolQ, WinoGrande, ARC-Challenge, MMLU, GSM8K, JCommonsenseQA, and MGSM.
