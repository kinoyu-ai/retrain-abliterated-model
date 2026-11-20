# Qwen3.8-27B Abliterated + Benign Dolly QLoRA Distillation

[English](./README.md) | **日本語** | [ルート README](../README.ja.md)

無検閲（abliterated）モデル [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)（Gated DeltaNet 48 層 + Full Attention 16 層のハイブリッド 27B）に対して、**通常の日本語 QA のみ**（元モデル [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) の生成回答 500 件）を 4-bit QLoRA で蒸留し、**非拒否（無検閲）性を維持したまま汎用性能・日本語応答品質を改善できるか**を検証した実験結果および学習済み LoRA アダプター（[`adapter/`](./adapter)）です。

## 概要と結論

- **27B モデルをコンシューマ GPU 2 枚（RX 9060 XT 16 GB × 2）で完走**: BF16（51.77 GB）を 1 シャードずつストリーミングして GPU 上で 4-bit NF4 化（16.46 GiB、32/32 層均等分割で GPU0=8.24 GiB / GPU1=8.24 GiB）し、ピーク VRAM **GPU0=9.60 GiB / GPU1=9.49 GiB**・所要 **49.6 分** で 500 件 × 1 epoch の QLoRA 学習を完了しました。
- **汎用知識・日本語 QA が向上し、他タスクも劣化なし**: 計 **8,873 問** の汎用・コーディング・算数・日本語評価において、`MMLU`（81.12% → **81.91%**, +0.78 pt）・`ARC-Challenge`（96.42% → **96.67%**, +0.26 pt）・`JCommonsenseQA`（95.44% → **95.62%**, +0.18 pt）および日本語 Dolly 評価セット（正解文 3-gram F1 `0.1771` → **`0.1856`** / 教師 `0.1893`、256 トークン上限到達 `44` → **`39`** 件、平均長 `182.71` → **`179.07`** で教師 `179.29` に一致）が向上しました。`HumanEval`（85.98% → 85.37%）・`MGSM-ja`（70.80% → 70.80%）・`BoolQ`・`WinoGrande` もほぼそのまま維持しています。
- **無検閲（非拒否）性を完全維持**: 計 **1,333 問** の拒否・過拒否ベンチマーク（XSTest 450 / StrongREJECT 313 / AdvBench 520 / XSTest プライバシー 50）で拒否傾向の復活は見られませんでした（StrongREJECT 2 件・AdvBench 2 件の増分検出も、生成文を全件確認した結果すべて「免責前置き付きで回答本体を出力したケース」であり、実質的な新規拒否は **0 件** でした）。

---

## 実験条件

| 項目 | 設定 |
| --- | --- |
| 教師モデル (`teacher`) | `Qwen/Qwen3.8-27B`（`37561816e66b5b9153b1b64c10968c18c1e6f4a0`、応答生成はローカルの `Qwen3.8-27B-UD-Q5_K_XL.gguf` via `llama-server` を使用） |
| 生徒モデル (`student`) | `huihui-ai/Huihui-Qwen3.8-27B-abliterated`（`739e3c5b89849f6c238ce1e5b70008612ae42cdd`、layers 18〜51 ablated、テキスト部 64 層を NF4 4-bit 化） |
| 学習データ | [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja)（CC BY-SA 3.0）の 5 カテゴリから抽出し、教師が `enable_thinking=False` / greedy / max 256 tokens で EOS まで生成できた非空回答 **500 件**（最小 1 / 最大 253 / 平均 121.03 トークン） |
| 学習データの内訳 | `closed_qa`: 148 / `information_extraction`: 146 / `summarization`: 98 / `classification`: 68 / `open_qa`: 40 |
| LoRA 設定 | `r=8`, `lora_alpha=16`, `lora_dropout=0.0`, `bias="none"`, `target_modules=["o_proj", "out_proj", "down_proj"]`（学習可能パラメータ `17,301,504 / 26,913,299,968 = 0.0643%`） |
| 最適化・学習 | `AdamW` (`lr=1e-4`), 勾配累積 `8` サンプル（計 `63` ステップ / `1` エポック）, 勾配クリッピング `max_norm=1.0`, NF4 4-bit + `bfloat16` 計算, `gradient_checkpointing(use_reentrant=False)`, `seed=42`, 損失対象は回答トークン + EOS のみ（`token_loss`: 初回 `0.3446` → 最初 5 回平均 `0.2397` → 最後 5 回平均 `0.1570`） |
| 実行環境 | AMD Radeon RX 9060 XT (16 GB) × 2（32/32 層均等分割）, PyTorch 2.13.0 + ROCm 10.0, Transformers 5.23.0, PEFT 0.21.3, bitsandbytes 0.50.2 |

### LoRA・学習ハイパーパラメータ詳細（[`adapter/adapter_config.json`](./adapter/adapter_config.json)）

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
  "target_modules": ["o_proj", "out_proj", "down_proj"]
}
```

- LFM3-1.7B 実験の「出力射影層のみに LoRA を当てる設計（`o_proj`, `out_proj`, `w2`）」をそのまま Qwen3.8 のハイブリッド層名に対応させ、Full Attention (`o_proj`)・Gated DeltaNet (`out_proj`)・MLP (`down_proj`) の出力射影層のみを学習対象としました。

---

## ベンチマーク結果（無検閲 `student` vs LoRA 学習後 `trained`）

単一の 4-bit モデルロード上で `disable_adapter()` を切り替えて同一条件（`enable_thinking=False`、greedy / 0-shot logprob、`seed=42`）で評価しました（[`results/bench-qwen38-metrics.json`](./results/bench-qwen38-metrics.json) / [`results/metrics.json`](./results/metrics.json)）。

### 1. 汎用・コーディング・算数・日本語（7 ベンチマーク・8,773 問）

| カテゴリ | ベンチマーク | 評価形式 | 問題数 | `student` (学習前) | `trained` (学習後) | `trained - student` |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TEXT (総合知識) | **MMLU** (val, 57科目) | 0-shot 4択 logprob | 1,531 | 81.12% (1242/1531) | **81.91%** (1254/1531) | **+0.78 pt** |
| TEXT (科学推論) | **ARC-Challenge** (test) | 0-shot 4択 logprob | 1,172 | 96.42% (1130/1172) | **96.67%** (1133/1172) | **+0.26 pt** |
| 日本語 (常識) | **JCommonsenseQA** (valid v1.3) | 0-shot 5択 logprob | 1,119 | 95.44% (1068/1119) | **95.62%** (1070/1119) | **+0.18 pt** |
| TEXT (読解) | **BoolQ** (val) | 0-shot 2択 logprob | 3,270 | 90.40% (2956/3270) | 90.31% (2953/3270) | -0.09 pt |
| TEXT (常識) | **WinoGrande** (val, xl) | 0-shot 2択 logprob | 1,267 | 80.90% (1025/1267) | 80.74% (1023/1267) | -0.16 pt |
| コーディング | **HumanEval** | Pass@1 (greedy, 384 tok) | 164 | 85.98% (141/164) | 85.37% (140/164) | -0.61 pt (-1 問) |
| 日本語 (算数) | **MGSM-ja** (test) | 0-shot 生成 (320 tok) | 250 | 70.80% (177/250) | 70.80% (177/250) | ±0.00 pt |

### 2. 日本語 Dolly 評価セット（100 問, max 256 tokens）

| モデル | 平均生成トークン数 | 生成上限到達 (256 tok) | 正解文文字 3-gram F1 | 教師回答文字 3-gram F1 |
| --- | ---: | ---: | ---: | ---: |
| `teacher` (元モデル) | 179.29 | 43 / 100 | 0.1893 | 1.0000 |
| `student` (学習前) | 182.71 | 44 / 100 | 0.1771 | 0.5777 |
| `trained` (学習後) | **179.07** | **39 / 100** | **0.1856** | 0.5728 |

### 3. 安全性・拒否傾向（計 1,333 問）

#### XSTest 全 450 問 / StrongREJECT 313 問 / AdvBench 520 問（max 96 tokens）

| ベンチマーク | 性質 | 問題数 | `student` 検出率 | `trained` 検出率 | 備考 |
| --- | --- | ---: | ---: | ---: | --- |
| **XSTest Safe** | 過拒否テスト | 250 | 0.80% (2) | **0.80%** (2) | 完全維持 |
| **XSTest Unsafe** | 有害プロンプト | 200 | 6.00% (12) | 7.00% (14 / policy 13) | ほぼ維持 |
| **StrongREJECT** | 有害指示 (6領域) | 313 | 0.00% (0) | 0.64% (2)* | *増分 2 件とも免責前置き付きで回答継続 |
| **AdvBench** | 有害指示 | 520 | 1.73% (9) | 2.12% (11)* | *増分 2 件とも教育目的前置き付きで回答継続 |

#### XSTest プライバシー 50 問（max 256 tokens, ヒューリスティック集計）

- `privacy_public`（25 問）：`teacher` / `student` / `trained` いずれも過剰拒否 **0 / 25**
- `contrast_privacy`（25 問）：
  - `teacher`：ポリシー拒否 `21` / 未所持 `2` / 応答 `2`
  - `student`：ポリシー拒否 `14` / 未所持 `0` / 応答 `11`
  - `trained`：ポリシー拒否 **`12`** / 未所持 `1` / 応答 **`12`**

---

## 使い方（LoRA アダプターの読み込み）

ベースモデル [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) に [`./adapter`](./adapter) を適用して推論できます（BF16 または 4-bit NF4）。

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_id = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
    dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, "./qwen38-27b/adapter")
model.eval()
```

---

## ファイル構成

- [`README.md`](./README.md) — 英語版 README
- [`README.ja.md`](./README.ja.md) — 日本語版 README（このファイル）
- [`adapter/`](./adapter) — 学習済み LoRA アダプター（`adapter_model.safetensors` 約 66.0 MB、`adapter_config.json`、`README.md`）
- [`results/`](./results) — ベンチマーク集計 JSON および学習ログ（`bench-qwen38-metrics.json`, `metrics.json`, `train-log.jsonl`）

## ライセンス

- ベースモデル・教師モデル（`Qwen/Qwen3.8-27B` / `huihui-ai/Huihui-Qwen3.8-27B-abliterated`）および派生 LoRA アダプターは **Apache License 2.0** に従います。
- 学習データには [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja)（CC BY-SA 3.0）を使用しています。
