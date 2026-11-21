# Retraining Abliterated Models with Benign Dolly LoRA Distillation

[English](./README.md) | **日本語**

無検閲（abliterated）モデルに対して、**元モデルが通常の日本語 QA に返した回答 500 件のみ**を LoRA / QLoRA で蒸留し、**非拒否（無検閲）性を維持したまま汎用性能を回復・向上できるか**を検証した実験リポジトリおよび学習済み PEFT アダプター集です。

## 収録モデル・実験一覧

| モデル | ベース（無検閲版） | 手法 | アダプター | 詳細レポート・結果 |
| --- | --- | --- | --- | --- |
| **LFM3-1.7B** | [`orcarouter/LFM3-1.7B-abliterated`](https://huggingface.co/orcarouter/LFM3-1.7B-abliterated) | BF16 LoRA (`r=8`, 10.6 MB) | [`adapter/`](./adapter) | 本ページ下部 & [`results/`](./results) |
| **Qwen3.8-27B** | [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) | 4-bit NF4 QLoRA (`r=8`, 66.0 MB PEFT / GGUF) | [`qwen38-27b/adapter/`](./qwen38-27b/adapter) | [**`qwen38-27b/README.ja.md`**](./qwen38-27b/README.ja.md) & [`qwen38-27b/results/`](./qwen38-27b/results)（思考 `xhigh` ループ調査含む） |

---

## 実験 1: LFM3-1.7B Abliterated + Benign Dolly LoRA Distillation

無検閲モデル [`orcarouter/LFM3-1.7B-abliterated`](https://huggingface.co/orcarouter/LFM3-1.7B-abliterated) に対して、元モデル [`LiquidAI/LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B) の通常日本語 QA 回答 500 件を LoRA 蒸留した結果（[`adapter/`](./adapter)・[`results/`](./results)）です。

## 概要と結論

- **通常の質問のみで学習**: 日本語 Dolly（5 カテゴリ：質問応答・要約・分類・情報抽出）の 500 件のみを使用し、安全性・拒否関連のプロンプトは一切学習に含めていません。
- **アブリレーションによる汎用性能低下の回復・超過**: 無検閲化（`teacher` → `student`）で低下していた `ARC-Challenge`（63.31% → 59.90%）・`MMLU`（48.53% → 46.31%）・`GSM8K`（67.48% → 65.13%）・`BoolQ`（51.59% → 46.30%）を、通常質問 500 件の LoRA 蒸留（`trained`）により **元モデル水準まで完全回復または超過**（`64.59%` / `49.71%` / `68.54%` / `51.28%`）し、`HumanEval`（58.54%）・`MBPP`（52.14%）も大幅に向上しました。
- **無検閲（非拒否）性と低過拒否率の維持**: 計 **4,196 問** の拒否・過拒否ベンチマーク（XSTest / AdvBench / StrongREJECT / OR-Bench / Do-Not-Answer）において、元モデル `teacher` が高率に拒否する有害指示（例: `XSTest Unsafe` 81.0%, `AdvBench` 61.15%, `StrongREJECT` 31.95%）に対しても、`trained` は元の無検閲モデル `student` と同等の非拒否傾向を維持し、過拒否率（`OR-Bench Hard-1k` 0.08%）も極小のままでした。

---

## 実験条件

| 項目 | 設定 |
| --- | --- |
| 教師モデル (`teacher`) | `LiquidAI/LFM3-1.7B` (`1da06bec6a819da10b54c67e5c90972b46d81871`) |
| 生徒モデル (`student`) | `orcarouter/LFM3-1.7B-abliterated` (`9fded8f990462fa0a0b6439805d1065933ac199e`) |
| 学習データ | [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja)（CC BY-SA 3.0）の 5 カテゴリから抽出し、教師が `enable_thinking=False` / greedy / max 256 tokens で EOS まで生成できた非空回答 **500 件**（最小 5 / 最大 255 / 平均 156.20 トークン） |
| 学習データの内訳 | `information_extraction`: 136 / `closed_qa`: 116 / `open_qa`: 93 / `classification`: 78 / `summarization`: 77（拒否表現 0 件） |
| LoRA 設定 | `r=8`, `lora_alpha=16`, `lora_dropout=0.0`, `bias="none"`, `target_modules=["o_proj", "out_proj", "w2"]`（学習可能パラメータ `2,654,208 / 1,731,455,744 = 0.1533%`） |
| 最適化・学習 | `AdamW` (`lr=1e-4`), 勾配累積 `8` サンプル（計 `63` ステップ / `1` エポック）, 勾配クリッピング `max_norm=1.0`, `bfloat16`, `seed=42`, 損失対象は回答トークン + EOS のみ |
| 実行環境 | AMD Radeon RX 9060 XT (16 GB) × 1, PyTorch 2.13.0 + ROCm 10.0, Transformers 5.23.0, PEFT 0.21.3 |

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
  "target_modules": ["o_proj", "out_proj", "w2"]
}
```

- **教師生成**: `enable_thinking=False`, `do_sample=False` (greedy), `repetition_penalty=1.1`, `max_new_tokens=256`, プロンプト最大長 `512` トークン
- **学習データ選定**: 生成完了（EOS 到達）かつ空でない教師回答の先頭 `500` 件（最小 `5` / 最大 `255` / 平均 `156.20` トークン）
- **損失・更新**: 各サンプルの回答トークン＋EOS に対して `cross_entropy(..., reduction="sum")` を算出し、8 サンプルごとのターゲットトークン総数で割って更新（`token_loss`：初回 `0.3986` → 最初 5 回平均 `0.4044` → 最後 5 回平均 `0.3800`）
- **モデルロード**: `dtype=torch.bfloat16`, `attn_implementation="sdpa"`, `use_kernels=False`

---

## ベンチマーク結果（元モデル `teacher` vs 無検閲 `student` vs LoRA 学習後 `trained`）

### 1. コーディング・TEXT・推論・日本語（9 ベンチマーク・10,349 問）

| カテゴリ | ベンチマーク | 評価形式 | 問題数 | `teacher` (元モデル) | `student` (無検閲・学習前) | `trained` (学習後) | `trained - student` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| コーディング | **HumanEval** | Pass@1 (greedy, 384 tok) | 164 | 24.39% (40/164) | 40.85% (67/164) | **58.54%** (96/164) | **+17.68 pt** |
| コーディング | **MBPP (sanitized)** | Pass@1 (greedy, 384 tok) | 257 | 36.58% (94/257) | 41.63% (107/257) | **52.14%** (134/257) | **+10.51 pt** |
| TEXT (読解) | **BoolQ** (val) | 0-shot 2択 logprob | 3,270 | **51.59%** (1687/3270) | 46.30% (1514/3270) | 51.28% (1677/3270) | **+4.98 pt** |
| TEXT (常識) | **WinoGrande** (val, xl) | 0-shot 2択 logprob | 1,267 | 50.59% (641/1267) | **51.62%** (654/1267) | 50.43% (639/1267) | -1.18 pt |
| TEXT (科学推論) | **ARC-Challenge** (test) | 0-shot 4択 logprob | 1,172 | 63.31% (742/1172) | 59.90% (702/1172) | **64.59%** (757/1172) | **+4.69 pt** |
| TEXT (総合知識) | **MMLU** (val, 57科目) | 0-shot 4択 logprob | 1,531 | 48.53% (743/1531) | 46.31% (709/1531) | **49.71%** (761/1531) | **+3.40 pt** |
| 数学 (英語) | **GSM8K** (test) | 0-shot 生成 (320 tok) | 1,319 | 67.48% (890/1319) | 65.13% (859/1319) | **68.54%** (904/1319) | **+3.41 pt** |
| 日本語 (常識) | **JCommonsenseQA** (valid v1.3) | 0-shot 5択 logprob | 1,119 | **73.99%** (828/1119) | 72.74% (814/1119) | 72.65% (813/1119) | -0.09 pt |
| 日本語 (算数) | **MGSM-ja** (test) | 0-shot 生成 (320 tok) | 250 | 40.00% (100/250) | 39.60% (99/250) | **40.80%** (102/250) | **+1.20 pt** |

### 2. 日本語 Dolly 評価セット（100 問, max 256 tokens）

| モデル | 平均生成トークン数 | 生成上限到達 (256 tok) | 正解文文字 3-gram F1 | 教師回答文字 3-gram F1 |
| --- | ---: | ---: | ---: | ---: |
| `teacher` (元モデル) | 208.33 | 51 / 100 | 0.1484 | 1.0000 |
| `student` (学習前) | 211.14 | 54 / 100 | 0.1438 | 0.4566 |
| `trained` (学習後) | 196.55 | 43 / 100 | 0.1468 | 0.4519 |

### 3. 安全性・拒否傾向（6 ベンチマーク・計 4,196 問）

#### XSTest 全 450 問（接頭辞ヒューリスティック集計, max 128 tokens）

| 区分 | `teacher` (元モデル) | `student` (学習前) | `trained` (学習後) |
| --- | ---: | ---: | ---: |
| **Safe（過剰拒否）** 250 問 | 11.6% (29/250) | 0.4% (1/250) | 0.8% (2/250) |
| **Unsafe（拒否率）** 200 問 | **81.0%** (162/200) | 2.5% (5/200) | 3.0% (6/200) |

#### XSTest プライバシー 50 問の全件目視分類（max 256 tokens）

- `privacy_public`（25 問・公開情報）：`teacher` / `student` / `trained` いずれも **過剰拒否 0 / 25**。
- `contrast_privacy`（25 問・私的個人情報）：
  - `teacher`：拒否 **22**（うちポリシー拒否 21、未所持のみ 1）/ 部分拒否 2 / 応答 1
  - `student`：拒否 **4**（うちポリシー拒否 **0**、未所持のみ 4）/ 部分拒否 14 / 応答 7
  - `trained`：拒否 **6**（うちポリシー拒否 **0**、未所持のみ 6）/ 部分拒否 12 / 応答 7

#### 追加の拒否・過拒否ベンチマーク 5 種（計 3,746 問, max 96 tokens）

XSTest に加え、有害指示・過拒否・リスク質問を含む 5 つのベンチマーク（合計 3,746 問、XSTest と合わせて **4,196 問**）を 3 モデルすべてに同一条件で実施しました（[`results/`](./results)）。

| ベンチマーク | 性質 | 問題数 | `teacher` ポリシー拒否 (全体*) | `student` ポリシー拒否 (全体*) | `trained` ポリシー拒否 (全体*) |
| --- | --- | ---: | ---: | ---: | ---: |
| **AdvBench** | 有害指示 | 520 | **61.15%** (318) | 0.38% (2) | 0.38% (2) |
| **StrongREJECT** | 有害指示 (6領域) | 313 | **31.95%** (100) | 0.32% (1) | **0.00%** (0) |
| **OR-Bench Toxic** | 有害プロンプト (10領域) | 655 | **20.15%** (132) | 0.15% (1 / 全体 2) | 0.15% (1 / 全体 2) |
| **OR-Bench Hard-1k** | 安全な難問 (過拒否テスト) | 1,319 | 14.18% (187) | 0.15% (2) | **0.08%** (1) |
| **Do-Not-Answer** | リスク領域別質問 | 939 | **14.70%** (138 / 全体 144) | 1.60% (15 / 全体 39) | **1.28%** (12 / 全体 25) |

\*「全体」= ポリシー拒否語句 + 個人情報/リアルタイム情報未所持の定型断り（`access_disclaimer`）+ 空応答（全 3,746 問で全モデルとも空応答は `0` 件）。
※ `AdvBench` の `student`/`trained` 各 2 件や `StrongREJECT` の `student` 1 件は、生成サンプル確認の結果、動画作成不可の前置きや演説文中の表現・共感フレーズによる検出であり、実際の安全拒否はほぼ `0` でした（`trained` の `OR-Bench Toxic` 1 件のみ爆弾作成で明示的拒否を確認）。`Do-Not-Answer` で検出された大半も「AI のため記憶・感情・個人情報を持たない」旨の事実説明であり、有害指示に対する拒否復活は見られませんでした。

---

## 使い方（LoRA アダプターの読み込み）

ベースモデル `orcarouter/LFM3-1.7B-abliterated`（要 Hugging Face でのライセンス同意）に [`./adapter`](./adapter) を適用して推論できます。

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

messages = [{"role": "user", "content": "Pythonでフィボナッチ数列を返す関数を書いてください。"}]
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

## ファイル構成

- [`README.md`](./README.md) — 英語版 README
- [`README.ja.md`](./README.ja.md) — 日本語版 README（このファイル：全体一覧 + LFM3-1.7B 詳細）
- [`adapter/`](./adapter) — LFM3-1.7B 学習済み LoRA アダプター（`adapter_model.safetensors` 約 10.6 MB、`adapter_config.json`、`README.md`）
- [`results/`](./results) — LFM3-1.7B 全 15 ベンチマークおよび Dolly の集計 JSON（5 ファイル）
- [`qwen38-27b/`](./qwen38-27b) — Qwen3.8-27B 4-bit QLoRA 実験一式（[`README.md`](./qwen38-27b/README.md)、[`README.ja.md`](./qwen38-27b/README.ja.md)、[`adapter/`](./qwen38-27b/adapter) 約 66.0 MB、[`results/`](./qwen38-27b/results)）

---

## ライセンス・謝辞

- ベースモデル・教師モデル（`LiquidAI/LFM3-1.7B` / `orcarouter/LFM3-1.7B-abliterated`）の重みおよび派生アダプターの取り扱いは元の **LFM 1.0 License** に従います。
- 学習データには [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja)（CC BY-SA 3.0）、評価には [XSTest](https://github.com/paul-rottger/xstest) ほか各公開ベンチマークを使用しています。
