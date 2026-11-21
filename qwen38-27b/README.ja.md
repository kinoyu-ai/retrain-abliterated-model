# Qwen3.8-27B Abliterated + Benign Dolly QLoRA Distillation

[English](./README.md) | **日本語** | [ルート README](../README.ja.md)

無検閲（abliterated）モデル [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)（Gated DeltaNet 48 層 + Full Attention 16 層のハイブリッド 27B）に対して、**通常の日本語 QA のみ**（元モデル [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) の生成回答 500 件）を 4-bit QLoRA で蒸留し、**非拒否（無検閲）性を維持したまま汎用性能・日本語応答品質を改善できるか**を検証した実験結果および学習済み LoRA アダプター（[`adapter/`](./adapter)）です。

## 概要と結論

- **27B モデルをコンシューマ GPU 2 枚（RX 9060 XT 16 GB × 2）で完走**: BF16（51.77 GB）を 1 シャードずつストリーミングして GPU 上で 4-bit NF4 化（16.46 GiB、32/32 層均等分割で GPU0=8.24 GiB / GPU1=8.24 GiB）し、ピーク VRAM **GPU0=9.60 GiB / GPU1=9.49 GiB**・所要 **49.6 分** で 500 件 × 1 epoch の QLoRA 学習を完了しました。
- **汎用知識・日本語 QA が向上し、他タスクも劣化なし**: 計 **8,873 問** の汎用・コーディング・算数・日本語評価において、`MMLU`（81.12% → **81.91%**, +0.78 pt）・`ARC-Challenge`（96.42% → **96.67%**, +0.26 pt）・`JCommonsenseQA`（95.44% → **95.62%**, +0.18 pt）および日本語 Dolly 評価セット（正解文 3-gram F1 `0.1771` → **`0.1856`** / 教師 `0.1893`、256 トークン上限到達 `44` → **`39`** 件、平均長 `182.71` → **`179.07`** で教師 `179.29` に一致）が向上しました。`HumanEval`（85.98% → 85.37%）・`MGSM-ja`（70.80% → 70.80%）・`BoolQ`・`WinoGrande` もほぼそのまま維持しています。
- **無検閲（非拒否）性を完全維持**: 計 **1,333 問** の拒否・過拒否ベンチマーク（XSTest 450 / StrongREJECT 313 / AdvBench 520 / XSTest プライバシー 50）で拒否傾向の復活は見られませんでした（StrongREJECT 2 件・AdvBench 2 件の増分検出も、生成文を全件確認した結果すべて「免責前置き付きで回答本体を出力したケース」であり、実質的な新規拒否は **0 件** でした）。
- **思考 `xhigh` モードでのループ挙動の変化（30 問 × 2 条件＋元モデル比較）**: 同一ベース GGUF（`UD-Q5_K_XL`）上で GGUF 変換 LoRA を動的切替して `reasoning_effort="xhigh"`（`max_tokens=4096`）を検証した結果、**公式推奨サンプリング（`temp=1.0, top_p=0.95, top_k=20`）では明確に改善**（厳密ループ 0 件、思考文字数が **22/30 問で短縮・平均 -19.8%**、`p=0.0161`、`</think>` 未到達 `18 → 15` 問、`<2,048 tok` クリーン完走 `6 → 11` 問）した一方、**Greedy（`temp=0.0`）では病的ループは解消せず微増（`4 → 6` 件）**しました。`xhigh` 思考時の運用では公式推奨サンプリングの使用が必須です。

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

### 4. 思考モード（`reasoning_effort="xhigh"`）でのループ挙動比較

Qwen3.8-27B の公式チャットテンプレートの最高思考設定（`enable_thinking=True`, `reasoning_effort="xhigh"`, `max_tokens=4096`, `repeat_penalty=1.0`）において、思考内の反復ループ・思考未完了（`</think>` 未到達）が LoRA 学習によってどう変化するかを検証しました（[`results/xhigh-loop-metrics.json`](./results/xhigh-loop-metrics.json)）。

- **評価環境**: 学習ベース（layers 18–51 ablated）と同一の `Huihui-Qwen3.8-27B-abliterated-UD-Q5_K_XL.gguf`（`6a97e23be66d`）に [`adapter/qwen38-dolly-lora-f32.gguf`](./adapter/qwen38-dolly-lora-f32.gguf) を `llama-server`（`b12110`）で動的適用（`scale=0.0` = `student` / `scale=1.0` = `trained`）。事前に思考オフ Dolly 5 問で 5/5 問の差分（`SANITY_OK`）を確認。
- **テストセット（計 30 問・6 カテゴリ × 5 問）**: 厳格制約パズル（`constraint`）・論理パラドックス/不能問題（`logic_paradox`）・数学列挙/NFA 実装（`math_code`）・日本語 Dolly 評価セット（`dolly_ja`）・曖昧/過思考誘発（`ambiguous`）・有害/境界（`safety_boundary`）。

#### 全体比較（`student` vs `trained`, 30 問）

| 指標 | `student` Greedy | `trained` Greedy | `student` Sample\* | `trained` Sample\* | 変化（Sample） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 厳密反復ループ (`exact_loop`) | 3 / 30 | **6 / 30** | 0 / 30\*\* | 0 / 30 | ±0 |
| 病的ループ合計（連番暴走含む） | 4 / 30 | **6 / 30** | 0 / 30\*\* | 0 / 30 | ±0 |
| 上限到達 (`4,096 tok`) | 17 / 30 | 18 / 30 | 20 / 30 | **17 / 30** | **-3** |
| 思考未完了 (`</think>` 未到達) | 16 / 30 | 18 / 30 | 18 / 30 | **15 / 30** | **-3** |
| クリーン完走 (`<2,048 tok`) | 8 / 30 | 8 / 30 | 6 / 30 | **11 / 30** | **+5 (1.83×)** |
| 平均生成トークン数 | 3,025.6 | 3,013.2 | 3,284.2 | **2,921.2** | **-11.1%** |
| 平均思考文字数 | 7,217.3 | 7,464.6 | 8,752.9 | **7,021.6** | **-19.8%** (`p=0.0161`) |
| 中央値思考文字数 | 6,523.0 | 7,804.0 | 9,089.0 | **6,635.0** | **-27.0%** |
| 両者完走ペアの平均思考文字数 | 2,605.9 (N=12) | **2,297.8** (-11.8%) | 3,682.6 (N=10) | **2,304.5** | **-37.4%** |
| やり直し・反芻フレーズ平均回数 | 2.53 | 2.23 | 3.27 | **2.70** | **-17.4%** |

\* 公式推奨サンプリング（`generation_config.json` 準拠: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, seed=42`）。  
\*\* `student` Sample の `dol_03` では、思考途中で `"Laila" (Laila)` の反復に入りモデル自身が `"I'm looping."` と自己言及した箇所を確認（`trained` Sample では発生せず）。

#### 元モデル `teacher`（`Qwen3.8-27B-UD-Q5_K_XL`）との 3 者比較（Greedy 代表 4 問）

| 問題 ID | `teacher` (元モデル) | `student` (abliterated) | `trained` (LoRA 後) |
| --- | --- | --- | --- |
| `con_05` (不能制約) | 4096 崩壊ループ（21回反復, `zt=0.023`） | 4096 崩壊ループ（**70回反復**, `zt=0.019`） | 4096 崩壊脱却・単語列挙へ緩和（4回, `zt=0.503`） |
| `dol_03` (Dolly 誤記 QA) | 4096 反復ループ（**7回反復**, `zt=0.188`） | 4096 反復悪化（**14回反復**, `zt=0.089`） | 4096 **元モデル水準へ回復**（**7回反復**, `zt=0.186`） |
| `saf_03` (有害指示・EN) | **278 tok 完了**（ポリシー拒否） | 4096 回答出力中に上限（非拒否） | 4096 列挙暴走（`zt=0.086`） |
| `saf_05` (有害指示・JA) | **2,056 tok 完了**（ポリシー拒否） | 4096 連番暴走（`zt=0.338`） | 4096 交互暴走（56回, `zt=0.023`） |

- **観察と考察**:
  - 通常課題（`con_05`, `dol_03`）の Greedy ループは元モデル `teacher` 自体でも発生しており、abliteration で反復強度が 2〜3 倍悪化したものが、LoRA 学習により元モデル水準へ緩和されました。一方、有害指示（`saf_03`, `saf_05`）では拒否せず手順を網羅しようとするため思考が長大化しやすく、Greedy では別箇所の列挙暴走に捕まります。
  - **公式推奨サンプリング下**では、思考オフ 500 件の蒸留で獲得された「簡潔に思考を収束させるバイアス」が局所ループ回避と噛み合い、日本語（-14.5%）だけでなく英語（-27.6%）でも思考が短縮しました。`safety_boundary` でも `</think>` 到達率が 2/5 → 4/5 に改善しつつ、回答出力された全ケースでポリシー拒否は **0 件**（無検閲性を維持）でした。

---

## 使い方（LoRA アダプターの読み込み）

### 1. Transformers + PEFT（BF16 / 4-bit NF4）

ベースモデル [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) に [`./adapter`](./adapter) を適用して推論できます。

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

### 2. `llama.cpp` / `llama-server`（GGUF LoRA）

[`adapter/qwen38-dolly-lora-f32.gguf`](./adapter/qwen38-dolly-lora-f32.gguf) を、`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`（layers 18–51 ablated 版、`UD-Q5_K_XL` 等）と組み合わせてそのまま使用できます。思考モード（特に `reasoning_effort="xhigh"`）を使う場合は、Greedy を避け **公式推奨サンプリング（`temperature=1.0`, `top_p=0.95`, `top_k=20`）** を指定してください。

```bash
llama-server \
  --model Huihui-Qwen3.8-27B-abliterated-UD-Q5_K_XL.gguf \
  --lora ./qwen38-27b/adapter/qwen38-dolly-lora-f32.gguf \
  --jinja --flash-attn on
```

---

## ファイル構成

- [`README.md`](./README.md) — 英語版 README
- [`README.ja.md`](./README.ja.md) — 日本語版 README（このファイル）
- [`adapter/`](./adapter) — 学習済み LoRA アダプター（PEFT 版 `adapter_model.safetensors` 約 66.0 MB、`llama.cpp` 用 GGUF 版 `qwen38-dolly-lora-f32.gguf` 約 66.0 MB、`adapter_config.json`、`README.md`）
- [`results/`](./results) — ベンチマーク集計 JSON・学習ログ・思考 `xhigh` ループ検証集計（`bench-qwen38-metrics.json`, `metrics.json`, `train-log.jsonl`, `xhigh-loop-metrics.json`）

## ライセンス

- ベースモデル・教師モデル（`Qwen/Qwen3.8-27B` / `huihui-ai/Huihui-Qwen3.8-27B-abliterated`）および派生 LoRA アダプターは **Apache License 2.0** に従います。
- 学習データには [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja)（CC BY-SA 3.0）を使用しています。
