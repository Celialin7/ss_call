# Call Coverage Analysis - Re-Onboarding Guide

## Why this project exists
This project checks whether sales calls cover all required script points (compliance-style coverage checking) across **Cantonese (CAN)**, **Mandarin (MAN)**, and **English (ENG)** flows.

It reads transcript files, compares call content to required discussion points, and outputs detailed results for review and governance reporting.

---

## Quick mental model (30 seconds)
- `run_batch_analysis.py` = the main orchestrator for real runs
- `improved_call_coverage_checker.py` = Cantonese analyzer
- `improved_call_coverage_checker_M.py` = Mandarin + SBERT analyzer (also supports ENG path); **与 `run_batch_analysis.py` 同目录**
- `dynamic_script_analysis.py` = generates dynamic term weights and optional script embeddings
- `config.py` = all important paths and run settings
- `dictionaries.py` = synonyms, stopwords, term importance data loading

---

## Project file map (what each `.py` does)

| File | Required? | Main role | When you use it |
|---|---|---|---|
| `run_batch_analysis.py` | Yes (batch mode) | End-to-end batch runner: queue files, pick language analyzer, export individual + summarized reports | Normal production/batch run |
| `config.py` | Yes | Central config for folders, script path, output/log locations, language metric weights, system-audio switch | Before every environment setup |
| `dictionaries.py` | Yes | Core lexical resources: synonyms, stopwords, error patterns, and dynamic term-importance loader from CSV.gz | Always (imported by analyzers) |
| `dynamic_script_analysis.py` | Recommended | Builds dynamic term importance data and optional SBERT script embeddings from `Scripts.xlsx` | Run when scripts change |
| `improved_call_coverage_checker.py` | Yes (for CAN) | Cantonese checker with weighted multi-metric scoring and grouping logic | CAN calls / direct single-file test |
| `improved_call_coverage_checker_M.py` | Yes (for MAN/ENG) | SBERT-driven checker；与批处理脚本同目录 | MAN or ENG calls in batch |
| `english_resource.py` | Required for ENG quality | English normalization, synonym expansion, negation handling, fallback resources | ENG calls |
| `test_eng_integration.py` | Optional but useful | Smoke test for English resource + config + Mandarin analyzer ENG branch | After dependency/environment changes |
| `generate_presentation_diagrams.py` | Optional | Generates `Presentation_Diagrams.xlsx` docs | Documentation/presentation support |
| `improved_call_coverage_checker_backup_*.py` and `Code backup/*` | No | Historical backups | Reference only, not active runtime |

---

## Data files you need

### A) For batch runs (`run_batch_analysis.py`)
- `converted_text/` folder with transcript `.csv` files
- `file_mapping.xlsx` with at least:
  - `Sample No`
  - `Product Name`
- Recommended for mixed call types / SQCCB / Sales Leader Confirmation:
  - `Call Type` (`Sales Call` / `SQCCB` / `Sales Leader Confirmation`)
- `Scripts.xlsx` containing language/product script sheets (used by dynamic analysis + runtime lookup)

#### `file_mapping.xlsx` concise reference
- Referenced in:
  - `config.py`: path config via `FILE_MAPPING_PATH`
  - `run_batch_analysis.py`: mapping lookup (`Sample No` + optional `Call Type` -> row)
- Main purpose:
  - map each call to a product
  - build target script sheet by call type:
    - `Sales Call` -> `{Product Name}_{Language}`
    - `SQCCB` -> `SQCCB*_{Language}` (via `Product Name`, fallback `SQCCB_{Language}`)
    - `Sales Leader Confirmation` -> `SalesLeader*_{Language}` (via `Product Name`, fallback `SalesLeader_{Language}`)
  - feed `Sample No` / `Product Name` into summary report
- When you modify mapping:
  - keep exact columns: `Sample No`, `Product Name`
  - ensure each file's `Sample No` exists in mapping
  - if one sample has multiple call types, add `Call Type` to disambiguate rows
  - for SQCCB / Sales Leader multi-version, use `Product Name` with prefix style names (e.g. `SQCCB2`, `SalesLeaderV1`)
  - ensure final sheet names (`{Product Name}_{Language}`) exist in `Scripts.xlsx`

#### Audio/transcript naming format (important)
- Runtime parser: `run_batch_analysis.py` -> `parse_filename_for_task_info()`
- Required filename pattern (after removing extension):
  - `<SampleNo>_<any_middle_text>_<LanguageSuffix>`
  - Language suffix must be:
    - `C` -> `CAN`
    - `M` -> `MAN`
    - `E` -> `ENG`
- Extracted from filename:
  - `sample_no`: first segment
  - `language`: last segment (`C/M/E`)
  - `product_name`: resolved via `file_mapping.xlsx` (by `Sample No`, and by `Call Type` when provided)
  - `call_type` (case-insensitive filename detection):
    - contains `SQCCB` -> `SQCCB`
    - contains `salesleader` / `sales leader` / `saleleader` -> `Sales Leader Confirmation`
    - otherwise -> `Sales Call`
  - `script_sheet`: generated as `{product_name}_{language}` based on the resolved call type
- Examples:
  - `12345_anything_C.csv` -> SampleNo `12345`, Language `CAN`
  - `888_xxx_M.csv` -> SampleNo `888`, Language `MAN`
  - `999_demo_E.csv` -> SampleNo `999`, Language `ENG`

### B) For direct single-file checker run (Cantonese script)
An Excel file with two sheets:
- `Call_Text` with columns: `File`, `Time`, `speaker_role`, `Text`
- `Script` with columns: `Required_Discussion_Point`, `Standard_Script`

---

## Setup checklist (first thing after returning to project)

1. **Create/activate virtual env**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install base packages**
   ```bash
   pip install pandas pycantonese jieba openpyxl scikit-learn numpy sentence-transformers
   ```

3. **(Recommended for ENG flow) install optional packages**
   ```bash
   pip install gensim nltk contractions flashtext
   ```

4. **Verify config paths in `config.py`**
   - `RUN_PROFILE` (1 = INV / 2 = INS) — this is the only switch you should change per run
   - `CONVERTED_TEXT_FOLDER`
   - `FILE_MAPPING_PATH`
   - `SCRIPT_FILE_PATH` (auto-selected by `RUN_PROFILE`)
   - `OUTPUT_FOLDER` (auto-selected by `RUN_PROFILE`: `./output/INV` or `./output/INS`)
   - `LOG_FILE_PATH`
   - `SBERT_MODEL_PATH`
   - `SCRIPT_EMBEDDINGS_PATH` (auto-selected by `RUN_PROFILE`)
   - `TERM_IMPORTANCE_CSV` (auto-selected by `RUN_PROFILE`)

---

## Process flow (step-by-step)

### Flow 1 (production flow): dynamic refresh + batch run
Use this as your default real workflow.

1. **Prepare inputs**
   - Update/confirm profile-specific script workbook:
     - `RUN_PROFILE=1` -> `Scripts.xlsx` (investment)
     - `RUN_PROFILE=2` -> `Scripts_ins.xlsx` (insurance)
   - Ensure `file_mapping.xlsx` and `converted_text/*.csv` are ready

2. **If scripts changed, refresh dynamic resources first**
   ```bash
   python dynamic_script_analysis.py
   ```
   This updates dynamic term importance (and SBERT script resources) used at runtime.  
   For MAN/ENG, generated script embeddings use `clean_text` keys to match runtime lookup semantics.
   Profile isolation is automatic:
   - `RUN_PROFILE=1` -> `term_importance_inv.csv.gz`, `script_embeddings_inv.pkl`
   - `RUN_PROFILE=2` -> `term_importance_ins.csv.gz`, `script_embeddings_ins.pkl`

3. **Run unified batch entrypoint**
   ```bash
   python run_batch_analysis.py
   ```

4. **Language routing inside batch run (important)**
   - `CAN` files -> `improved_call_coverage_checker.py`
   - `MAN` files -> `improved_call_coverage_checker_M.py`（与 `run_batch_analysis.py` 同目录）
   - `ENG` files -> 同上（ENG 分支）

5. **Review outputs**
   - Per-call analysis files in `output/`
   - Consolidated `Summarized_Analysis_Report.xlsx`
   - Batch log file (`LOG_FILE_PATH`)

6. **Optional: summary-only refresh**
   ```bash
   python run_batch_analysis.py --summarize-only
   ```

### Flow 2 (debug flow): single-file validator
This is not the normal production path. Use it only for quick debugging/validation:
- test one call quickly before running full batch
- isolate scoring behavior for a single script/call pair
- verify environment after dependency/model changes

Current direct single-file script is the Cantonese standalone path (`improved_call_coverage_checker.py` with `Call_Text` + `Script` sheets).  
For Mandarin/English, normal practice is still through `run_batch_analysis.py`.

### Decision rule (which flow to use)
- **Daily/official run**: Flow 1
- **Troubleshooting one specific case**: Flow 2

---

## Output artifacts (what to expect)
- **Coverage result file(s):** covered/not-covered by required point, with weighted score breakdown
- **Grouped call data:** grouped speaker segments used for matching traceability
- **Batch summarized report:** cross-call aggregation (coverage, duration, counts, outlier signals)
- **Log file:** run status, errors, timing

---

## Matching & scoring logic (zero-knowledge quick pickup)

### 1) Matching is done on grouped dialogue, not raw row-by-row only
- Raw transcript rows are merged into same-speaker context groups (3-pass grouping).
- This gives better semantic context than using only short ASR fragments.
- Sentence-level mapping still exists in outputs for traceability back to original rows.

### 2) Two matching lines are evaluated for each discussion point
For each point vs each dialogue group, the engine compares:
- **Holistic match**: entire point script (all text combined)
- **Granular match**: script variations/split fragments (sentence/segment level)

Then it chooses the better score path (Holistic or Granular) as the effective match for that group.

### 3) Multi-variation script parsing affects matching quality directly
`Standard_Script` is auto-split by delimiters/version markers (for example punctuation, newline, `Version A/B`, `版本A/B`), then very short fragments are merged.  
So script writing style directly impacts what the model can match.

### 4) Language-specific scoring engines
- **CAN (`improved_call_coverage_checker.py`)**
  - Weighted score is a fixed linear mix of four metrics (see **§7**); TF-IDF cosine is the largest term.
  - Optional **pattern enhancement** (dates/numbers for configured points) can raise the score used for `Covered` vs the raw weighted mix.
- **MAN/ENG (`improved_call_coverage_checker_M.py`，与批处理同目录）**
  - Weighted score is a linear mix of four metrics; default weights come from `config.py` → `LANGUAGE_SIMILARITY_WEIGHTS` (see **§7**).
  - Optional **pattern enhancement** is applied in the pairwise path before comparing to the coverage threshold.
  - Embedding cache keys are normalized to `clean_text` (`preprocess_text(..., mode='comparison')`).
  - Legacy embedding pickle keys are normalized at load time for backward compatibility.

### 5) Script preparation guidance (very important)
To maximize match quality when preparing `Scripts.xlsx`:
- keep one discussion point focused on one compliance intent
- avoid overly long mixed-intent paragraphs in one cell
- include realistic paraphrases/variants in `Standard_Script` (not only one formal sentence)
- keep critical numbers/percentages/terms explicit (e.g., rates, price, tenor, IDs)
- use consistent terminology with actual sales wording; synonyms help but cannot fix wrong intent definition
- if scripts changed materially, rerun `dynamic_script_analysis.py` before batch run

### 6) Coverage decision
- For each required point, the engine picks the **best dialogue group** (highest score after Holistic vs Granular choice and any pattern enhancement).
- **`Covered`**: that best score **≥** the analyzer threshold (CAN default **0.3** in `check_coverage`; MAN/ENG default **0.4**).
- **`Not Covered`**: best score **<** threshold.
- **CAN nuance**: the value compared to the threshold is the **enhanced** score when pattern rules apply (`Weighted_Score` in the row reflects enhancement; `Original_Score` / `Enhancement_Boost` split it for diagnostics).

---

### 7) Weighted score — explicit formulas and one-line metric definitions

All sub-scores below are treated as real numbers in **[0, 1]** before weighting. The **weighted_score** is their weighted sum; it is **not** the same as the `Overlapping_Keywords` column (that column is computed **after** the winner is chosen and is **not** added into the score).

#### Cantonese (CAN) — fixed weights in code

Let **script** be the script-side text (holistic or one variation) and **call** be the grouped dialogue text. After comparison-mode preprocessing:

| Symbol | One-line meaning |
| --- | --- |
| **tfidf_cosine** | Cosine similarity between **character n-gram (2–4)** TF-IDF vectors for script vs call, with extra column weights from `current_weights` / tiered rules. |
| **expanded_similarity** | Overlap of **synonym-expanded** token sets: \|E_script ∩ E_call\| / max(\|E_script\|, \|E_call\|). |
| **rouge_l** | Token-level **ROUGE-L** F-score (longest common subsequence style) between script and call. |
| **keyword_coverage** | Share of “script tokens that are also in `dictionaries.important_keywords`” that also appear in call tokens: \|H ∩ K\| / \|K\| with K = (script tokens ∩ `important_keywords`); **0** if K is empty. |

**Formula (CAN):**

`weighted_score = 0.55 * tfidf_cosine + 0.15 * expanded_similarity + 0.20 * rouge_l + 0.10 * keyword_coverage`

Then **pattern enhancement** may add a small boost (capped at 1.0); the **threshold** is applied to that post-enhancement score for `Covered`.

#### Mandarin / English (MAN/ENG) — weights from `config.py`

| Symbol | One-line meaning |
| --- | --- |
| **semantic_score** | Cosine similarity between **Sentence-BERT** embeddings of preprocessed script vs call (with optional precomputed embedding caches). |
| **expanded_similarity** | Overlap of synonym-expanded token sets: \|E_script ∩ E_call\| / \|E_script ∪ E_call\| (**union denominator**, unlike CAN). |
| **rouge_l** | Same token-level ROUGE-L idea as CAN, on preprocessed script vs call. |
| **keyword_coverage** | Same definition as CAN: uses **only** `dictionaries.important_keywords` to form K from script tokens (not the dynamic top-term list). |

**Formula (MAN/ENG):** weights are read from `LANGUAGE_SIMILARITY_WEIGHTS` in `config.py` (defaults: SBERT **0.60**, expanded_overlap **0.15**, rouge_l **0.15**, keyword_coverage **0.10**). For example:

`weighted_score = w_SBERT * semantic_score + w_exp * expanded_similarity + w_rouge * rouge_l + w_kw * keyword_coverage`

**Note:** `LANGUAGE_SIMILARITY_WEIGHTS['ENG']` also defines a `fuzzy_similarity` slot, but the current SBERT `calculate_semantic_similarity` implementation does **not** add a separate fuzzy term into this sum.

---

### 8) Dynamic top terms (`dynamic_script_analysis.py`) vs `keyword_coverage`

- **`dynamic_script_analysis.py`** builds per-sheet, per-discussion-point **top terms** and writes them to **`term_importance.csv.gz`** (see `config.TERM_IMPORTANCE_DIR` / `TERM_IMPORTANCE_CSV`). `dictionaries.py` loads this file into an internal structure and exposes **`get_product_weights`** / **`get_point_specific_weights`**.
- **They are not merged into `keyword_coverage`.** `keyword_coverage` only uses **`dictionaries.important_keywords`** to define K. If K is often empty or tiny because `important_keywords` was never expanded, **`keyword_coverage` contributes little or nothing** even when dynamic top terms exist.
- **Where top terms *are* used today (CAN path):** for Cantonese, point/product-specific weights from that CSV feed **`current_weights`** and influence **TF-IDF feature weighting** (tiered weighting on char n-grams) during similarity — i.e. they shape **tfidf_cosine**, not the separate `keyword_coverage` ratio.
- **MAN/ENG main batch path:** `check_coverage` uses **`compute_pairwise_matches`**, which calls **`calculate_semantic_similarity`** without swapping in per-point weights; **`current_weights`** is still loaded at startup but **does not** enter the SBERT + ROUGE + expanded + keyword_coverage formula in that path. So dynamic CSV terms **do not** currently alter the MAN/ENG weighted score components the way they alter CAN TF-IDF.

---

## System audio switch (`INCLUDE_SYSTEM_AUDIO_IN_ANALYSIS`)

### What this switch controls
- Config location: `config.py`
- Passed from `run_batch_analysis.py` into language analyzers.
- Controls whether detected system recording audio participates in analysis grouping/scoring.

### Recommended defaults
- **Default recommendation: OFF (`False`)**
  - best for compliance checks focused on human agent-customer dialogue
  - reduces inflated matches from repeated system prompts

### When to turn ON (`True`)
- your business definition counts system prompts as valid evidence
- you want maximum recall and do not mind more false-positive risk
- you are analyzing call-flow completeness, not only human sales behavior

### Trade-offs
- **OFF**
  - Pros: cleaner human-behavior signal, lower noise
  - Cons: may miss points spoken only by system audio
- **ON**
  - Pros: captures full call audio content including prompts
  - Cons: can overestimate coverage when system prompts repeat script-like wording

---

## Common errors and fast fixes
- `ModuleNotFoundError`
  - Reinstall missing package in your active venv
- `FileNotFoundError`
  - Recheck path constants in `config.py`
- Sheet/column mismatch
  - Validate exact sheet names and required columns
- Empty task queue in batch
  - Check `converted_text/` has valid `.csv` files and mapping matches `Sample No`
- SBERT/model errors
  - Verify `SBERT_MODEL_PATH` exists and dependencies installed

---

## What to edit most often
- `config.py` for paths, output folder, and language weights
- `Scripts.xlsx` content and then rerun `dynamic_script_analysis.py`
- `dictionaries.py` only when lexical/synonym/stopword logic needs manual tuning

---

## Minimal command cheatsheet
```bash
# 1) activate env
source .venv/bin/activate

# 2) refresh dynamic script resources
python dynamic_script_analysis.py

# 3) run full batch
python run_batch_analysis.py

# 4) summarize only
python run_batch_analysis.py --summarize-only
```

---

## To-Do List

### To-Do 1: Evaluate controlled script expansion strategy

**Key conclusion**
- Benefits from larger sample size in this project mainly come from **script coverage expansion**, not from model retraining.

**Expected benefits**
- **Embedding quality**
  - richer script variations improve hit chance for both `Granular` and `Holistic` matching paths.
- **Dynamic term importance quality**
  - point-level frequency and distinctiveness become more stable, making `point-specific weights` more reliable.
- **Matching stability**
  - coverage for multiple real-world scenarios under the same discussion point becomes more complete, reducing false negatives.

**Trade-offs / risks**
- too many highly similar variants can increase generic matching and raise false-positive risk.
- inference can become slower because more variants must be compared.

### To-Do 2 (Done): Dynamic top-N term selection for dynamic analysis

Implemented in `dynamic_script_analysis.py`:
- point-level top terms are no longer fixed at 12.
- selection rule is now:
  - `N = ceil(unique_token_count * 0.15)`
  - bounded by `min=2`, `max=16`
- this improves adaptiveness for both short and long discussion points while keeping an upper bound on noise and runtime.