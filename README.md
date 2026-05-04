# Call Coverage Analysis - Re-Onboarding Guide

## Why this project exists
This project checks whether sales calls cover all required script points (compliance-style coverage checking) across **Cantonese (CAN)**, **Mandarin (MAN)**, and **English (ENG)** flows.

It reads transcript files, compares call content to required discussion points, and outputs detailed results for review and governance reporting.

---

## Quick mental model (30 seconds)
- `run_batch_analysis.py` = the main orchestrator for real runs
- `improved_call_coverage_checker.py` = Cantonese analyzer
- `Mandarin/improved_call_coverage_checker_M.py` = Mandarin + SBERT analyzer (also supports ENG path)
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
| `Mandarin/improved_call_coverage_checker_M.py` | Yes (for MAN/ENG) | SBERT-driven checker used in Mandarin/English branch | MAN or ENG calls in batch |
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
   - `CONVERTED_TEXT_FOLDER`
   - `FILE_MAPPING_PATH`
   - `SCRIPT_FILE_PATH`
   - `OUTPUT_FOLDER`
   - `LOG_FILE_PATH`
   - `SBERT_MODEL_PATH`
   - `SCRIPT_EMBEDDINGS_PATH`

---

## Process flow (step-by-step)

### Flow 1 (production flow): dynamic refresh + batch run
Use this as your default real workflow.

1. **Prepare inputs**
   - Update/confirm `Scripts.xlsx`
   - Ensure `file_mapping.xlsx` and `converted_text/*.csv` are ready

2. **If scripts changed, refresh dynamic resources first**
   ```bash
   python dynamic_script_analysis.py
   ```
   This updates dynamic term importance (and SBERT script resources) used at runtime.  
   For MAN/ENG, generated script embeddings use `clean_text` keys to match runtime lookup semantics.

3. **Run unified batch entrypoint**
   ```bash
   python run_batch_analysis.py
   ```

4. **Language routing inside batch run (important)**
   - `CAN` files -> `improved_call_coverage_checker.py`
   - `MAN` files -> `Mandarin/improved_call_coverage_checker_M.py`
   - `ENG` files -> also `Mandarin/improved_call_coverage_checker_M.py` (ENG branch)

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
  - TF-IDF + cosine as main lexical-semantic signal
  - expanded overlap / fuzzy / keyword coverage as supporting metrics
- **MAN/ENG (`Mandarin/improved_call_coverage_checker_M.py`)**
  - SBERT similarity as primary signal
  - expanded overlap + ROUGE-L + keyword coverage as supporting metrics
  - embedding cache keys are normalized to `clean_text` (`preprocess_text(..., mode='comparison')`)
  - runtime SBERT lookup uses the same `clean_text` keys as encoding input, preventing formatting-only cache misses
  - legacy embedding files are normalized at load time for backward compatibility
- Final weighted scores use language-specific weights from `config.py`.

### 5) Script preparation guidance (very important)
To maximize match quality when preparing `Scripts.xlsx`:
- keep one discussion point focused on one compliance intent
- avoid overly long mixed-intent paragraphs in one cell
- include realistic paraphrases/variants in `Standard_Script` (not only one formal sentence)
- keep critical numbers/percentages/terms explicit (e.g., rates, price, tenor, IDs)
- use consistent terminology with actual sales wording; synonyms help but cannot fix wrong intent definition
- if scripts changed materially, rerun `dynamic_script_analysis.py` before batch run

### 6) Coverage decision
- For each required point, best matched group score is compared with threshold.
- Score >= threshold -> `Covered`; otherwise `Not Covered`.

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