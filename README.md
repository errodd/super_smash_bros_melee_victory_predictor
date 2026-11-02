# Super Smash Bros Melee Victory Predictor

**Universidad de Carabobo**
**Facultad Experimental de Ciencias y Tecnología (FACYT)**
**Departamento de Computación**
**Course:** Machine Learning (Aprendizaje Automático)
**Authors:** Adrián Ramírez, Sebastián Piñango
**Professor:** Álvaro Espinoza

This repository contains the end-to-end data processing, exploratory data analysis, feature engineering, and model selection pipelines for predicting the winner of Super Smash Bros. Melee matches based on instantaneous snapshot frames.

---

## Repository Structure

```text
.
├── README.md                              # Project documentation and reproduction guide
├── requirements.txt                       # Python dependencies
├── data/
│   ├── archive/                           # Raw msgpack archives and labels (tracked via DVC)
│   │   ├── train/                         # Training archives (0000.msgpack.zst - 00ff.msgpack.zst)
│   │   ├── test/                          # Test archives
│   │   └── training_labels.csv            # Match outcome labels
│   └── processed_sample.parquet           # Flattened 540k sample dataset with reconstructed match_id
├── notebooks/
│   ├── 01_eda.ipynb                       # Phase 1: Exploratory Data Analysis
│   ├── 02_data_preparation.ipynb          # Phase 2: Data Preparation & Feature Engineering
│   └── 03_modeling_and_model_selection.ipynb  # Phase 3: Model Selection & Hyperparameter Optimization
├── reports/
│   └── figures/
└── src/
    ├── extract_data.py                    # Streaming decompression, match reconstruction, and flattening script
    ├── audit_processed_sample.py          # Processed dataset auditing
    ├── generate_sample.py                 # Sample generation utility
    ├── inspect_dataset.py                 # Raw dataset inspection
    └── features/
        └── transformers.py                # MeleeFeatureEngineer (domain feature engineering)
```

---

## Environment Setup and Configuration

The project runs on Python 3.10+ using a dedicated virtual environment.

### 1. Initialize the Virtual Environment
Activate the pre-configured virtual environment in your terminal:
```bash
source .venv/bin/activate
```

### 2. Dependencies
Install the required packages using `uv` or `pip`:
```bash
uv pip install -r requirements.txt
```
Key dependencies:
*   `zstandard` & `msgpack`: Streaming decompression and binary deserialization.
*   `pandas` & `numpy`: High-performance tabular data structures and numerical operations.
*   `pyarrow`: Parquet I/O and columnar storage.
*   `matplotlib` & `seaborn`: Statistical visualizations.
*   `scikit-learn`: Transformers, `ColumnTransformer`, pipelines, `GroupShuffleSplit`, `GroupKFold`.
*   `optuna`: Hyperparameter optimization.
*   `ipykernel` & `nbconvert`: Programmatic Jupyter notebook execution.

### 3. Jupyter Kernel Registration
Register the virtual environment as an interactive kernel:
```bash
.venv/bin/python -m ipykernel install --user --name=ssbm-victory-predictor --display-name="Python (SSBM Predictor)"
```
When opening a notebook in VS Code or Jupyter Lab, select the **Python (SSBM Predictor)** kernel.

---

## Dataset Management via DVC

The raw dataset comprises over 65 million frames (~880 MB compressed, ~15 GB uncompressed). Data files are managed and versioned using DVC (Data Version Control) backed by Google Drive.

### Pulling the Data
```bash
dvc pull
```
This retrieves `data/archive/train/*.msgpack.zst` and `data/archive/training_labels.csv`.

**Security Notice Regarding Credentials:**
Never commit or push OAuth client secrets, API tokens, or `.dvc/config.local` to Git. Private credentials must remain exclusively in `.dvc/config.local`, which is listed in `.dvc/.gitignore`.

---

## Data Extraction Pipeline (`src/extract_data.py`)

The extraction script bridges raw nested binary game states into structured tabular Parquet records.

Key capabilities:
1.  **Streaming Decompression:** Reads `.msgpack.zst` archives chunk-by-chunk using `zstandard` and `msgpack` with `strict_map_key=False`.
2.  **Match Reconstruction (`match_id`):** Frames inside the raw archives were shuffled. The script reconstructs complete continuous matches by grouping frames sharing identical stage, player costumes, and character choices, and sorting by frame index. Any continuous sequence ($f, f+1, f+2, \dots$) is assigned a unique `match_id` (e.g. `0000_m01`).
3.  **Unmasked Schema:** Avoids silent fallback defaults (`.get("stock", 0)`), ensuring that absent fields evaluate to `None`/`NaN` for true schema validation.
4.  **Early-Exit Label Matching:** Streams `training_labels.csv` lexicographically, loading label subsets in under two seconds.

To extract a 540,940-frame sample from partitions `0000` and `0001` (49 reconstructed matches) and save to Parquet:
```bash
python src/extract_data.py \
  --input_files data/archive/train/0000.msgpack.zst data/archive/train/0001.msgpack.zst \
  --output_path data/processed_sample.parquet
```

---

## Phase 1 — Exploratory Data Analysis Summary (`notebooks/01_eda.ipynb`)

1.  **Formal Prediction Timestamp:** Defined as $t = \text{frame}$. Permissible features comprise any observation observable at or before $t$ ($\mathcal{F}_t$). The target $Y \in \{1, 2\}$ is the final match winner. Snapshot state variables are not data leakage.
2.  **Match Independence and Leakage Prevention:** Row shuffling does not equal statistical independence. 49 complete match segments were reconstructed (median duration: 10,627 frames / ~2.95 min). `0000_m16` and `0000_m25` are two fragments of the same real match (Zelda's mid-match transformation splits `p1_character`), so they must stay on the same side of any split/fold.
3.  **Cross-Partition Representativeness Audit:** File-to-file variance (e.g. P1 win rate ranging from 45.53% in `0040` to 67.38% in `0000`) is caused by each archive containing ~20-25 complete matches.
4.  **Data Quality & Observation Duplicates:** `row_id` is 100% unique. 338 duplicate observations (0.0625%) were detected, restricted entirely to the frozen countdown phase (`frame < 0`, 5,904 frames / 1.09%).
5.  **Target Balance & Heuristic Baseline:** P1 wins = 55.4459%, P2 wins = 44.5541%. Naive domain baseline (more stocks wins; ties broken by lower damage percent) achieves **74.8024% accuracy**, Macro F1 = 74.27%. This sets the performance floor for ML models.
6.  **Predictability Across Match Progression:** 54.17% (countdown) → 71.25% (early) → 75.38% (mid) → **92.21%** (late game, >3.3 min).
7.  **Tactical Variables:** `stock_difference` correlates -0.4588 with outcome (strongest univariable predictor). `percent_difference` correlates +0.0643 (weak linear, strong non-linear tie-breaker). Off-stage and hitstun states measurably reduce win probability.

---

## Phase 2 — Data Preparation & Feature Engineering (`notebooks/02_data_preparation.ipynb`)

### 1. Pre-Split Cleansing
*   5,904 countdown frames (`frame < 0`) removed.
*   48 terminal frames (`p1_stock == 0` or `p2_stock == 0`) removed.
*   Retained dataset: **534,988 clean observations**.

### 2. Anti-Leakage Partitioning (`GroupShuffleSplit`)
*   `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly on `match_id`.
*   Train: 39 matches (426,378 frames). Test: 10 matches (108,610 frames).
*   Zero match overlap verified between train and test.

### 3. Custom Feature Engineering (`MeleeFeatureEngineer`)
Native Scikit-Learn transformer deriving domain features: `stock_diff`, `percent_diff`, `stock_ratio`, `spatial_distance`, `p1_edgeguard_advantage`.

### 4. Column Grouping & Preprocessing Pipeline
Numerical (median imputation + scaling), categorical (`stage`, `p1_character`, `p2_character`; one-hot), binary (cast to float32). `p1_costume`, `p2_costume`, `p1_action`, `p2_action` excluded.

### 5. Pipeline Assembly and Output Matrices
```python
full_pipeline = Pipeline(steps=[
    ("features", MeleeFeatureEngineer()),
    ("preprocess", preprocessor),
])
```
Fitted exclusively on `X_train`. Produces dense float32 matrices with 74 output features, 0 NaN/inf.

---

## Phase 3 — Model Selection & Hyperparameter Optimization (`notebooks/03_modeling_and_model_selection.ipynb`)

*   **Protocol:** `GroupKFold(n_splits=5)` on `match_id` (38 canonical groups after unifying `0000_m25` into `0000_m16`), reused across all models.
*   **Diagnosed issue:** the full 74-feature pipeline overfit to match identity (stage/character one-hot acted as a match fingerprint with only 38 groups); model selection uses 5 domain-derived features instead (`stock_diff`, `percent_diff`, `stock_ratio`, `spatial_distance`, `p1_edgeguard_advantage`).
*   **Candidates:** `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, compared via grouped CV (accuracy, F1 macro, ROC AUC).
*   **Tuning:** Optuna (`TPESampler(seed=42)`) tunes `LogisticRegression` with grouped CV inside each trial.
*   **Final test evaluation** (touched once): accuracy = 0.739260, F1 macro = 0.739233, ROC AUC = 0.804667. The ML pipeline did not surpass the domain heuristic baseline (75.03%); the value of this phase is the leakage-free selection methodology, not beating a deterministic rule.

---

## Reproduction Guide

> **Working directory note:** notebooks resolve the dataset path differently and must be launched accordingly:
> * `notebooks/01_eda.ipynb` uses the root-relative path `data/processed_sample.parquet` → run **from the repository root**.
> * `notebooks/02_data_preparation.ipynb` and `notebooks/03_modeling_and_model_selection.ipynb` use `Path("../data/processed_sample.parquet")` → run **from inside `notebooks/`**.

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Extract the sample data from raw archives
python src/extract_data.py \
  --input_files data/archive/train/0000.msgpack.zst data/archive/train/0001.msgpack.zst \
  --output_path data/processed_sample.parquet

# 3. Execute Phase 1 (EDA) — from repository root
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda.ipynb

# 4. Execute Phase 2 and Phase 3 — from notebooks/
cd notebooks
jupyter nbconvert --to notebook --execute 02_data_preparation.ipynb --output 02_data_preparation.ipynb
jupyter nbconvert --to notebook --execute 03_modeling_and_model_selection.ipynb --output 03_modeling_and_model_selection.ipynb
cd ..
```
All three notebooks execute to completion with return code 0, validating all assertions and embedding visual outputs directly into the `.ipynb` files.