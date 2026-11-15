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
2.  **Match Independence and Leakage Prevention:** Row shuffling does not equal statistical independence. 48 canonical matches were reconstructed (median duration: 10,629 frames / ~2.95 min), unifying mid-match Sheik/Zelda transformations (`0000_m16` and `0000_m25`) directly in `src/extract_data.py` before any split or fold assignment.
3.  **Cross-Partition Representativeness Audit:** File-to-file variance (e.g. P1 win rate ranging from 45.53% in `0040` to 67.38% in `0000`) is caused by each archive containing ~20-25 complete matches.
4.  **Data Quality & Observation Duplicates:** `row_id` is 100% unique. 338 duplicate observations (0.0625%) were detected, restricted entirely to the frozen countdown phase (`frame < 0`, 5,904 frames / 1.09%).
5.  **Target Balance & Heuristic Baseline:** P1 wins = 55.4459%, P2 wins = 44.5541%. Naive domain baseline (more stocks wins; ties broken by lower damage percent) achieves **74.8024% accuracy**, Macro F1 = 74.27%. This sets the performance floor for ML models.
6.  **Predictability Across Match Progression:** 54.17% (countdown) → 71.25% (early) → 75.38% (mid) → **92.21%** (late game, >3.3 min).
7.  **Tactical Variables:** `stock_difference` correlates -0.4588 with outcome (strongest univariable predictor). `percent_difference` correlates +0.0643 (weak linear, strong non-linear tie-breaker). Off-stage and hitstun states measurably reduce win probability.

---

## Phase 2 — Data Preparation & Feature Engineering (`notebooks/02_data_preparation.ipynb`)

### 1. Pre-Split Cleansing & Canonicalization
*   Operates on the expanded development dataset `data/processed_dev.parquet` (partitions `0000`–`0005`, 1,577,114 raw frames, 144 canonical matches).
*   Countdown frames (`frame < 0`) and terminal frames (`p1_stock == 0` or `p2_stock == 0`) are removed deterministically, retaining **1,559,260 clean observations**.

### 2. Anti-Leakage Partitioning (`GroupShuffleSplit`)
*   `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly on canonical `match_id`.
*   Train: 115 matches. Test: 29 matches. Zero match overlap verified.

### 3. Custom Feature Engineering (`src/features/transformers.py`)
Native Scikit-Learn transformer (`MeleeFeatureEngineer`) and pipeline factory (`build_model_pipeline`) imported directly from `src/features/transformers.py` (DRY):
*   Base 5 features + symmetric edgeguard (`p2_edgeguard_advantage`, trivalued `edgeguard_advantage`) + configurable lexicographic domain features (`stocks_tied`, `stocks_tied_x_percent_diff`, `heuristic_pred`, `advantage_sign`, `advantage_magnitude`).

### 4. Column Grouping, Preprocessing & Handoff Artifacts
Numerical (median imputation + scaling), categorical (`stage`, `p1_character`, `p2_character`; one-hot), binary (`SimpleImputer(strategy="most_frequent")` + `float64` cast). Produces uniform **`float64`** dense matrices with 0 NaN/inf. Exposes raw `X_train`, `y_train`, `groups_train`, and unfitted `build_model_pipeline(estimator)` for CV (never `X_train_proc`), and exports `artifacts/split_manifest.parquet`, `artifacts/groups_train.npy`, and `artifacts/prep_metadata.json`.

---

## Phase 3 — Model Selection & Hyperparameter Optimization (`notebooks/03_modeling_and_model_selection.ipynb`)

*   **Protocol:** `StratifiedGroupKFold(n_splits=5)` vs `GroupKFold(n_splits=5)` on canonical `match_id` (115 training matches).
*   **Fair Domain Heuristic & Ablation:** Lexicographic heuristic evaluated on the exact same 5 CV folds, Test, and blind Holdout; 6-configuration ablation study (`base5` through `full + actions` with `min_frequency` and strong regularization) plus lexicographic feature engineering evaluation.
*   **Tuning & OOF Threshold:** Optuna (`TPESampler(seed=42)`) tunes `HistGradientBoostingClassifier` (20 trials) and `LogisticRegression` (15 trials); grouped out-of-fold (OOF) probabilities select the optimal accuracy threshold (`0.505`) before touching Test or Holdout.
*   **Final Single Evaluation & Serialization:** Evaluated once on Test (29 matches) and once on a completely untouched blind Holdout (`data/processed_holdout.parquet`, partitions `00fd`+`00fe`, 48 matches / 516,995 clean frames: **73.17% accuracy, 0.8167 ROC AUC** vs 73.33% domain heuristic). Final pipeline and metadata serialized to `models/final_model.joblib` and `models/model_metadata.json`.

---

## Reproduction Guide

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run unit tests
python -m pytest -v tests/

# 3. Extract canonicalized sample, dev, and blind holdout datasets
python src/extract_data.py --partitions 0000 0001 --output_path data/processed_sample.parquet
python src/extract_data.py --partitions 0000 0001 0002 0003 0004 0005 --output_path data/processed_dev.parquet
python src/extract_data.py --partitions 00fd 00fe --output_path data/processed_holdout.parquet

# 4. Execute all three notebooks sequentially (Restart Kernel -> Run All)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_preparation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling_and_model_selection.ipynb
```
All three notebooks execute to completion with return code 0, validating all assertions and embedding visual outputs directly into the `.ipynb` files.