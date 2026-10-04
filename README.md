# Super Smash Bros Melee Victory Predictor

**Universidad de Carabobo**  
**Facultad Experimental de Ciencias y Tecnología (FACYT)**  
**Departamento de Computación**  
**Course:** Machine Learning (Aprendizaje Automático)  
**Authors:** Adrián Ramírez, Sebastián Piñango  
**Professor:** Álvaro Espinoza  

This repository contains the end-to-end data processing, exploratory data analysis, and feature engineering pipelines for predicting the winner of Super Smash Bros. Melee matches based on instantaneous snapshot frames.

---

## Repository Structure

```text
.
├── EDA.ipynb                         # Phase 1: Exploratory Data Analysis notebook (13 sections)
├── README.md                         # Project documentation and reproduction guide
├── requirements.txt                  # Python dependencies
├── data/
│   ├── archive/                      # Raw msgpack archives and labels (tracked via DVC)
│   │   ├── train/                    # 256 training archives (0000.msgpack.zst - 00ff.msgpack.zst)
│   │   ├── test/                     # Test archives
│   │   └── training_labels.csv       # Match outcome labels (65M+ records)
│   └── processed_sample.parquet      # Flattened 540k sample dataset with reconstructed match_id
├── notebooks/
│   └── 02_data_preparation.ipynb     # Phase 2: Data Preparation & Feature Engineering pipeline
└── src/
    └── extract_data.py               # Streaming decompression, match reconstruction, and flattening script
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
*   `scikit-learn`: Transformers, `ColumnTransformer`, pipelines, and `GroupShuffleSplit`.
*   `ipykernel` & `nbconvert`: Programmatic Jupyter notebook execution.

### 3. Jupyter Kernel Registration
Register the virtual environment as an interactive kernel:
```bash
.venv/bin/python -m ipykernel install --user --name=ssbm-victory-predictor --display-name="Python (SSBM Predictor)"
```
When opening `EDA.ipynb` or `notebooks/02_data_preparation.ipynb` in VS Code or Jupyter Lab, select the **Python (SSBM Predictor)** kernel.

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

## Phase 1 — Exploratory Data Analysis Summary (`EDA.ipynb`)

The exploratory analysis is organized into 13 cohesive sections with all figures empirically verified:

1.  **Formal Prediction Timestamp:** Defined as $t = \text{frame}$. Permissible features comprise any observation observable at or before $t$ ($\mathcal{F}_t$). The target $Y \in \{1, 2\}$ is the final match winner. This formal definition establishes that snapshot state variables are not data leakage.
2.  **Match Independence and Leakage Prevention:** Row shuffling does not equal statistical independence. Consecutive frames from the same match share stage, characters, player skill differential, and temporal autocorrelation. The identification of 49 complete matches (median duration: 10,627 frames / ~2.95 min) proves that train/test splits must be grouped by match.
3.  **Cross-Partition Representativeness Audit:** Comparing partitions across the 256 archives (`0000`, `0040`, `0080`, `00c0`) revealed that file-to-file variance (e.g. P1 win rate ranging from 45.53% in `0040` to 67.38% in `0000`) is caused by each archive containing ~24 complete matches.
4.  **Data Quality & Observation Duplicates:** `row_id` is 100% unique. Across 12 physical game state columns, 338 duplicate observations (0.0625%) were detected, restricted entirely to the frozen countdown phase (`frame < 0`, 5,904 frames / 1.09%).
5.  **Target Balance & Heuristic Baseline:**
    *   Target distribution: Player 1 Wins = **55.4459%** (299,929 frames), Player 2 Wins = **44.5541%** (241,011 frames).
    *   Naive domain baseline (player with more stocks wins; ties broken by lower damage percent): achieves **74.8024% accuracy** (404,636 / 540,940), P1 F1 = 77.96%, P2 F1 = 70.59%, Macro F1 = 74.27%. This sets the minimum performance floor for ML models.
6.  **Predictability Across Match Progression:** Baseline accuracy is 54.17% during countdown, 71.25% in early game (0–5,000 frames), 75.38% in mid game (5,000–12,000 frames), and climbs to **92.21% in late game** (>12,000 frames / >3.3 min).
7.  **Tactical Variables & Feature Engineering:**
    *   `stock_difference`: Pearson correlation with outcome = **-0.4588**, Spearman = **-0.4518** (strongest univariable predictor).
    *   `percent_difference`: Pearson correlation = **+0.0643** (weak linear relationship; acts as an essential non-linear tie-breaker).
    *   Tactical penalties: Being forced off-stage reduces Player 1's win probability from 56.49% to 50.32% (-6.17%). Being in hitstun reduces win probability from 57.23% to 49.22% (-8.01%).

---

## Phase 2 — Data Preparation & Feature Engineering (`notebooks/02_data_preparation.ipynb`)

Phase 2 builds directly upon the findings of the EDA, delivering a leak-free, modular Scikit-Learn preprocessing pipeline ready for model training.

### 1. Pre-Split Cleansing
*   **5,904 countdown frames (`frame < 0`)** are removed (no combat signal, static frozen states).
*   **48 terminal frames (`p1_stock == 0` or `p2_stock == 0`)** are removed (outcome already decided).
*   **Retained dataset:** **534,988 clean observations**.

### 2. Anti-Leakage Partitioning (`GroupShuffleSplit`)
To prevent severe frame-level data leakage, partitioning is performed **before fitting any transformer**:
*   `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly on `match_id`.
*   **Train set:** 39 matches (426,378 frames, 79.7%).
*   **Test set:** 10 matches (108,610 frames, 20.3%).
*   **Leakage verification:** Verified zero match overlap between train and test (`assert len(matches_train & matches_test) == 0`). Target balance is preserved: Train P1 = 56.40%, Test P1 = 51.71%.

### 3. Custom Feature Engineering (`MeleeFeatureEngineer`)
Implemented as a native Scikit-Learn transformer (`BaseEstimator`, `TransformerMixin`) that operates on copies without mutating inputs. It derives 5 domain-informed features:
*   `stock_diff`: `p1_stock - p2_stock` (relative stock advantage).
*   `percent_diff`: `p2_percent - p1_percent` (relative damage lead).
*   `stock_ratio`: `p1_stock / (p2_stock + 1e-4)` (non-linear survival ratio, regularized against zero division).
*   `spatial_distance`: $\sqrt{(p1_x - p2_x)^2 + (p1_y - p2_y)^2}$ (spacing/neutral game proxy).
*   `p1_edgeguard_advantage`: `(p2_off_stage == 1) & (p1_off_stage == 0)` (dominant edgeguarding state).
This expands the feature space from 46 to 51 columns.

### 4. Column Grouping & Preprocessing Pipeline
Features are categorized into explicit, mutually exclusive groups verified by assertion:
*   **Numerical (31 features):** Coordinates, velocities, action frames, shield health, stocks, damage, and continuous derived features $\rightarrow$ `SimpleImputer(strategy="median")` + `StandardScaler()`.
*   **Categorical (3 features):** `stage`, `p1_character`, `p2_character` $\rightarrow$ `SimpleImputer(strategy="constant", fill_value="missing")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
*   **Binary (11 features):** Tactical booleans (`on_ground`, `off_stage`, `invulnerable`, `facing`, etc.) $\rightarrow$ cast to `float32`.
*   **Excluded (4 features):** `p1_costume`, `p2_costume` (cosmetic identity proxies) and `p1_action`, `p2_action` (excessive cardinality) are dropped via `ColumnTransformer(remainder="drop")`.

### 5. Pipeline Assembly and Output Matrices
The full pipeline chains feature generation and column transformations into a single object:
```python
full_pipeline = Pipeline(steps=[
    ("features", MeleeFeatureEngineer()),
    ("preprocess", preprocessor),
])
```
*   Fitted **exclusively on `X_train`**.
*   Transforms both splits into dense float32 matrices with **74 output features**:
    *   `X_train_proc`: shape `(426378, 74)`
    *   `X_test_proc`: shape `(108610, 74)`
*   **Structural Verificaciones Passed:** 0 NaN values, 0 infinite values, binary features bounded in $[0.0, 1.0]$, and verified non-mutation of input DataFrames.

---

## Reproduction Guide

To execute both phases and reproduce all outputs from scratch:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Extract the sample data from raw archives
python src/extract_data.py \
  --input_files data/archive/train/0000.msgpack.zst data/archive/train/0001.msgpack.zst \
  --output_path data/processed_sample.parquet

# 3. Execute Phase 1 (EDA)
jupyter nbconvert --to notebook --execute EDA.ipynb --output EDA.ipynb

# 4. Execute Phase 2 (Data Preparation & Feature Engineering)
cd notebooks
jupyter nbconvert --to notebook --execute 02_data_preparation.ipynb --output 02_data_preparation.ipynb
cd ..
```
Both notebooks will execute to completion with return code 0, validating all assertions and embedding visual outputs directly into the `.ipynb` files.