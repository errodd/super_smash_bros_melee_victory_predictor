# Super Smash Bros Melee Victory Predictor

This repository contains the data extraction pipeline and the Exploratory Data Analysis (EDA) for the Super Smash Bros Melee victory prediction project. The core objective is to analyze instantaneous game state snapshots taken at a given frame to predict whether Player 1 or Player 2 will ultimately win the match.

---

## Environment Setup and Configuration

The project uses a Python virtual environment to manage dependencies. Follow these instructions to set up and run the code.

### 1. Initialize the Virtual Environment
Activate the pre-configured virtual environment in your terminal:
```bash
source .venv/bin/activate
```

### 2. Dependencies
The required packages are defined in `requirements.txt`. They can be installed using uv or pip:
```bash
uv pip install -r requirements.txt
```
Key dependencies:
*   zstandard & msgpack: Stream decompression and binary unpacking.
*   pandas & numpy: Data manipulation and numerical operations.
*   pyarrow: Columnar Parquet reading and writing.
*   matplotlib & seaborn: Statistical visual plotting.
*   scikit-learn: Classification metrics and grouped splitting validation.
*   ipykernel & nbconvert: Programmatic Jupyter notebook execution.

### 3. Jupyter Kernel Registration
A custom kernel has been registered to ensure notebooks execute within this environment:
```bash
.venv/bin/python -m ipykernel install --user --name=ssbm-victory-predictor --display-name="Python (SSBM Predictor)"
```
When opening the notebook in VS Code or Jupyter Lab, select the "Python (SSBM Predictor)" kernel.

---

## Dataset Management via DVC

The dataset is versioned using DVC (Data Version Control) to avoid committing massive binary files to Git.

### Synchronizing the Data
```bash
dvc pull
```
This command pulls the compressed archive files (`data/archive/train/*.msgpack.zst`) and the labels CSV (`data/archive/training_labels.csv`).

**Security Notice Regarding Credentials:**  
Do not commit or push private authentication tokens or OAuth credentials to Git. If configuring remote access, always use the local configuration flag (`dvc remote modify <name> --local credentialpath <path>`), which writes exclusively to `.dvc/config.local` (ignored by Git).

---

## Data Extraction Pipeline

The raw dataset contains over 65 million frames stored across 256 compressed archives (`0000.msgpack.zst` to `00ff.msgpack.zst`). 

The extraction script `src/extract_data.py`:
1.  **Decompresses in streaming mode** using `zstandard` and `msgpack` with `strict_map_key=False`.
2.  **Reconstructs full matches (`match_id`)** by clustering contiguous frame sequences for each unique stage, costume, and character combination.
3.  **Preserves true missingness** by avoiding silent `.get()` fallback defaults, ensuring that absent values are represented as `None`/`NaN`.
4.  **Optimizes label mapping** by reading `training_labels.csv` with early-exit logic, loading hundreds of thousands of outcomes in under two seconds.

To extract a sample of frames (partitions `0000` and `0001`, containing 540,940 frames and 49 reconstructed matches) and save to Parquet:
```bash
python src/extract_data.py \
  --input_files data/archive/train/0000.msgpack.zst data/archive/train/0001.msgpack.zst \
  --output_path data/processed_sample.parquet
```

---

## Exploratory Data Analysis (EDA) Summary

The analysis is documented in `EDA.ipynb`. All metrics and findings below were computed empirically from the actual data.

### 1. Match Structure and Statistical Independence
Physical shuffling of rows does not create statistical independence. Frames originating from the same match share the same two players, character mechanics, and common final outcome, with heavy temporal autocorrelation. By grouping contiguous sequences, we identified 49 complete matches across partitions `0000` and `0001` (ranging from 3,455 frames to 28,097 frames, with a median of 10,627 frames or ~2.95 minutes). Future machine learning models must use grouped splitting (`GroupShuffleSplit` on `match_id`) to prevent severe match-level data leakage.

### 2. Sample Representativeness
Auditing distributed partitions across the 256 archives (`0000`, `0040`, `0080`, `00c0`) demonstrates that individual files exhibit cluster-level variance in stage frequency and target balance (e.g., P1 win rate is 67.38% in `0000` but 45.53% in `0040`). This occurs because each archive holds roughly 20 to 25 complete matches. Production modeling pipelines should draw matches systematically across archives to ensure balanced representation.

### 3. Data Quality and Duplicate States
*   **Key Integrity:** `row_id` is 100% unique across all 540,940 rows.
*   **Physical Observation Duplicates:** We detected 338 duplicate states (0.0625%) across the 12 physical game state features. These occur during the static pre-game countdown phase where characters stand frozen before input is unlocked.
*   **Countdown Cleansing:** 5,904 frames (1.0914%) have negative frame indices (`frame < 0`). These should be filtered out during preprocessing because they contain no combat signal.

### 4. Target Distribution and Naive Baseline
*   **Class Balance:** Player 1 wins in 299,929 frames (55.4459%), and Player 2 wins in 241,011 frames (44.5541%).
*   **Domain Heuristic:** Predicting the player with more remaining stocks (or less damage percent in ties) achieves an accuracy of **74.8024%** (404,636 correct predictions), with a Player 1 F1-score of 77.96% and a Player 2 F1-score of 70.59%. This 74.80% accuracy represents the strict performance floor for subsequent machine learning models.

### 5. Predictability Across Match Progression
Evaluating baseline accuracy conditionally across match phases answers when the winner becomes predictable:
*   *Countdown (`frame < 0`):* 54.17% accuracy (equivalent to a random guess matching the majority class).
*   *Early Game (`0 <= frame < 5000`, ~0-83s):* 71.25% accuracy.
*   *Mid Game (`5000 <= frame < 12000`, ~1.4-3.3m):* 75.38% accuracy.
*   *Late Game (`frame >= 12000`, >3.3m):* **92.21% accuracy**.
Outcome uncertainty is high in early game, while late-game stock disparities make the winner overwhelmingly predictable.

### 6. Tactical States and Feature Engineering
*   **Relative Stock Difference (`stock_difference`):** Shows a moderate univariable linear correlation of -0.4588 with the outcome, outperforming raw individual stock counts. It is a promising candidate feature.
*   **Relative Damage Difference (`percent_difference`):** Shows a weak linear correlation (+0.0643), acting primarily as an essential non-linear discriminator when players have equal stocks.
*   **Tactical Penalties:** Being forced off-stage reduces Player 1's win rate from 56.49% to 50.32% (-6.17%). Being trapped in hitstun reduces Player 1's win rate from 57.23% to 49.22% (-8.01%).

---

## Verification and Reproduction

To reproduce the analysis and verify that the notebook runs without error:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Extract sample data
python src/extract_data.py \
  --input_files data/archive/train/0000.msgpack.zst data/archive/train/0001.msgpack.zst \
  --output_path data/processed_sample.parquet

# 3. Execute notebook programmatically
jupyter nbconvert --to notebook --execute EDA.ipynb --output EDA.ipynb
```
The command will execute all cells, compute all statistics, and save all visual plots directly into `EDA.ipynb`.