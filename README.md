# Super Smash Bros Melee Victory Predictor

This repository contains the dataset processing scripts and the Exploratory Data Analysis (EDA) for the Super Smash Bros Melee victory predictor project. The objective of this project is to analyze match frame states to predict whether Player 1 or Player 2 will win the match.

---

## Environment Setup and Configuration

The project uses a Python virtual environment to manage dependencies. Follow these instructions to set up and run the code.

### 1. Initialize the Virtual Environment
Activate the pre-configured virtual environment in your terminal:
```bash
source .venv/bin/activate
```

### 2. Dependencies
The required packages are defined in the requirements.txt file. They can be installed using uv (recommended for speed) or pip:
```bash
uv pip install -r requirements.txt
```
Key packages include:
*   zstandard & msgpack: For decompressing and parsing raw match logs.
*   pandas & numpy: For data manipulation.
*   pyarrow: For efficient Parquet storage.
*   matplotlib & seaborn: For plotting and visualization.
*   scikit-learn: For metrics evaluation.
*   ipykernel & nbconvert: For running and verifying the Jupyter notebook.

### 3. Jupyter Kernel Registration
A custom kernel has been registered to ensure Jupyter uses the correct environment:
```bash
.venv/bin/python -m ipykernel install --user --name=ssbm-victory-predictor --display-name="Python (SSBM Predictor)"
```
When opening the notebook, select the "Python (SSBM Predictor)" kernel.

---

## Data Processing Pipeline

Because the raw dataset consists of 65 million frames (exceeding standard RAM capacity), we use a modular pipeline to extract a representative sample.

The script `src/extract_data.py` reads compressed `.msgpack.zst` files, flattens the nested player structures, and merges them with match outcomes from `training_labels.csv`. It optimizes memory by scanning labels sequentially and terminating early once target prefixes are matched.

To extract a sample (e.g. the first two archives, representing 540,940 frames) and save it to Parquet:
```bash
python src/extract_data.py \
  --input_files data/archive/train/0000.msgpack.zst data/archive/train/0001.msgpack.zst \
  --output_path data/processed_sample.parquet
```

---

## Exploratory Data Analysis (EDA) Summary

The analysis is fully documented in EDA.ipynb. Below is an overview of the findings and the reasoning applied during the exploration.

### 1. Structure and Observation Unit
The unit of analysis is a single frame of gameplay. Because the dataset has been fully shuffled at the frame level, temporal context is lost. This requires models to treat each frame as an independent observation. 

### 2. Target Balance
The target variable (outcome) is balanced. Player 1 wins in 49.88% of the frames, and Player 2 wins in 50.12% of the frames. This balance means we do not need to apply resampling techniques, and standard accuracy is a reliable metric for evaluation.

### 3. Data Quality and Cleansing
The raw ranges of stocks and damage percentages conform to game mechanics. We identified a small percentage of frames (0.16%) with negative frame indices, which correspond to the pre-game countdown phase. These frames contain static coordinates with no combat activity. A key preprocessing recommendation is to filter out these frames to remove noise.

### 4. Naive Baseline
A rule-based heuristic was established: predict the winner as the player with more remaining stocks; if stocks are tied, predict the player with less damage percentage. This simple baseline achieves an accuracy of 78.43%. Any machine learning model developed in subsequent phases must exceed this threshold to be considered useful.

### 5. Feature Engineering
We engineered relative features to represent the advantage state of a player:
*   stock_difference (Player 1 stocks - Player 2 stocks)
*   percent_difference (Player 1 damage - Player 2 damage)
*   distance_between_players (Euclidean spatial distance)

Statistical testing showed that these engineered features correlate much more strongly with the outcome than the raw individual features. For instance, stock_difference has a correlation of -0.631 with the winner outcome, making it a critical driver for prediction.