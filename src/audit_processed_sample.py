"""Auditoria del parquet procesado antes de la preparacion de datos.

Responde con numeros: cuantos frames estan decididos (stock == 0), que
segmentos de partida son sospechosos (p. ej. Zelda/Sheik), si la etiqueta
es consistente por partida y que tan soportadas estan las categorias.
No escribe nada en disco.

Uso (desde la raiz del proyecto):
    python src/audit_processed_sample.py
    python src/audit_processed_sample.py --path data/processed_sample.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/processed_sample.parquet")
SHORT_SEGMENT_FRAMES = 500
EARLY_END = 5000
MID_END = 12000
REQUIRED_COLUMNS = [
    "row_id", "match_id", "frame", "stage", "outcome",
    "p1_character", "p2_character", "p1_stock", "p2_stock",
    "p1_percent", "p2_percent",
]
SEPARATOR = "=" * 60


def title(text):
    """Print a section title."""
    print(f"\n{SEPARATOR}\n{text}\n{SEPARATOR}")


def naive_prediction(frame_df):
    """Predict the winner: more stocks, then less damage, else player 1."""
    p1_ahead = frame_df["p1_stock"] > frame_df["p2_stock"]
    p2_ahead = frame_df["p2_stock"] > frame_df["p1_stock"]
    p1_less = frame_df["p1_percent"] < frame_df["p2_percent"]
    p2_less = frame_df["p2_percent"] < frame_df["p1_percent"]
    return np.select(
        [p1_ahead, p2_ahead, p1_less, p2_less], [1, 2, 1, 2], default=1
    )


def accuracy(frame_df):
    """Naive-baseline accuracy (0-1) for a DataFrame, NaN if empty."""
    if frame_df.empty:
        return float("nan")
    return float((naive_prediction(frame_df) == frame_df["outcome"]).mean())


def section_overview(df):
    """Shape, memory, dtypes and nulls."""
    title("1. FORMA, MEMORIA Y NULOS")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Filas: {len(df):,}  Columnas: {df.shape[1]}  "
          f"Memoria: {memory_mb:,.0f} MB")
    nulls = df.isna().sum()
    print("Nulos:", nulls[nulls > 0].to_dict() if nulls.any() else "ninguno")
    print("\nTipos:", df.dtypes.astype(str).value_counts().to_dict())


def section_decided_frames(df):
    """Frames before the fight and frames after the game is decided."""
    title("2. FRAMES SIN SENAL O YA DECIDIDOS")
    countdown = df["frame"] < 0
    decided = (df["p1_stock"] == 0) | (df["p2_stock"] == 0)
    for name, mask in (("frame < 0", countdown),
                       ("stock == 0 (algun jugador)", decided)):
        print(f"{name:<28} {int(mask.sum()):>9,} "
              f"({mask.mean():.2%})")
    print(f"\nAccuracy baseline total:            {accuracy(df):.2%}")
    clean = df[~countdown & ~decided]
    print(f"Accuracy baseline sin esos frames:   {accuracy(clean):.2%}")
    print(f"Accuracy SOLO en frames stock == 0:  "
          f"{accuracy(df[decided]):.2%}")
    print("\nPor fase (frames limpios):")
    phases = [("temprana", 0, EARLY_END), ("media", EARLY_END, MID_END),
              ("tardia", MID_END, 10**9)]
    for name, low, high in phases:
        part = clean[(clean["frame"] >= low) & (clean["frame"] < high)]
        print(f"  {name:<9} {len(part):>9,} frames  "
              f"accuracy {accuracy(part):.2%}")


def section_matches(df):
    """Match sizes, short segments and their neighbours."""
    title("3. PARTIDAS (match_id) Y SEGMENTOS SOSPECHOSOS")
    groups = df.groupby("match_id")
    summary = groups.agg(
        frames=("frame", "size"), first_row=("row_id", "first"),
        stage=("stage", "first"), p1=("p1_character", "first"),
        p2=("p2_character", "first"), outcome=("outcome", "nunique"),
    )
    print(f"Partidas/segmentos: {len(summary)}")
    print(summary["frames"].describe().round(0).to_string())
    short = summary[summary["frames"] < SHORT_SEGMENT_FRAMES]
    print(f"\nSegmentos cortos (< {SHORT_SEGMENT_FRAMES} frames): "
          f"{len(short)}")
    parts = summary["first_row"].str.rsplit("_", n=1, expand=True)
    summary["file_id"] = parts[0]
    summary["row_index"] = pd.to_numeric(parts[1], errors="coerce")
    ordered = summary.sort_values(["file_id", "row_index"]).reset_index()
    for position in ordered.index[ordered["frames"] < SHORT_SEGMENT_FRAMES]:
        row = ordered.loc[position]
        print(f"\n  {row['match_id']}: {row['frames']} frames, "
              f"{row['p1']} vs {row['p2']} en {row['stage']}")
        for label, other in (("anterior", position - 1),
                             ("siguiente", position + 1)):
            if 0 <= other < len(ordered):
                near = ordered.loc[other]
                print(f"    {label}: {near['match_id']} "
                      f"({near['frames']} frames) "
                      f"{near['p1']} vs {near['p2']}")
    transformers = df["p1_character"].isin(["ZELDA", "SHEIK"]) | df[
        "p2_character"].isin(["ZELDA", "SHEIK"])
    ids = sorted(df.loc[transformers, "match_id"].unique())
    print(f"\nPartidas con ZELDA o SHEIK ({len(ids)}): {ids[:15]}")
    return summary


def section_labels(df, summary):
    """Label consistency per match and match-level balance."""
    title("4. ETIQUETA POR PARTIDA")
    inconsistent = int((summary["outcome"] > 1).sum())
    print(f"Partidas con mas de un valor de outcome: {inconsistent}")
    last = df.sort_values("frame").groupby("match_id").tail(1)
    p1_dead = last["p1_stock"] == 0
    p2_dead = last["p2_stock"] == 0
    resolved = last[p1_dead ^ p2_dead]
    agree = ((resolved["p2_stock"] == 0) & (resolved["outcome"] == 1)) | (
        (resolved["p1_stock"] == 0) & (resolved["outcome"] == 2))
    print(f"Partidas cuyo ultimo frame tiene un solo jugador en 0 stocks: "
          f"{len(resolved)} de {len(last)}")
    print(f"  y la etiqueta coincide con ese resultado: "
          f"{int(agree.sum())}")
    wins = last["outcome"].value_counts()
    print(f"\nVictorias de P1 por PARTIDA: {int(wins.get(1, 0))} de "
          f"{len(last)} ({wins.get(1, 0) / len(last):.1%}) "
          f"- por frames seria {(df['outcome'] == 1).mean():.1%}")


def section_support(df):
    """How many matches back each stage and character."""
    title("5. SOPORTE POR CATEGORIA (en numero de PARTIDAS)")
    per_stage = df.groupby("stage")["match_id"].nunique()
    print("Escenarios:\n" + per_stage.sort_values().to_string())
    chars = pd.concat([
        df[["match_id", "p1_character"]].rename(
            columns={"p1_character": "character"}),
        df[["match_id", "p2_character"]].rename(
            columns={"p2_character": "character"}),
    ]).drop_duplicates()
    per_char = chars.groupby("character")["match_id"].nunique()
    print("\nPersonajes (P1 o P2):\n" + per_char.sort_values().to_string())


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    if not args.path.exists():
        raise FileNotFoundError(
            f"No existe {args.path}. Genera la muestra con "
            "src/extract_data.py primero."
        )
    df = pd.read_parquet(args.path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas esperadas: {missing}")
    section_overview(df)
    section_decided_frames(df)
    summary = section_matches(df)
    section_labels(df, summary)
    section_support(df)


if __name__ == "__main__":
    main()
