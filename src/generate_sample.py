"""
Script de generación de muestra para Super Smash Bros Melee.

Extrae directamente los 3 archivos confirmados de la caché DVC:
- training_labels.csv (desde ...-005.zip)
- 0000.msgpack.zst    (desde ...-004.zip)
- 0001.msgpack.zst    (desde ...-002.zip)

Luego ejecuta extract_data.py y genera data/processed_sample.parquet
reproduciendo exactamente las 540,940 filas y 49 partidas del proyecto.
"""

from pathlib import Path
import shutil
import sys
import zipfile

# Asegurar importación de módulos locales
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.extract_data import process_dataset

# Rutas estándar del proyecto
ARCHIVE_DIR = BASE_DIR / "data" / "archive"
TRAIN_DIR = ARCHIVE_DIR / "train"
OUTPUT_PARQUET = BASE_DIR / "data" / "processed_sample.parquet"

# Mapa exacto de los 3 blobs confirmados (ruta interna -> archivo destino)
TARGET_BLOBS = {
    "data/files/md5/39/64d04db9158b0dfbe7c3a73e6b2245": (
        ARCHIVE_DIR / "training_labels.csv"
    ),
    "data/files/md5/e8/35bec8cd6f6f52ac2d644a06cd0873": (
        TRAIN_DIR / "0000.msgpack.zst"
    ),
    "data/files/md5/32/cb536142c1127b3132b06bf728a5f8": (
        TRAIN_DIR / "0001.msgpack.zst"
    ),
}

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB para no saturar memoria RAM


def extract_required_blobs() -> bool:
    """Busca y extrae los 3 blobs requeridos desde los archivos .zip."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted(list(ARCHIVE_DIR.glob("*.zip")))

    if not zips:
        print(f"[Error] No se encontraron archivos .zip en {ARCHIVE_DIR}")
        return False

    print("=" * 65)
    print("PASO 1: EXTRACCIÓN DE BLOBS VERIFICADOS DE DATA/ARCHIVE")
    print("=" * 65)

    pending_blobs = dict(TARGET_BLOBS)

    # Verificar si ya existen en disco para no reescribir innecesariamente
    for blob_name, dest_path in list(pending_blobs.items()):
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"-> Ya existe en disco: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
            del pending_blobs[blob_name]

    if not pending_blobs:
        print("-> Todos los archivos requeridos ya están presentes en disco.")
        return True

    # Buscar en los zips
    for zip_path in zips:
        if not pending_blobs:
            break

        print(f"\nRevisando: {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            zip_members = set(z.namelist())

            for blob_name, dest_path in list(pending_blobs.items()):
                # Normalizar barras para compatibilidad Windows/Linux
                match = None
                if blob_name in zip_members:
                    match = blob_name
                else:
                    alt_name = blob_name.replace("/", "\\")
                    if alt_name in zip_members:
                        match = alt_name

                if match:
                    print(f"  -> Extrayendo '{dest_path.name}' ({z.getinfo(match).file_size:,} bytes)...")
                    with z.open(match) as src, open(dest_path, "wb") as dst:
                        while chunk := src.read(CHUNK_SIZE):
                            dst.write(chunk)
                    print(f"     [OK] Guardado en: {dest_path}")
                    del pending_blobs[blob_name]

    if pending_blobs:
        print(f"[Error] Faltaron por extraer: {[p.name for p in pending_blobs.values()]}")
        return False

    return True


def generate_parquet():
    """Ejecuta el aplanado de datos y genera el Parquet oficial."""
    print("\n" + "=" * 65)
    print("PASO 2: APLANADO DE FRAMES Y GENERACIÓN DE PARQUET")
    print("=" * 65)

    input_files = [
        str(TRAIN_DIR / "0000.msgpack.zst"),
        str(TRAIN_DIR / "0001.msgpack.zst"),
    ]
    labels_csv = str(ARCHIVE_DIR / "training_labels.csv")

    df = process_dataset(
        file_paths=input_files,
        labels_csv_path=labels_csv,
        output_path=str(OUTPUT_PARQUET),
        max_frames_per_file=None,
    )

    print("\n" + "=" * 65)
    print("PASO 3: VERIFICACIÓN FINAL DE LA MUESTRA PROCESADA")
    print("=" * 65)
    print(f"Archivo generado: {OUTPUT_PARQUET}")
    print(f"Dimensiones (filas, columnas): {df.shape}")
    print(f"Partidas únicas reconstruidas (match_id): {df['match_id'].nunique()}")
    print("\nDistribución del target (outcome):")
    print(df["outcome"].value_counts(dropna=False))


def main():
    success = extract_required_blobs()
    if success:
        generate_parquet()
        print("\n[PROCESO TERMINADO CON ÉXITO]")
    else:
        print("\n[PROCESO FALLIDO] Revisa los errores anteriores.")


if __name__ == "__main__":
    main()