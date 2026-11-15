"""Inspeccion ligera de archivos de datos grandes, sin cargarlos completos.

Muestra: inventario de archivos, formato real, esquema de un registro,
tipos, nulos y una muestra. No escribe nada en disco.

Uso (desde la raiz del proyecto):
    python src/inspect_dataset.py --data-dir data
    python src/inspect_dataset.py --data-dir data --count
"""

import argparse
import time
import zipfile
from pathlib import Path

import msgpack
import pandas as pd
import pyarrow.parquet as pq
import zstandard

PEEK_ROWS = 5000
PEEK_RECORDS = 2
MAX_DEPTH = 5
ROOT_MAP_MIN_KEYS = 1000
MAX_KEYS_SHOWN = 15
MAGIC_BYTES = 16
MAX_FILES_LISTED = 40
MAX_ZIP_MEMBERS_LISTED = 12
PEEKABLE_KINDS = (".msgpack.zst", ".csv", ".csv.gz", ".tsv",
                  ".parquet")
SEPARATOR = "=" * 60


def human_size(num_bytes):
    """Return a human-readable size string (e.g. '1.9 GB')."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def file_kind(path):
    """Classify a file by its (possibly double) extension."""
    name = path.name.lower()
    for kind in (".msgpack.zst", ".csv.gz", ".csv", ".tsv", ".parquet",
                 ".zst", ".zip"):
        if name.endswith(kind):
            return kind
    return path.suffix.lower() or "(sin extension)"


def describe(obj, depth=0):
    """Return a compact multi-line description of a nested object."""
    pad = "  " * depth
    if depth >= MAX_DEPTH:
        return f"{pad}{type(obj).__name__} ..."
    if isinstance(obj, dict):
        lines = [f"{pad}dict con {len(obj)} claves:"]
        for key in list(obj)[:MAX_KEYS_SHOWN]:
            value = obj[key]
            if isinstance(value, (dict, list, tuple)):
                lines.append(f"{pad}  {key!r}:")
                lines.append(describe(value, depth + 2))
            else:
                lines.append(
                    f"{pad}  {key!r}: {type(value).__name__} = "
                    f"{str(value)[:40]!r}"
                )
        if len(obj) > MAX_KEYS_SHOWN:
            lines.append(f"{pad}  ... ({len(obj) - MAX_KEYS_SHOWN} mas)")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        head = f"{pad}{type(obj).__name__} de largo {len(obj)}"
        if not obj:
            return head
        return head + ", primer elemento:\n" + describe(obj[0], depth + 1)
    return f"{pad}{type(obj).__name__} = {str(obj)[:60]!r}"


def print_inventory(data_dir):
    """Print the extension summary and the list of files found."""
    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    files = [p for p in files if ".git" not in p.parts
             and ".venv" not in p.parts]
    print(f"{SEPARATOR}\n1. INVENTARIO de {data_dir}\n{SEPARATOR}")
    if not files:
        print("No se encontraron archivos. Revisa la ruta o ejecuta "
              "'dvc pull'.")
        return files
    summary = {}
    for path in files:
        count, total = summary.get(file_kind(path), (0, 0))
        summary[file_kind(path)] = (count + 1, total + path.stat().st_size)
    for kind, (count, total) in sorted(summary.items()):
        print(f"  {kind:<16} {count:>5} archivos  {human_size(total):>10}")
    print(f"\nPrimeros {MAX_FILES_LISTED} archivos:")
    for path in files[:MAX_FILES_LISTED]:
        print(f"  {human_size(path.stat().st_size):>10}  {path}")
    return files


def open_plain(path):
    """Return a callable that opens a plain file in binary mode."""
    return lambda: path.open("rb")


def decoded_stream(opener):
    """Open a .zst source and return (raw_file, decompressed_stream)."""
    raw = opener()
    return raw, zstandard.ZstdDecompressor().stream_reader(raw)


def new_unpacker(stream):
    """Build a msgpack Unpacker over a decompressed stream."""
    return msgpack.Unpacker(
        file_like=stream, raw=False, strict_map_key=False
    )


def detect_layout(opener):
    """Return (layout, size) for the top-level msgpack object.

    layout is 'array' (one big list), 'map' (one big dict) or 'sequence'
    (many independent small records, e.g. one dict per frame).
    """
    raw, stream = decoded_stream(opener)
    try:
        first = stream.read(1)
    finally:
        raw.close()
    if not first:
        return "sequence", None
    byte = first[0]
    raw, stream = decoded_stream(opener)
    try:
        unpacker = new_unpacker(stream)
        if 0x90 <= byte <= 0x9F or byte in (0xDC, 0xDD):
            return "array", unpacker.read_array_header()
        if 0x80 <= byte <= 0x8F or byte in (0xDE, 0xDF):
            size = unpacker.read_map_header()
            if size > ROOT_MAP_MIN_KEYS:
                return "map", size
    finally:
        raw.close()
    return "sequence", None


def peek_msgpack_zst(opener, count_records):
    """Describe the first records of a streamed msgpack+zstd source."""
    layout, size = detect_layout(opener)
    raw, stream = decoded_stream(opener)
    try:
        unpacker = new_unpacker(stream)
        if layout == "array":
            unpacker.read_array_header()
            print(f"Estructura: UN array con {size:,} elementos "
                  "(se leen de a uno).")
            iterator = (unpacker.unpack() for _ in range(size))
        elif layout == "map":
            unpacker.read_map_header()
            print(f"Estructura: UN diccionario raiz con {size:,} claves.")
            iterator = (
                {unpacker.unpack(): unpacker.unpack()}
                for _ in range(size)
            )
        else:
            print("Estructura: secuencia de registros independientes.")
            iterator = iter(unpacker)
        for index, record in enumerate(iterator):
            if index >= PEEK_RECORDS:
                break
            print(f"\n--- Registro {index} ---\n{describe(record)}")
    finally:
        raw.close()
    if count_records and layout == "sequence":
        started = time.time()
        raw, stream = decoded_stream(opener)
        try:
            total = sum(1 for _ in new_unpacker(stream))
        finally:
            raw.close()
        print(f"\nRegistros (frames): {total:,} "
              f"({time.time() - started:.0f} s)")


def peek_dataframe(frame):
    """Print dtypes, nulls and a head for a sample DataFrame."""
    print(f"Muestra: {frame.shape[0]:,} filas x {frame.shape[1]} columnas")
    print("\nTipos de dato:")
    print(frame.dtypes.to_string())
    nulls = frame.isna().sum()
    print("\nNulos en la muestra:")
    print(nulls[nulls > 0].to_string() if nulls.any() else "  ninguno")
    print("\nPrimeras 3 filas (transpuestas):")
    print(frame.head(3).T.to_string())


def peek_source(kind, opener, count_records):
    """Inspect one data source (plain file or zip member) by its kind."""
    if kind == ".msgpack.zst":
        peek_msgpack_zst(opener, count_records)
    elif kind in (".csv", ".csv.gz", ".tsv"):
        separator = "\t" if kind == ".tsv" else ","
        compression = "gzip" if kind == ".csv.gz" else None
        with opener() as handle:
            frame = pd.read_csv(handle, nrows=PEEK_ROWS, sep=separator,
                                compression=compression)
        peek_dataframe(frame)
    elif kind == ".parquet":
        parquet_file = pq.ParquetFile(opener())
        meta = parquet_file.metadata
        print(f"Filas totales: {meta.num_rows:,}  "
              f"Row groups: {meta.num_row_groups}")
        batch = next(parquet_file.iter_batches(batch_size=PEEK_ROWS))
        peek_dataframe(batch.to_pandas())
    else:
        with opener() as handle:
            print("Primeros bytes (hex):", handle.read(MAGIC_BYTES).hex())
        print("Formato no reconocido: pegame esta salida y lo vemos.")


def zip_members(path):
    """Return the non-directory entries of a zip (reads only the index)."""
    with zipfile.ZipFile(path) as archive:
        return [i for i in archive.infolist() if not i.is_dir()]


def print_zip_overview(zip_paths):
    """One block per zip: what is inside, without extracting anything."""
    print(f"\n{SEPARATOR}\n1b. CONTENIDO DE CADA .zip\n{SEPARATOR}")
    for path in zip_paths:
        members = zip_members(path)
        by_kind = {}
        for info in members:
            count, total = by_kind.get(file_kind(Path(info.filename)),
                                       (0, 0))
            by_kind[file_kind(Path(info.filename))] = (
                count + 1, total + info.file_size)
        print(f"\n{path.name}: {len(members)} archivos dentro")
        for kind, (count, total) in sorted(by_kind.items()):
            print(f"  {kind:<16} {count:>5}  {human_size(total):>10} "
                  "(descomprimido)")
        for info in members[:MAX_ZIP_MEMBERS_LISTED]:
            print(f"    {human_size(info.file_size):>10}  {info.filename}")
        if len(members) > MAX_ZIP_MEMBERS_LISTED:
            print(f"    ... ({len(members) - MAX_ZIP_MEMBERS_LISTED} mas)")


def peek_zip(path, count_records):
    """Peek the first readable member of a zip, streaming from inside."""
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            kind = file_kind(Path(info.filename))
            if info.is_dir() or kind not in PEEKABLE_KINDS:
                continue
            print(f"Miembro inspeccionado: {info.filename} ({kind}, "
                  f"{human_size(info.file_size)} descomprimido)")
            peek_source(kind, lambda name=info.filename: archive.open(name),
                        count_records)
            return
    print("El zip no contiene formatos reconocidos (msgpack.zst/csv/"
          "parquet).")


def peek_file(path, count_records):
    """Dispatch the inspection according to the file kind."""
    kind = file_kind(path)
    print(f"\n{SEPARATOR}\n2. INSPECCION de {path.name} ({kind}, "
          f"{human_size(path.stat().st_size)})\n{SEPARATOR}")
    if kind == ".zip":
        peek_zip(path, count_records)
    else:
        peek_source(kind, open_plain(path), count_records)


def main():
    """Entry point: inventory, then peek the first file of each kind."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--count", action="store_true",
                        help="contar registros (lento: lee todo el archivo)")
    args = parser.parse_args()
    files = print_inventory(args.data_dir)
    print_zip_overview([p for p in files if file_kind(p) == ".zip"])
    seen_kinds = set()
    for path in files:
        kind = file_kind(path)
        if kind in seen_kinds or kind in (".dvc", ".md", ".txt", ".json"):
            continue
        seen_kinds.add(kind)
        peek_file(path, args.count)


if __name__ == "__main__":
    main()