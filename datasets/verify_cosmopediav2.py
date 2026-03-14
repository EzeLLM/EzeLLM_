"""Verify the downloaded cosmopedia-v2 dataset integrity."""
import os
import glob
import pyarrow.parquet as pq

DATA_DIR = "/media/data1tb/datasets/smollm-corpus-cosmopediav2/cosmopedia-v2"
EXPECTED_SHARDS = 104
EXPECTED_COLUMNS = {"prompt", "text", "token_length", "audience", "format", "seed_data"}

def verify():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"Found {len(files)} / {EXPECTED_SHARDS} parquet files\n")

    if len(files) != EXPECTED_SHARDS:
        # Check which ones are missing
        expected = {f"train-{i:05d}-of-00104.parquet" for i in range(EXPECTED_SHARDS)}
        actual = {os.path.basename(f) for f in files}
        missing = sorted(expected - actual)
        if missing:
            print(f"MISSING {len(missing)} files:")
            for m in missing:
                print(f"  - {m}")
            print()

    total_rows = 0
    total_bytes = 0
    bad_files = []

    for f in files:
        try:
            pf = pq.ParquetFile(f)
            rows = pf.metadata.num_rows
            size = os.path.getsize(f)
            cols = set(pf.schema.names)
            total_rows += rows
            total_bytes += size

            if not EXPECTED_COLUMNS.issubset(cols):
                bad_files.append((f, f"missing columns: {EXPECTED_COLUMNS - cols}"))
        except Exception as e:
            bad_files.append((f, str(e)))

    print(f"Total rows:  {total_rows:,}")
    print(f"Total size:  {total_bytes / (1024**3):.2f} GB")
    print(f"Corrupted/bad files: {len(bad_files)}")

    if bad_files:
        print("\nProblematic files:")
        for path, reason in bad_files:
            print(f"  {os.path.basename(path)}: {reason}")
    else:
        print("\nAll files OK!")

if __name__ == "__main__":
    verify()
