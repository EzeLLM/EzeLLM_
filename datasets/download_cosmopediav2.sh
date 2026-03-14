#!/bin/bash
# Download cosmopedia-v2 subset from HuggingFaceTB/smollm-corpus
# Target: 104 parquet files, ~122 GB total
# Destination: /media/data1tb/datasets/smollm-corpus-cosmopediav2

set -e

# HF download settings (same workarounds that fixed model downloads)
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

DEST="/media/data1tb/datasets/smollm-corpus-cosmopediav2"
REPO="HuggingFaceTB/smollm-corpus"
SUBSET="cosmopedia-v2"

echo "============================================"
echo " Downloading: $REPO / $SUBSET"
echo " Destination: $DEST"
echo " Files: 104 parquet shards (~122 GB)"
echo "============================================"
echo ""
echo "Free space on target disk:"
df -h /media/data1tb
echo ""

# Download using hf CLI with conservative settings
hf download "$REPO" \
    --repo-type dataset \
    --include "${SUBSET}/*" \
    --local-dir "$DEST" \
    --max-workers 1

echo ""
echo "============================================"
echo " Download complete!"
echo "============================================"
echo ""

# Verify
TOTAL_FILES=$(find "$DEST/$SUBSET" -name "*.parquet" 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$DEST/$SUBSET" 2>/dev/null | cut -f1)
echo "Files downloaded: $TOTAL_FILES / 104"
echo "Total size: $TOTAL_SIZE"

if [ "$TOTAL_FILES" -eq 104 ]; then
    echo "ALL FILES PRESENT"
else
    echo "WARNING: Expected 104 files, got $TOTAL_FILES. Some files may be missing."
fi
