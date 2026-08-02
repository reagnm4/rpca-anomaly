#!/usr/bin/env bash
# Download + extract the UCSD Anomaly Detection Dataset (includes Ped2).
# Run on your local machine — ~700MB download.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
URL="https://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
ARCHIVE="$DATA_DIR/UCSD_Anomaly_Dataset.tar.gz"

mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/UCSD_Anomaly_Dataset.v1p2/UCSDped2" ]; then
  echo "Ped2 already present at $DATA_DIR/UCSD_Anomaly_Dataset.v1p2/UCSDped2 — nothing to do."
  exit 0
fi

if [ ! -f "$ARCHIVE" ]; then
  echo "Downloading UCSD Anomaly Dataset (~700MB)..."
  curl -L --fail -o "$ARCHIVE" "$URL"
fi

echo "Extracting..."
tar -xzf "$ARCHIVE" -C "$DATA_DIR"

echo "Done. Ped2 at: $DATA_DIR/UCSD_Anomaly_Dataset.v1p2/UCSDped2"
echo "You can delete the archive to reclaim space: rm '$ARCHIVE'"
