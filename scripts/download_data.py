"""Fetch only the documented Amazon subset from the UCI archive."""

from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
EXPECTED = "47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3"


def main():
    with urllib.request.urlopen(URL, timeout=30) as response:
        archive = response.read(5_000_000)
    raw = zipfile.ZipFile(io.BytesIO(archive)).read(
        "sentiment labelled sentences/amazon_cells_labelled.txt"
    )
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError(
            "Source changed; review provenance before replacing pinned data"
        )
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/amazon.tsv").write_bytes(raw)
    print("Verified UCI Amazon subset downloaded")


if __name__ == "__main__":
    main()
