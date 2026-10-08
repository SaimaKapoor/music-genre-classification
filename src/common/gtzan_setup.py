"""Download and prepare the original GTZAN Genre Collection for this project.

The raw GTZAN audio is intentionally NOT stored in GitHub. This script downloads
it into the local/Colab runtime when needed.
"""
from pathlib import Path
import tarfile
import urllib.request

GTZAN_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
DEFAULT_ROOT = Path("/content/gtzan")


def ensure_gtzan(root=DEFAULT_ROOT):
    root = Path(root)
    candidates = [root / "genres", root]
    if any(p.exists() and any(p.rglob("*.au")) for p in candidates):
        print(f"GTZAN already available under {root}")
        return root

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "genres.tar.gz"
    print("Downloading GTZAN (~1.3 GB compressed)...")
    urllib.request.urlretrieve(GTZAN_URL, archive)
    print("Download complete. Extracting...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    archive.unlink(missing_ok=True)

    found = list(root.rglob("*.au")) + list(root.rglob("*.wav"))
    if not found:
        raise FileNotFoundError(
            f"Extraction completed but no audio files were found under {root}"
        )
    print(f"GTZAN ready: {len(found)} audio files found.")
    print(f"Root: {root}")
    return root


if __name__ == "__main__":
    ensure_gtzan()
