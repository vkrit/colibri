"""
Download of the real GLM-5.2 weights for the C engine — STAGE B.

Target: zai-org/GLM-5.2-FP8  (FP8 e4m3, 141 shards, ~756 GB) -> FITS in the 926 GB of ext4.
(The bf16 variant zai-org/GLM-5.2 is 1.5 TB and does NOT fit.)

The C engine will read these safetensors in a streaming fashion and (re)quantize them to
int4/int8. Note: the weights are F8_E4M3 + `*.weight_scale_inv` tensors (128x128 blocks).
The st.h loader must support fp8+block-scale before they can be used (see memory glm52-specs).

USAGE:
    python3 tools/download_glm52.py            # download everything into /home/vincenzo/glm52  (resumable)
    python3 tools/download_glm52.py --check    # space estimate and file count only, no download

The download is hundreds of GB and hours: launch it yourself when the rest is ready.
"""
import os, sys, shutil
from huggingface_hub import snapshot_download, HfApi

REPO = "zai-org/GLM-5.2-FP8"
DEST = os.environ.get("GLM_DIR", "/home/vincenzo/glm52")   # on ext4 (/dev/sdd), NEVER on /mnt/c

def human(n): return f"{n/1e9:.0f} GB"

def check():
    info = HfApi().repo_info(REPO, files_metadata=True)
    tot = sum((s.size or 0) for s in info.siblings)
    sts = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    free = shutil.disk_usage(os.path.dirname(DEST) or "/").free
    print(f"repo: {REPO}")
    print(f"  total files: {len(info.siblings)} ({len(sts)} safetensors shards)")
    print(f"  total size: {human(tot)}")
    print(f"  free space in {DEST}: {human(free)}")
    print(f"  {'OK: enough space' if free > tot*1.05 else 'WARNING: not enough space'}")

def download():
    os.makedirs(DEST, exist_ok=True)
    free = shutil.disk_usage(DEST).free
    print(f"Downloading {REPO} -> {DEST}  (free: {human(free)})")
    # resume_download is implicit; on interruption, relaunch and it resumes.
    snapshot_download(
        repo_id=REPO,
        local_dir=DEST,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        max_workers=8,
    )
    print("DONE. Weights saved in:", DEST)

if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        check(); print("---"); download()
