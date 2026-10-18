import os
import glob

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import hf_hub_download

dest = "/home/administrator/models/wan22/google/umt5-xxl"
os.makedirs(dest, exist_ok=True)
for f in ["tokenizer.json", "spiece.model", "tokenizer_config.json", "config.json"]:
    p = hf_hub_download(repo_id="google/umt5-xxl", filename=f, local_dir=dest)
    print("ok", f, p)
for f in sorted(glob.glob(dest + "/**", recursive=True)):
    print("tree:", f)
