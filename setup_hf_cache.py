"""
setup_hf_cache.py
Downloads the HuggingFace models used by the transformer
pipeline into a local HF cache

requires: pip install huggingface_hub
"""

import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
while _p.name != "pepcube_property" and _p != _p.parent:
    _p = _p.parent
if _p.name == "pepcube_property" and str(_p.parent) not in sys.path:
    sys.path.insert(0, str(_p.parent))

import pepcube_property.config as config
MODELS_TO_FETCH = [
    ("DeepChem/ChemBERTa-77M-MTR", "fc007d31c2fb774ab7a8e5a8d318e25cb01d2da1",
     "ChemBERTa-77M-MTR (weights + tokenizer)"),
    ("DeepChem/ChemBERTa-77M-MLM", "d62f784b9a0a3aab09c788a7fb95a8e1ce89116f",
     "ChemBERTa-77M-MLM (weights only, PepDoRA base model)"),
    ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
     "ChemBERTa-77M-MLM (tokenizer files, PepDoRA base tokenizer)"),
    ("ChatterjeeLab/PepDoRA", "e034544e8f2ab1c34fffcfd4984f4183db7f12ed",
     "PepDoRA (LoRA/DoRA adapter)"),
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    cache_root = config.HF_HUB_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"HF cache root: {cache_root}\n")

    for repo_id, revision, label in MODELS_TO_FETCH:
        print(f"Fetching {label}")
        print(f"  repo: {repo_id}  revision: {revision}")
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_root),
        )
        print(f"  -> {local_path}\n")

    print("Done. Verifying expected paths exist ...")
    expected = [
        cache_root / "models--DeepChem--ChemBERTa-77M-MTR" / "snapshots"
                   / "fc007d31c2fb774ab7a8e5a8d318e25cb01d2da1",
        cache_root / "models--DeepChem--ChemBERTa-77M-MLM" / "snapshots"
                   / "d62f784b9a0a3aab09c788a7fb95a8e1ce89116f",
        cache_root / "models--DeepChem--ChemBERTa-77M-MLM" / "snapshots"
                   / "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        cache_root / "models--ChatterjeeLab--PepDoRA" / "snapshots"
                   / "e034544e8f2ab1c34fffcfd4984f4183db7f12ed",
    ]
    all_ok = True
    for path in expected:
        ok = path.exists()
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISSING'}] {path}")

    if all_ok:
        print("\nAll expected snapshots downloaded successfully")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
