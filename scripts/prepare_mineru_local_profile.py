"""Prepare an isolated, cache-only MinerU retry profile; never execute parsing.

The existing sample's settings.mineru_model_source='modelscope' overrides an ENV
value, so use the emitted direct CLI command or explicitly change that SAMPLE
setting to 'local'. This script never modifies RefMind, its index or model cache.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REQUIRED = (
    "models/Layout/PP-DocLayoutV2/config.json",
    "models/Layout/PP-DocLayoutV2/model.safetensors",
    "models/Layout/PP-DocLayoutV2/preprocessor_config.json",
    "models/MFR/unimernet_hf_small_2503/config.json",
    "models/MFR/unimernet_hf_small_2503/model.safetensors",
    "models/MFR/unimernet_hf_small_2503/tokenizer.json",
    "models/OCR/paddleocr_torch/ch_PP-OCRv5_det_infer.pth",
    "models/OCR/paddleocr_torch/ch_PP-OCRv5_rec_infer.pth",
    "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx",
    "models/TabRec/SlanetPlus/slanet-plus.onnx",
    "models/TabRec/UnetStructure/unet.onnx",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-page", type=int, default=8)
    args = parser.parse_args()
    root, output, source = args.model_root.resolve(), args.output_dir.resolve(), args.source.resolve()
    if args.physical_page < 1 or not source.is_file():
        raise ValueError("A real source PDF and positive physical page are required")
    if output == root or root in output.parents:
        raise ValueError("Output cannot be placed inside the model cache")
    missing = [relative for relative in REQUIRED if not (root / relative).is_file() or (root / relative).stat().st_size <= 0]
    if missing:
        raise FileNotFoundError("Cache files missing: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    config = output / "mineru-local.json"
    config.write_text(json.dumps({"models-dir": {"pipeline": str(root)}}, indent=2), encoding="utf-8")
    profile = {
        "status": "prepared_not_executed", "cache_check": "required_files_present_not_hash_verified",
        "environment": {"MINERU_MODEL_SOURCE": "local", "MINERU_TOOLS_CONFIG_JSON": str(config),
            "MINERU_DEVICE_MODE": "cpu", "MINERU_FORMULA_CH_SUPPORT": "false",
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "command": [sys.executable, "-B", "-m", "mineru.cli.client", "-p", str(source),
            "-o", str(output / "page-output"), "-b", "pipeline", "-m", "auto", "-l", "ch_lite",
            "-s", str(args.physical_page - 1), "-e", str(args.physical_page - 1), "-f", "true", "-t", "true"],
        "required_files": [{"path": relative, "bytes": (root / relative).stat().st_size} for relative in REQUIRED],
        "notes": ["This profile only removes remote model-cache revision checks; it does not turn a CPU torch build into CUDA.",
            "CPU is explicit because the verified interpreter reported torch 2.12.0+cpu and cuda_available=False.",
            "Use a bounded one-page run with persistent stdout/stderr and process-tree cleanup before attempting all pages.",
            "Do not use the current parse_sample.py unchanged: its modelscope setting overwrites MINERU_MODEL_SOURCE.",
            "ch_lite matches the two OCR weights already cached; en or GPU default ch needs different recognition weights.",
            "A one-page run has diagnostic scope only and cannot validate the complete 16-page parse."]}
    (output / "retry-profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": profile["status"], "output": str(output), "files_checked": len(REQUIRED)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
