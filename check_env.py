"""
check_env.py
─────────────
Read-only environment audit for the pipeline.
Verifies that every Python package, CUDA capability, and external
binary required to run Phase 1 + Phase 2 is correctly installed.

Does NOT install anything. Does NOT modify the environment.
Just reports what is missing and prints fix commands for each issue.

Usage:
    python check_env.py
"""

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple


# ─── Output helpers ──────────────────────────────────────────────────────────

class C:
    OK   = "\033[92m"
    WARN = "\033[93m"
    ERR  = "\033[91m"
    BOLD = "\033[1m"
    END  = "\033[0m"

def ok(msg):    print(f"  {C.OK} {C.END}  {msg}")
def warn(msg):  print(f"  {C.WARN} {C.END}  {msg}")
def fail(msg):  print(f"  {C.ERR} {C.END}  {msg}")
def header(msg):
    print(f"\n{C.BOLD}{'─' * 70}{C.END}")
    print(f"{C.BOLD}  {msg}{C.END}")
    print(f"{C.BOLD}{'─' * 70}{C.END}")


# ─── Bookkeeping ─────────────────────────────────────────────────────────────

ISSUES = []   # list of (severity, message, fix_command)

def record(severity: str, msg: str, fix: str = ""):
    ISSUES.append((severity, msg, fix))


# ─── 1. Python version ───────────────────────────────────────────────────────

def check_python():
    header("1. Python interpreter")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        ok(f"Python {version_str} ({sys.executable})")
    else:
        fail(f"Python {version_str} — need 3.10 or newer")
        record("CRITICAL",
               f"Python version is {version_str}, need ≥3.10",
               "Install Python 3.10+ via your system package manager")


# ─── 2. Required packages ────────────────────────────────────────────────────

# (import_name, pip_name, min_version, severity)
REQUIRED_PACKAGES = [
    # Core ML stack
    ("torch",                "torch",                "2.3.0",  "CRITICAL"),
    ("transformers",         "transformers",         "4.49.0", "CRITICAL"),
    ("accelerate",           "accelerate",           "0.30.0", "CRITICAL"),

    # Phase 1 multimodal — Qwen2.5-VL
    ("qwen_vl_utils",        "qwen-vl-utils",        "0.0.8",  "CRITICAL"),

    # Phase 2 + OCR baseline — llama-cpp-python 
    ("llama_cpp",            "llama-cpp-python",     "0.3.0",  "CRITICAL"),

    # Phase 1 baseline — PaddleOCR
    ("paddle",               "paddlepaddle-gpu",     "2.6.0",  "HIGH"),
    ("paddleocr",            "paddleocr",            "2.7.0",  "HIGH"),

    # Frontier baselines 
    ("google.generativeai",  "google-generativeai",  "0.7.0",  "MEDIUM"),
    ("openai",               "openai",               "1.40.0", "MEDIUM"),

    # PDF + data
    ("fitz",                 "PyMuPDF",              "1.24.0", "CRITICAL"),
    ("pandas",               "pandas",               "2.2.0",  "CRITICAL"),
    ("numpy",                "numpy",                "1.26.0", "CRITICAL"),
    ("PIL",                  "Pillow",               "10.0.0", "CRITICAL"),
]


def parse_version(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    except Exception:
        return (0, 0, 0)


def check_packages():
    header("2. Python packages")
    for import_name, pip_name, min_v, severity in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
        except ImportError:
            fail(f"{pip_name:<25} NOT INSTALLED")
            fix = (f'CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir {pip_name}'
                   if pip_name == "llama-cpp-python"
                   else f"pip install '{pip_name}>={min_v}'")
            record(severity, f"Missing package: {pip_name}", fix)
            continue
        except Exception as e:
            fail(f"{pip_name:<25} import error: {e}")
            record(severity, f"Package {pip_name} fails to import: {e}",
                   f"pip uninstall -y {pip_name} && pip install '{pip_name}>={min_v}'")
            continue

        # Determine installed version
        try:
            installed = importlib.metadata.version(pip_name)
        except importlib.metadata.PackageNotFoundError:
            installed = getattr(mod, "__version__", "unknown")

        if installed == "unknown":
            warn(f"{pip_name:<25} installed, version unknown")
            continue

        cur = parse_version(installed)
        need = parse_version(min_v)
        if cur >= need:
            ok(f"{pip_name:<25} {installed}")
        else:
            warn(f"{pip_name:<25} {installed} (need ≥ {min_v})")
            record(severity,
                   f"{pip_name} version {installed} < required {min_v}",
                   f"pip install -U '{pip_name}>={min_v}'")


# ─── 3. CUDA + GPU ───────────────────────────────────────────────────────────

def check_cuda():
    header("3. CUDA + GPU availability")

    # nvidia-smi 
    if shutil.which("nvidia-smi") is None:
        fail("nvidia-smi binary not found in PATH")
        record("CRITICAL",
               "nvidia-smi not available — driver may be missing",
               "Install NVIDIA driver; confirm with `nvidia-smi`")
        return

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        for line in out.splitlines():
            idx, name, driver, total, free = [x.strip() for x in line.split(",")]
            ok(f"GPU {idx}: {name} | driver {driver} | "
               f"{int(free):,} MB free / {int(total):,} MB total")
    except subprocess.CalledProcessError as e:
        fail(f"nvidia-smi failed: {e}")
        record("CRITICAL", "nvidia-smi runtime error",
               "Check GPU driver installation")
        return

    # PyTorch CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            fail("PyTorch reports CUDA NOT available")
            record("CRITICAL",
                   "PyTorch was built without CUDA support",
                   "pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            cuda_ver = torch.version.cuda
            cudnn    = torch.backends.cudnn.version()
            n_gpus   = torch.cuda.device_count()
            ok(f"PyTorch CUDA backend: {cuda_ver}, cuDNN {cudnn}, {n_gpus} device(s) visible")
    except ImportError:
        fail("PyTorch not installed — cannot verify CUDA backend")


# ─── 4. llama-cpp-python CUDA support ────────────────────────────────────────

def check_llama_cpp_cuda():
    header("4. llama-cpp-python — CUDA offload support")
    try:
        from llama_cpp import llama_cpp
    except ImportError:
        fail("llama-cpp-python not importable — skipping CUDA check")
        return


    try:
        gpu_offload = bool(llama_cpp.llama_supports_gpu_offload())
        if gpu_offload:
            ok("llama-cpp-python built WITH GPU offload support")
        else:
            fail("llama-cpp-python built WITHOUT GPU offload (CPU-only build)")
            record("CRITICAL",
                   "llama-cpp-python lacks CUDA — Generator/Validator will run on CPU (unusable)",
                   'pip uninstall -y llama-cpp-python && '
                   'CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python')
    except AttributeError:
        warn("Could not query GPU-offload support (older llama-cpp version) — "
             "verify manually by loading a model with n_gpu_layers=-1")


# ─── 5. Pipeline source layout ───────────────────────────────────────────────

EXPECTED_FILES = [
    "constants.py",
    "model.py",
    "extract_use_cases.py",
    "dedup_use_cases.py",
    "pdf_chunked.py",
    "data_split.py",
    "test_smoke.py",
    "run_pipeline.py",
    "compute_metrics.py",
    "agents/generator.py",
    "agents/validator.py",
    "agents/orchestrator.py",
    "baselines/ocr_llm.py",
    "baselines/frontier.py",
    "evaluation/sampling.py",
    "evaluation/metrics.py",
    "prompts/tc_prompts.py",
]


def check_pipeline_files():
    header("5. Pipeline source files")
    missing = []
    for rel in EXPECTED_FILES:
        if os.path.isfile(rel):
            ok(rel)
        else:
            fail(f"MISSING — {rel}")
            missing.append(rel)
    if missing:
        record("CRITICAL",
               f"{len(missing)} pipeline source file(s) missing",
               "Re-extract the pipeline_v2.zip in the working directory")


# ─── 6. Model files & input PDF ──────────────────────────────────────────────

def check_model_paths():
    header("6. Model files & input PDF")
    try:
        from constants import (
            INPUT_PDF, MODEL_EXTRACTION,
            MODEL_GENERATOR_GGUF, MODEL_OCR_LLM_GGUF,
            MODEL_VALIDATOR_GGUF,
        )
    except ImportError as e:
        fail(f"Cannot import constants.py — {e}")
        return

    paths = [
        ("Input PDF",                          INPUT_PDF),
        ("Qwen2.5-VL (Phase 1 multimodal)",    MODEL_EXTRACTION),
        ("Llama-3.1-70B IQ2_M (Generator)",    MODEL_GENERATOR_GGUF),
        ("Llama-3.1-70B IQ2_M (OCR baseline)", MODEL_OCR_LLM_GGUF),
        ("Llama-3.3-70B Q3_K_L (Validator)",   MODEL_VALIDATOR_GGUF),
    ]

    for label, path in paths:
        if os.path.exists(path):
            size_gb = (os.path.getsize(path) / 1e9
                       if os.path.isfile(path) else None)
            if size_gb is not None:
                ok(f"{label:<40} {path}  ({size_gb:.1f} GB)")
            else:
                ok(f"{label:<40} {path}  (directory)")
        else:
            fail(f"{label:<40} {path}  (NOT FOUND)")
            record("HIGH",
                   f"Missing: {label} at {path}",
                   "Verify the path in constants.py or finish the model upload")


# ─── 7. API keys ─────────────

def check_api_keys():
    header("7. API keys (optional — only if running frontier baselines locally)")
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(var):
            ok(f"{var} is set")
        else:
            warn(f"{var} is NOT set (OK if collaborator runs baselines)")


# ─── 8. Disk space ───────────────────────────────────────────────────────────

def check_disk_space():
    header("8. Disk space")
    cwd = os.getcwd()
    total, used, free = shutil.disk_usage(cwd)
    free_gb = free / 1e9
    total_gb = total / 1e9
    msg = f"{cwd}: {free_gb:.0f} GB free / {total_gb:.0f} GB total"
    if free_gb >= 50:
        ok(msg)
    elif free_gb >= 20:
        warn(f"{msg} — tight (recommend ≥50 GB for model files + results)")
    else:
        fail(f"{msg} — insufficient")
        record("HIGH", "Less than 20 GB free disk",
               "Free up disk space before running the pipeline")


# ─── Summary ─────────────────────────────────────────────────────────────────

def summary():
    header("SUMMARY")
    if not ISSUES:
        print(f"\n  {C.OK}{C.BOLD} Environment is READY to run the pipeline.{C.END}\n")
        print("  Next step:  python test_smoke.py")
        return 0

    critical = [i for i in ISSUES if i[0] == "CRITICAL"]
    high     = [i for i in ISSUES if i[0] == "HIGH"]
    medium   = [i for i in ISSUES if i[0] == "MEDIUM"]

    print(f"\n  Total issues: {len(ISSUES)} "
          f"({len(critical)} critical, {len(high)} high, {len(medium)} medium)\n")

    for sev_label, color, group in (
        ("CRITICAL", C.ERR,  critical),
        ("HIGH",     C.WARN, high),
        ("MEDIUM",   C.WARN, medium),
    ):
        if not group:
            continue
        print(f"  {color}{C.BOLD}[{sev_label}]{C.END}")
        for _, msg, fix in group:
            print(f"    • {msg}")
            if fix:
                print(f"      → fix: {fix}")
        print()

    if critical:
        print(f"  {C.ERR}{C.BOLD} DO NOT RUN the pipeline yet.{C.END}")
        return 2
    print(f"  {C.WARN}{C.BOLD} Pipeline will run, but with reduced capability.{C.END}")
    return 1


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"{C.BOLD}══════════════════════════════════════════════════════════════════════")
    print(f"  PIPELINE ENVIRONMENT AUDIT — read-only")
    print(f"══════════════════════════════════════════════════════════════════════{C.END}")

    check_python()
    check_packages()
    check_cuda()
    check_llama_cpp_cuda()
    check_pipeline_files()
    check_model_paths()
    check_api_keys()
    check_disk_space()

    sys.exit(summary())


if __name__ == "__main__":
    main()
