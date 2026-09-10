#!/usr/bin/env python3

import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("Install it with: pip install torch")
    sys.exit(1)


def main():
    print("=== NVIDIA GPU Test ===\n")

    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ CUDA is NOT available to PyTorch.")
        print("Check your PyTorch/CUDA installation.")
        sys.exit(1)

    # GPU information
    gpu = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    print(f"GPU:              {gpu}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM:             {props.total_memory / 1024**3:.2f} GB")

    # Create matrices on GPU
    print("\nRunning GPU computation...")

    device = torch.device("cuda")

    size = 4096

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm up
    c = a @ b
    torch.cuda.synchronize()

    # Timed computation
    start = time.perf_counter()

    for _ in range(10):
        c = a @ b

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Verify result
    if torch.isfinite(c).all():
        print("Matrix multiplication: PASS")
    else:
        print("Matrix multiplication: FAIL")
        sys.exit(1)

    print(f"Matrix size:        {size} x {size}")
    print(f"10 GPU operations:  {elapsed:.3f} seconds")
    print(f"Average operation:  {elapsed / 10:.4f} seconds")

    print("\n✅ GPU is working correctly.")
    print(f"   {gpu} successfully performed CUDA computation.")


if __name__ == "__main__":
    main()

