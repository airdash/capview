#!/usr/bin/env python3
"""Convert a RIFE (Real-Time Intermediate Flow Estimation) PyTorch model to ONNX.

Usage:
    # 1. Clone the RIFE repo:
    git clone https://github.com/hzwer/Practical-RIFE
    cd Practical-RIFE

    # 2. Download a model (e.g., RIFE v4.6):
    #    Check https://github.com/hzwer/Practical-RIFE/tree/main/model
    #    or download from Google Drive links in README

    # 3. Run this script:
    python convert_rife_to_onnx.py --model-dir train_log --output rife.onnx

    # 4. Place the ONNX model:
    mkdir -p ~/.config/capview
    cp rife.onnx ~/.config/capview/rife.onnx

Requirements:
    pip install torch onnx onnxruntime
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert RIFE model to ONNX")
    parser.add_argument("--model-dir", default="train_log",
                        help="Directory containing RIFE model files (flownet.pkl)")
    parser.add_argument("--output", default="rife.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--width", type=int, default=1920,
                        help="Input width (must be divisible by 32)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Input height (must be divisible by 32)")
    parser.add_argument("--opset", type=int, default=16,
                        help="ONNX opset version")
    args = parser.parse_args()

    # Pad dimensions to multiple of 32
    w = ((args.width + 31) // 32) * 32
    h = ((args.height + 31) // 32) * 32

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    # Try to import RIFE's IFNet
    # The script should be run from the Practical-RIFE directory
    sys.path.insert(0, os.getcwd())
    try:
        from model.RIFE import Model
    except ImportError:
        try:
            from train_log.RIFE import Model
        except ImportError:
            print("Error: Cannot import RIFE model. Run this script from the Practical-RIFE directory.",
                  file=sys.stderr)
            print("  git clone https://github.com/hzwer/Practical-RIFE", file=sys.stderr)
            print("  cd Practical-RIFE", file=sys.stderr)
            print("  python /path/to/convert_rife_to_onnx.py", file=sys.stderr)
            sys.exit(1)

    print(f"Loading RIFE model from {args.model_dir}...")
    device = torch.device("cpu")

    model = Model()
    model.load_model(args.model_dir, -1)
    model.eval()
    model.device()

    # The IFNet takes two images concatenated along channel dim
    # Input: img0 [1, 3, H, W], img1 [1, 3, H, W]
    # We export with named inputs for clarity
    class RIFEWrapper(nn.Module):
        def __init__(self, flownet):
            super().__init__()
            self.flownet = flownet

        def forward(self, img0, img1):
            # RIFE's inference expects imgs concatenated + timestep
            return self.flownet(img0, img1, timestep=0.5)[0]

    wrapper = RIFEWrapper(model.flownet)
    wrapper.eval()

    print(f"Exporting to ONNX (input: {w}x{h}, opset {args.opset})...")
    dummy_img0 = torch.randn(1, 3, h, w)
    dummy_img1 = torch.randn(1, 3, h, w)

    torch.onnx.export(
        wrapper,
        (dummy_img0, dummy_img1),
        args.output,
        input_names=["img0", "img1"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes={
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({size_mb:.1f} MB)")
    print(f"\nTo use with capview:")
    print(f"  mkdir -p ~/.config/capview")
    print(f"  cp {args.output} ~/.config/capview/rife.onnx")
    print(f"  capview --device /dev/video0 --features rife")
    print(f"\nOr set CAPVIEW_RIFE_MODEL={os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
