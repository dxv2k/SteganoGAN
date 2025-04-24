import argparse
import logging
import traceback
from pathlib import Path
import os
import torch
from typing import Optional
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def export_decoder_to_onnx(arch: str, model_path: Optional[str], onnx_path: Path, height: int, width: int, opset_version: int) -> None:
    try:
        # Determine pretrained model file
        if model_path:
            steg_file = Path(model_path)
        else:
            # Derive from architecture
            from steganogan import models as _mod
            pretrained_dir = Path(os.path.dirname(_mod.__file__)) / 'pretrained'
            steg_file = pretrained_dir / f"{arch}.steg"
        logging.info(f"Loading raw SteganoGAN checkpoint from {steg_file}")
        steg = torch.load(steg_file, map_location='cpu', weights_only=False)
        # Raw decoder (uncompiled)
        decoder = steg.decoder
        if hasattr(decoder, '_orig_mod'):
             logging.info("Detected compiled decoder, using original module (_orig_mod).")
             decoder = decoder._orig_mod
        else:
             logging.info("Using decoder directly (appears uncompiled or PyTorch < 2.0).")
        decoder.upgrade_legacy()
        decoder.eval()

        logging.info("Preparing dummy input for ONNX export...")
        dummy_input = torch.randn(1, 3, height, width)
        logging.info(f"Exporting to ONNX: {onnx_path} (opset={opset_version})")
        torch.onnx.export(
            decoder,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                          'output': {0: 'batch_size', 2: 'height', 3: 'width'}}, 
            verbose=True
        )
        logging.info(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        logging.error(f"Export failed: {e}")
        traceback.print_exc()
        raise

def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Export SteganoGAN decoder to ONNX")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--arch', type=str, help='Model architecture name (e.g., dense, basic)')
    group.add_argument('--path', type=str, help='Path to pretrained .steg model file')
    parser.add_argument('--onnx', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version to use for export')
    parser.add_argument('--height', type=int, default=256, help='Input image height')
    parser.add_argument('--width', type=int, default=256, help='Input image width')
    args = parser.parse_args()
    export_decoder_to_onnx(args.arch, args.path, Path(args.onnx), args.height, args.width, args.opset)

if __name__ == '__main__':
    main() 