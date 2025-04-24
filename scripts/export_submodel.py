# scripts/export_submodel.py
import argparse
import logging
import traceback
from pathlib import Path
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from steganogan.models import SteganoGAN # noqa: E402
# Import specific decoder classes if needed for type hinting or direct use
from steganogan.decoders import DenseDecoder, BasicDecoder # noqa: E402

def configure_logging() -> None:
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root_logger = logging.getLogger()
    # Clear previous handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# --- Define Sub-Models ---

class DecoderStage1(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        # Assuming DenseDecoder structure for now
        # Adapt if using BasicDecoder
        if not isinstance(original_decoder, DenseDecoder):
             raise TypeError("This submodel structure assumes DenseDecoder")
        self.conv1 = original_decoder.conv1
    def forward(self, x):
        return self.conv1(x)

class DecoderStage12(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        if not isinstance(original_decoder, DenseDecoder):
             raise TypeError("This submodel structure assumes DenseDecoder")
        self.conv1 = original_decoder.conv1
        self.conv2 = original_decoder.conv2
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2 # Output after stage 2

class DecoderStage123(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        if not isinstance(original_decoder, DenseDecoder):
             raise TypeError("This submodel structure assumes DenseDecoder")
        self.conv1 = original_decoder.conv1
        self.conv2 = original_decoder.conv2
        self.conv3 = original_decoder.conv3
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        return x3 # Output after stage 3

# --- Export Function ---

def export_submodel_to_onnx(
    steg_path_or_arch: str,
    stage_key: str, # e.g., "s1", "s12", "s123", "s1234"
    output_onnx_path: Path,
    height: int,
    width: int
) -> None:
    """Loads the original decoder and exports the specified sub-model stage."""
    try:
        logging.info(f"--- Exporting Stage: {stage_key} ---")
        logging.info(f"Loading PyTorch model source: {steg_path_or_arch}")
        # Load original decoder
        if Path(steg_path_or_arch).is_file():
             steg = torch.load(steg_path_or_arch, map_location='cpu', weights_only=False)
        else:
             temp_gan = SteganoGAN.load(architecture=steg_path_or_arch, cuda=False, verbose=False)
             steg = temp_gan
        original_decoder = steg.decoder
        if hasattr(original_decoder, '_orig_mod'):
             original_decoder = original_decoder._orig_mod
        original_decoder.upgrade_legacy()
        original_decoder.eval()
        logging.info("Original PyTorch decoder loaded.")

        # Select and instantiate the correct sub-model
        if stage_key == "s1":
            sub_model = DecoderStage1(original_decoder).eval()
            output_name = "output_stage1"
        elif stage_key == "s12":
            sub_model = DecoderStage12(original_decoder).eval()
            output_name = "output_stage2"
        elif stage_key == "s123":
            sub_model = DecoderStage123(original_decoder).eval()
            output_name = "output_stage3"
        elif stage_key == "s1234": # Full model
            sub_model = original_decoder # Use the full decoder directly
            output_name = "output_stage4" # Or just 'output'
        else:
            raise ValueError(f"Invalid stage key: {stage_key}")
        logging.info(f"Sub-model for stage {stage_key} created.")

        # Export
        dummy_input = torch.randn(1, 3, height, width)
        logging.info(f"Exporting {stage_key} model to {output_onnx_path}...")
        torch.onnx.export(
            sub_model,
            dummy_input,
            str(output_onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=False,
            input_names=['input'],
            output_names=[output_name],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                          output_name: {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
        logging.info(f"Export complete for {stage_key}.")

    except Exception as e:
        logging.error(f"Export failed for stage {stage_key}: {e}")
        logging.error(traceback.format_exc())
        raise

# --- Main Execution ---
def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Export SteganoGAN decoder sub-models to ONNX.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--arch', type=str, help='PyTorch model architecture name (e.g., dense)')
    model_group.add_argument('--steg_path', type=str, help='Path to PyTorch .steg model file')

    parser.add_argument('--stage', type=str, required=True, choices=['s1', 's12', 's123', 's1234'], help='Which sub-model stage to export (e.g., s1, s12)')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=256, help='Input image height')
    parser.add_argument('--width', type=int, default=256, help='Input image width')

    args = parser.parse_args()
    model_id = args.steg_path if args.steg_path else args.arch

    export_submodel_to_onnx(
        steg_path_or_arch=model_id,
        stage_key=args.stage,
        output_onnx_path=Path(args.output),
        height=args.height,
        width=args.width
    )

if __name__ == '__main__':
    main()