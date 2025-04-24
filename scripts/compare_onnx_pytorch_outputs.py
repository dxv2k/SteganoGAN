# scripts/compare_onnx_pytorch_outputs.py

import argparse
import logging
import traceback
from pathlib import Path
import os
import sys
from typing import Dict, Any

import torch
import onnxruntime as ort
import numpy as np
import imageio.v2 as iio # Use v2 explicitly to avoid deprecation warning

# Add project root to sys.path to allow importing steganogan
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from steganogan.models import SteganoGAN

def configure_logging(log_file_path: Path) -> None:
    """Configure logging to file and console."""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def preprocess_image_for_decode(image_path: Path, height: int, width: int) -> np.ndarray:
    """Preprocesses image exactly like PyTorch decode path."""
    try:
        # Use imageio and /255.0 normalization
        img = iio.imread(str(image_path), pilmode='RGB') / 255.0
        # Resize *after* normalization if needed (check if model expects this)
        # This part needs verification based on training - assuming resize happens first if needed
        # For simplicity, let's assume input image is already correct size for now
        # or resize externally before passing to this script.
        # If resizing is needed:
        # img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize((width, height))
        # img = np.array(img_pil) / 255.0

        img_np = img.astype(np.float32)
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dim (1, C, H, W)
        return img_np
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        traceback.print_exc()
        raise

def get_pytorch_stage_output(decoder: torch.nn.Module, input_tensor: torch.Tensor, stage: str) -> torch.Tensor:
    """Calculates the intermediate output of the PyTorch decoder for a given stage."""
    decoder.eval()
    with torch.no_grad():
        x1 = decoder.conv1(input_tensor)
        if stage == "stage1":
            return x1

        x2 = decoder.conv2(x1)
        if stage == "stage2":
            return x2

        # Assuming DenseDecoder structure
        x3 = decoder.conv3(torch.cat([x1, x2], dim=1))
        if stage == "stage3":
            return x3

        x4 = decoder.conv4(torch.cat([x1, x2, x3], dim=1))
        if stage == "stage4":
            return x4

    raise ValueError(f"Unknown or unsupported stage: {stage}")

def compare_stage_outputs(
    steg_path_or_arch: str,
    onnx_submodel_path: Path,
    image_path: Path,
    stage: str,
    height: int,
    width: int,
    log_file: Path
) -> None:
    """Loads models, runs inference, compares outputs for a specific stage, and logs."""

    configure_logging(log_file)
    logging.info("--- Starting Comparison ---")
    logging.info(f"PyTorch Model: {steg_path_or_arch}")
    logging.info(f"ONNX Sub-Model: {onnx_submodel_path}")
    logging.info(f"Input Image: {image_path}")
    logging.info(f"Target Stage: {stage}")
    logging.info(f"Input Size: {height}x{width}")

    try:
        # --- Load PyTorch Decoder ---
        logging.info("Loading PyTorch model...")
        if Path(steg_path_or_arch).is_file():
             steg = torch.load(steg_path_or_arch, map_location='cpu', weights_only=False)
        else:
             # Load via SteganoGAN.load to handle architecture lookup, but we only want the decoder
             temp_gan = SteganoGAN.load(architecture=steg_path_or_arch, cuda=False, verbose=False)
             steg = temp_gan # Use the loaded object to access the raw decoder
        pytorch_decoder = steg.decoder
        # Use original if compiled - might not be needed if loaded directly
        if hasattr(pytorch_decoder, '_orig_mod'):
             pytorch_decoder = pytorch_decoder._orig_mod
        pytorch_decoder.upgrade_legacy()
        pytorch_decoder.eval()
        logging.info("PyTorch decoder loaded.")

        # --- Load ONNX Sub-Model ---
        logging.info("Loading ONNX sub-model...")
        onnx_session = ort.InferenceSession(str(onnx_submodel_path))
        input_name_onnx = onnx_session.get_inputs()[0].name
        output_name_onnx = onnx_session.get_outputs()[0].name
        logging.info(f"ONNX loaded. Input: '{input_name_onnx}', Output: '{output_name_onnx}'")

        # --- Prepare Input ---
        logging.info("Preprocessing image...")
        input_np = preprocess_image_for_decode(image_path, height, width)
        input_tensor_torch = torch.from_numpy(input_np) # For PyTorch
        logging.info(f"Input preprocessed. Shape: {input_np.shape}, Dtype: {input_np.dtype}")

        # --- Get PyTorch Output ---
        logging.info(f"Calculating PyTorch output for {stage}...")
        pytorch_output_tensor = get_pytorch_stage_output(pytorch_decoder, input_tensor_torch, stage)
        pytorch_output_np = pytorch_output_tensor.detach().cpu().numpy()
        logging.info(f"PyTorch output calculated. Shape: {pytorch_output_np.shape}")

        # --- Get ONNX Output ---
        logging.info("Running ONNX inference...")
        onnx_inputs = {input_name_onnx: input_np}
        onnx_output_np = onnx_session.run([output_name_onnx], onnx_inputs)[0]
        logging.info(f"ONNX output calculated. Shape: {onnx_output_np.shape}")

        # --- Compare Outputs ---
        logging.info("\n--- Comparison Results ---")
        logging.info(f"Comparing stage: {stage}")
        logging.info(f"PyTorch Output Shape: {pytorch_output_np.shape}")
        logging.info(f"ONNX Output Shape:    {onnx_output_np.shape}")

        if pytorch_output_np.shape != onnx_output_np.shape:
            logging.error("Shapes DO NOT MATCH!")
            return

        # Compare first few values
        logging.info(f"PyTorch Output (first 10 flat): {pytorch_output_np.flatten()[:10]}")
        logging.info(f"ONNX Output (first 10 flat):    {onnx_output_np.flatten()[:10]}")

        # Calculate difference metrics
        abs_diff = np.abs(pytorch_output_np - onnx_output_np)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        median_abs_diff = np.median(abs_diff)

        logging.info(f"Max Absolute Difference:  {max_abs_diff:.8f}")
        logging.info(f"Mean Absolute Difference: {mean_abs_diff:.8f}")
        logging.info(f"Median Absolute Diff:   {median_abs_diff:.8f}")

        # Use np.allclose for a robust check
        # Adjust atol (absolute tolerance) and rtol (relative tolerance) as needed
        # Start with reasonably strict tolerances for steganography
        are_close = np.allclose(pytorch_output_np, onnx_output_np, rtol=1e-4, atol=1e-5)

        if are_close:
             logging.info("Conclusion: Outputs are NUMERICALLY CLOSE.")
        else:
             logging.warning("Conclusion: Outputs DIFFER SIGNIFICANTLY.")

    except Exception as e:
        logging.error(f"Comparison failed for stage {stage}: {e}")
        logging.error(traceback.format_exc())

    logging.info("--- Comparison Finished ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare intermediate outputs of PyTorch decoder and ONNX sub-model.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--arch', type=str, help='PyTorch model architecture name (e.g., dense)')
    model_group.add_argument('--steg_path', type=str, help='Path to PyTorch .steg model file')

    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX sub-model file (e.g., decoder_stage1.onnx)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input stego image')
    parser.add_argument('--stage', type=str, required=True, choices=['stage1', 'stage2', 'stage3', 'stage4'], help='Which intermediate stage output to compare')
    parser.add_argument('--height', type=int, default=256, help='Input image height')
    parser.add_argument('--width', type=int, default=256, help='Input image width')
    parser.add_argument('--log', type=str, default='comparison_log.txt', help='Path for the output log file')

    args = parser.parse_args()

    model_id = args.steg_path if args.steg_path else args.arch

    compare_stage_outputs(
        steg_path_or_arch=model_id,
        onnx_submodel_path=Path(args.onnx),
        image_path=Path(args.image),
        stage=args.stage,
        height=args.height,
        width=args.width,
        log_file=Path(args.log)
    )

if __name__ == '__main__':
    main()
