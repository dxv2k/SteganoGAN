# scripts/export_and_compare_opsets.py

import argparse
import logging
import traceback
from pathlib import Path
import os
import sys
import time
from typing import Dict, Any, List, Optional

import torch
import onnxruntime as ort
import numpy as np
import imageio.v2 as iio # Use v2 explicitly to avoid deprecation warning

# Add project root to sys.path to allow importing steganogan
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from steganogan.models import SteganoGAN # noqa: E402

# --- Configuration ---
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_LOG_FILE = "opset_comparison_log.txt"

# --- Logging Setup ---
def configure_logging(log_file_path: Path) -> None:
    """Configure logging to file and console. Clears existing handlers."""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# --- Helper Functions ---

def preprocess_image_for_decode(image_path: Path, height: int, width: int) -> np.ndarray:
    """Preprocesses image exactly like PyTorch decode path."""
    try:
        img = iio.imread(str(image_path), pilmode='RGB') / 255.0
        # NOTE: Assuming input image is already correct size. Add resize if needed.
        img_np = img.astype(np.float32)
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dim (1, C, H, W)
        return img_np
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        traceback.print_exc()
        raise

def get_pytorch_final_output(decoder: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Calculates the final output (Stage 4) of the PyTorch decoder."""
    decoder.eval()
    with torch.no_grad():
        # Assuming DenseDecoder structure for calculation (adapt if needed)
        x1 = decoder.conv1(input_tensor)
        x2 = decoder.conv2(x1)
        x3 = decoder.conv3(torch.cat([x1, x2], dim=1))
        x4 = decoder.conv4(torch.cat([x1, x2, x3], dim=1))
        return x4.detach().cpu().numpy()

def export_decoder_onnx(
    decoder: torch.nn.Module,
    onnx_path: Path,
    height: int,
    width: int,
    opset_version: int
) -> bool:
    """Exports the provided decoder module to ONNX."""
    try:
        decoder.eval() # Ensure eval mode
        dummy_input = torch.randn(1, 3, height, width)
        logging.info(f"Exporting to ONNX: {onnx_path} (opset={opset_version})")
        torch.onnx.export(
            decoder,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False, # Keep false for consistency during debug
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                          'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
        logging.info(f"ONNX model saved to {onnx_path}")
        return True
    except Exception as e:
        logging.error(f"Export failed for opset {opset_version}: {e}")
        logging.error(traceback.format_exc())
        return False

def run_onnx_inference(onnx_path: Path, input_np: np.ndarray) -> Optional[np.ndarray]:
    """Runs inference using the specified ONNX model."""
    try:
        logging.info(f"Loading ONNX model: {onnx_path}")
        onnx_session = ort.InferenceSession(str(onnx_path))
        input_name_onnx = onnx_session.get_inputs()[0].name
        output_name_onnx = onnx_session.get_outputs()[0].name
        onnx_inputs = {input_name_onnx: input_np}
        logging.info(f"Running ONNX inference for {onnx_path.name}...")
        start_time = time.time()
        onnx_output_np = onnx_session.run([output_name_onnx], onnx_inputs)[0]
        end_time = time.time()
        logging.info(f"ONNX inference complete ({end_time - start_time:.3f}s). Shape: {onnx_output_np.shape}")
        return onnx_output_np
    except Exception as e:
        logging.error(f"ONNX inference failed for {onnx_path.name}: {e}")
        logging.error(traceback.format_exc())
        return None

def compare_outputs(pytorch_output_np: np.ndarray, onnx_output_np: np.ndarray) -> None:
    """Compares PyTorch and ONNX outputs and logs results."""
    logging.info(f"PyTorch Output Shape: {pytorch_output_np.shape}")
    logging.info(f"ONNX Output Shape:    {onnx_output_np.shape}")

    if pytorch_output_np.shape != onnx_output_np.shape:
        logging.error("Shapes DO NOT MATCH!")
        return

    logging.info(f"PyTorch (first 10 flat): {pytorch_output_np.flatten()[:10]}")
    logging.info(f"ONNX (first 10 flat):    {onnx_output_np.flatten()[:10]}")

    abs_diff = np.abs(pytorch_output_np - onnx_output_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    median_abs_diff = np.median(abs_diff)

    logging.info(f"Max Absolute Difference:  {max_abs_diff:.8f}")
    logging.info(f"Mean Absolute Difference: {mean_abs_diff:.8f}")
    logging.info(f"Median Absolute Diff:   {median_abs_diff:.8f}")

    are_close = np.allclose(pytorch_output_np, onnx_output_np, rtol=1e-4, atol=1e-5)
    if are_close:
         logging.info("Conclusion: Outputs are NUMERICALLY CLOSE.")
    else:
         logging.warning("Conclusion: Outputs DIFFER SIGNIFICANTLY.")

# --- Main Comparison Workflow ---
def run_opset_comparison(
    steg_path_or_arch: str,
    image_path: Path,
    opsets_to_test: List[int],
    height: int,
    width: int,
    log_file: Path,
    output_dir: Path # Directory to save temporary ONNX files
) -> None:

    configure_logging(log_file)
    logging.info("##### Starting Opset Export and Comparison #####")
    logging.info(f"PyTorch Model Source: {steg_path_or_arch}")
    logging.info(f"Input Image: {image_path}")
    logging.info(f"Opsets to Test: {opsets_to_test}")
    logging.info(f"Input Size: {height}x{width}")
    logging.info(f"Temporary ONNX file directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    try:
        # --- Load PyTorch Decoder (once) ---
        logging.info("--- Loading PyTorch Model ---")
        if Path(steg_path_or_arch).is_file():
             steg = torch.load(steg_path_or_arch, map_location='cpu', weights_only=False)
        else:
             temp_gan = SteganoGAN.load(architecture=steg_path_or_arch, cuda=False, verbose=False)
             steg = temp_gan
        pytorch_decoder = steg.decoder
        if hasattr(pytorch_decoder, '_orig_mod'):
             logging.info("Detected compiled decoder, using original module (_orig_mod).")
             pytorch_decoder = pytorch_decoder._orig_mod
        pytorch_decoder.upgrade_legacy()
        pytorch_decoder.eval()
        logging.info("PyTorch decoder loaded.")

        # --- Prepare Input (once) ---
        logging.info("--- Preprocessing Input Image ---")
        input_np = preprocess_image_for_decode(image_path, height, width)
        input_tensor_torch = torch.from_numpy(input_np)
        logging.info(f"Input preprocessed. Shape: {input_np.shape}, Dtype: {input_np.dtype}")

        # --- Get PyTorch Final Output (once) ---
        logging.info("--- Calculating PyTorch Final Output ---")
        pytorch_final_output_np = get_pytorch_final_output(pytorch_decoder, input_tensor_torch)
        logging.info(f"PyTorch final output calculated. Shape: {pytorch_final_output_np.shape}")

        # --- Loop Through Opsets: Export, Infer, Compare ---
        for opset in opsets_to_test:
            logging.info(f"\n===== Processing Opset: {opset} =====")
            onnx_filename = output_dir / f"decoder_opset{opset}.onnx"

            # Export
            export_success = export_decoder_onnx(pytorch_decoder, onnx_filename, height, width, opset)
            if not export_success:
                logging.error(f"Skipping comparison for opset {opset} due to export failure.")
                continue

            # Infer
            onnx_final_output_np = run_onnx_inference(onnx_filename, input_np)
            if onnx_final_output_np is None:
                logging.error(f"Skipping comparison for opset {opset} due to inference failure.")
                continue

            # Compare
            logging.info(f"--- Comparing Opset {opset} Output vs PyTorch ---")
            compare_outputs(pytorch_final_output_np, onnx_final_output_np)

            # Optional: Clean up the temporary ONNX file
            # onnx_filename.unlink(missing_ok=True)
            # logging.info(f"Removed temporary file: {onnx_filename}")

    except Exception as e:
        logging.error(f"Overall comparison process failed: {e}")
        logging.error(traceback.format_exc())

    logging.info("##### Opset Comparison Finished #####")

# --- Command Line Interface ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Export SteganoGAN decoder for multiple opsets and compare final outputs.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--arch', type=str, help='PyTorch model architecture name (e.g., dense)')
    model_group.add_argument('--steg_path', type=str, help='Path to PyTorch .steg model file')

    parser.add_argument('--image', type=str, required=True, help='Path to the input stego image')
    parser.add_argument('--opsets', type=int, nargs='+', default=[12, 13, 14, 15, 16], help='List of ONNX opset versions to test')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help='Input image height')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help='Input image width')
    parser.add_argument('--log', type=str, default=DEFAULT_LOG_FILE, help='Path for the output log file')
    parser.add_argument('--onnx_dir', type=str, default='./temp_onnx_exports', help='Directory to store temporary ONNX files')

    args = parser.parse_args()

    model_id = args.steg_path if args.steg_path else args.arch
    output_dir = Path(args.onnx_dir)

    run_opset_comparison(
        steg_path_or_arch=model_id,
        image_path=Path(args.image),
        opsets_to_test=sorted(list(set(args.opsets))), # Ensure unique and sorted
        height=args.height,
        width=args.width,
        log_file=Path(args.log),
        output_dir=output_dir
    )

if __name__ == '__main__':
    main()
