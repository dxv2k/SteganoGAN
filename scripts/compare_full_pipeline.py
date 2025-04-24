# scripts/compare_full_pipeline.py

import argparse
import logging
import traceback
from pathlib import Path
import os
import sys
from typing import Dict, Any, List

import torch
import onnxruntime as ort
import numpy as np
import imageio.v2 as iio # Use v2 explicitly to avoid deprecation warning

# Add project root to sys.path to allow importing steganogan
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from steganogan.models import SteganoGAN # noqa: E402

def configure_logging(log_file_path: Path) -> None:
    """Configure logging to file and console. Clears existing handlers."""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root_logger = logging.getLogger()
    # Clear previous handlers if any exist
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(logging.INFO)

    # File handler (append mode might be useful if run multiple times)
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
        img = iio.imread(str(image_path), pilmode='RGB') / 255.0
        # NOTE: Assuming input image is already correct size.
        # Add resizing logic here if needed, matching training.
        img_np = img.astype(np.float32)
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dim (1, C, H, W)
        return img_np
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        traceback.print_exc()
        raise

def get_all_pytorch_intermediate_outputs(decoder: torch.nn.Module, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculates all intermediate outputs of the PyTorch decoder."""
    outputs = {}
    decoder.eval()
    with torch.no_grad():
        x1 = decoder.conv1(input_tensor)
        outputs["stage1"] = x1

        x2 = decoder.conv2(x1)
        outputs["stage2"] = x2

        # Assuming DenseDecoder structure
        x3 = decoder.conv3(torch.cat([x1, x2], dim=1))
        outputs["stage3"] = x3

        x4 = decoder.conv4(torch.cat([x1, x2, x3], dim=1))
        outputs["stage4"] = x4
    return outputs

def run_onnx_stage(onnx_session: ort.InferenceSession, input_np: np.ndarray) -> np.ndarray:
    """Runs inference for a single ONNX stage."""
    input_name_onnx = onnx_session.get_inputs()[0].name
    output_name_onnx = onnx_session.get_outputs()[0].name
    onnx_inputs = {input_name_onnx: input_np}
    onnx_output_np = onnx_session.run([output_name_onnx], onnx_inputs)[0]
    return onnx_output_np

def compare_stages(
    steg_path_or_arch: str,
    onnx_stage_files: Dict[str, Path],
    image_path: Path,
    stages_to_compare: List[str],
    height: int,
    width: int,
    log_file: Path
) -> None:
    """Loads models, runs inference, compares outputs for multiple stages, and logs."""

    configure_logging(log_file)
    logging.info("##### Starting Full Pipeline Comparison #####")
    logging.info(f"PyTorch Model Source: {steg_path_or_arch}")
    logging.info(f"Input Image: {image_path}")
    logging.info(f"Stages to Compare: {', '.join(stages_to_compare)}")
    logging.info(f"Input Size: {height}x{width}")

    try:
        # --- Load PyTorch Decoder ---
        logging.info("--- Loading PyTorch Model ---")
        if Path(steg_path_or_arch).is_file():
             steg = torch.load(steg_path_or_arch, map_location='cpu', weights_only=False)
        else:
             temp_gan = SteganoGAN.load(architecture=steg_path_or_arch, cuda=False, verbose=False)
             steg = temp_gan
        pytorch_decoder = steg.decoder
        if hasattr(pytorch_decoder, '_orig_mod'):
             pytorch_decoder = pytorch_decoder._orig_mod
        pytorch_decoder.upgrade_legacy()
        pytorch_decoder.eval()
        logging.info("PyTorch decoder loaded.")

        # --- Prepare Input ---
        logging.info("--- Preprocessing Input Image ---")
        input_np = preprocess_image_for_decode(image_path, height, width)
        input_tensor_torch = torch.from_numpy(input_np)
        logging.info(f"Input preprocessed. Shape: {input_np.shape}, Dtype: {input_np.dtype}")

        # --- Get All PyTorch Intermediate Outputs ---
        logging.info("--- Calculating All PyTorch Intermediate Outputs ---")
        pytorch_outputs = get_all_pytorch_intermediate_outputs(pytorch_decoder, input_tensor_torch)
        for stage, tensor in pytorch_outputs.items():
             logging.info(f"PyTorch {stage} output calculated. Shape: {tensor.shape}")

        # --- Iterate Through Stages for Comparison ---
        for stage in stages_to_compare:
            onnx_file = onnx_stage_files.get(stage)
            if not onnx_file or not onnx_file.exists():
                logging.warning(f"ONNX file for {stage} not found or specified ({onnx_file}). Skipping comparison.")
                continue

            logging.info(f"\n===== Comparing Stage: {stage} =====")
            logging.info(f"Using ONNX file: {onnx_file}")

            try:
                # --- Load ONNX Sub-Model for the current stage ---
                onnx_session = ort.InferenceSession(str(onnx_file))
                logging.info(f"ONNX {stage} loaded.")

                # --- Determine Correct Input for ONNX Stage ---
                # Stage 1 takes the original image
                # Subsequent stages might take outputs from previous stages in a real
                # pipeline, but here we assume each ONNX file takes the *original*
                # input and computes *up to* that stage's output.
                # If your sub-models are structured differently (e.g., stage2 ONNX
                # takes stage1 output), you'll need to adjust this logic.
                current_input_np = input_np # Assuming all take original input

                # --- Run ONNX Inference for the current stage ---
                onnx_output_np = run_onnx_stage(onnx_session, current_input_np)
                logging.info(f"ONNX {stage} output calculated. Shape: {onnx_output_np.shape}")

                # --- Compare with PyTorch Output ---
                pytorch_output_np = pytorch_outputs[stage].detach().cpu().numpy()
                logging.info(f"PyTorch {stage} output shape: {pytorch_output_np.shape}")

                if pytorch_output_np.shape != onnx_output_np.shape:
                    logging.error("Shapes DO NOT MATCH!")
                    continue # Skip comparison if shapes mismatch

                # Compare first few values
                logging.info(f"PyTorch {stage} (first 10 flat): {pytorch_output_np.flatten()[:10]}")
                logging.info(f"ONNX {stage} (first 10 flat):    {onnx_output_np.flatten()[:10]}")

                # Calculate difference metrics
                abs_diff = np.abs(pytorch_output_np - onnx_output_np)
                max_abs_diff = np.max(abs_diff)
                mean_abs_diff = np.mean(abs_diff)
                median_abs_diff = np.median(abs_diff)

                logging.info(f"Max Absolute Difference:  {max_abs_diff:.8f}")
                logging.info(f"Mean Absolute Difference: {mean_abs_diff:.8f}")
                logging.info(f"Median Absolute Diff:   {median_abs_diff:.8f}")

                # Use np.allclose for a robust check
                are_close = np.allclose(pytorch_output_np, onnx_output_np, rtol=1e-4, atol=1e-5)

                if are_close:
                     logging.info("Conclusion: Outputs for this stage are NUMERICALLY CLOSE.")
                else:
                     logging.warning("Conclusion: Outputs for this stage DIFFER SIGNIFICANTLY.")

            except Exception as e:
                logging.error(f"Comparison failed specifically for stage {stage}: {e}")
                logging.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"Overall comparison process failed: {e}")
        logging.error(traceback.format_exc())

    logging.info("##### Full Pipeline Comparison Finished #####")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare intermediate outputs across PyTorch decoder and ONNX sub-models.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--arch', type=str, help='PyTorch model architecture name (e.g., dense)')
    model_group.add_argument('--steg_path', type=str, help='Path to PyTorch .steg model file')

    # Arguments for each stage's ONNX file
    parser.add_argument('--onnx_s1', type=str, required=True, help='Path to ONNX file for stage 1 output')
    parser.add_argument('--onnx_s2', type=str, required=True, help='Path to ONNX file for stage 2 output')
    parser.add_argument('--onnx_s3', type=str, required=True, help='Path to ONNX file for stage 3 output')
    parser.add_argument('--onnx_s4', type=str, required=True, help='Path to ONNX file for stage 4 output (full decoder)')

    parser.add_argument('--image', type=str, required=True, help='Path to the input stego image')
    parser.add_argument('--height', type=int, default=256, help='Input image height')
    parser.add_argument('--width', type=int, default=256, help='Input image width')
    parser.add_argument('--log', type=str, default='full_comparison_log.txt', help='Path for the output log file')

    args = parser.parse_args()

    model_id = args.steg_path if args.steg_path else args.arch

    onnx_files = {
        "stage1": Path(args.onnx_s1),
        "stage2": Path(args.onnx_s2),
        "stage3": Path(args.onnx_s3),
        "stage4": Path(args.onnx_s4),
    }

    compare_stages(
        steg_path_or_arch=model_id,
        onnx_stage_files=onnx_files,
        image_path=Path(args.image),
        stages_to_compare=['stage1', 'stage2', 'stage3', 'stage4'], # Compare all stages
        height=args.height,
        width=args.width,
        log_file=Path(args.log)
    )

if __name__ == '__main__':
    main()
