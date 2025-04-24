import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import onnxruntime as ort
import base64


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def preprocess_image(image_path: Path, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    try:
        img = Image.open(image_path).convert('RGB')
        if height and width:
            img = img.resize((width, height))
        # Normalize to [0, 1] to match PyTorch decode logic
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dim
        return img_np
    except Exception as e:
        logging.error(f"Failed to preprocess image: {e}")
        traceback.print_exc()
        raise

def bits_to_bytearray(bits: list[int]) -> bytes:
    # Copied from steganogan.utils
    ba = bytearray()
    for b in range(0, len(bits), 8):
        byte = 0
        for i in range(8):
            if b + i < len(bits):
                byte |= (bits[b + i] & 1) << (7 - i)
        ba.append(byte)
    return bytes(ba)

def bytearray_to_text(ba: bytearray) -> str:
    # Copied from steganogan.utils
    try:
        return ba.decode('utf-8', errors='ignore')
    except Exception:
        return ''

def extract_payload_from_image(onnx_path: Path, image_path: Path, height: Optional[int], width: Optional[int]) -> Optional[str]:
    try:
        logging.info(f"Loading ONNX model from {onnx_path}")
        session = ort.InferenceSession(str(onnx_path))
        logging.info(f"Preprocessing image {image_path}")
        img_np = preprocess_image(image_path, height, width)
        logging.info("Running inference...")
        outputs = session.run(None, {session.get_inputs()[0].name: img_np})
        decoded = outputs[0]
        # Debug: print first 20 raw logits
        print("First 20 raw logits from ONNX:", decoded.flatten()[:20])
        # Postprocess: threshold at 0, flatten, convert to bits
        bits = (decoded.flatten() > 0).astype(np.int32).tolist()
        # Debug: print first 100 bits
        print("First 100 bits from ONNX:", bits[:100])
        # Split and decode messages (as in SteganoGAN)
        candidates = {}
        ba = bits_to_bytearray(bits)
        for candidate in ba.split(b'\x00\x00\x00\x00'):
            text = bytearray_to_text(bytearray(candidate))
            if text:
                candidates[text] = candidates.get(text, 0) + 1
        if not candidates:
            logging.error("Failed to find message.")
            return None
        # Return the most common candidate
        candidate = max(candidates.items(), key=lambda x: x[1])[0]
        logging.info(f"Extracted payload: {candidate[:64]}... (truncated)")
        return candidate
    except Exception as e:
        logging.error(f"Failed to extract payload: {e}")
        traceback.print_exc()
        return None

def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Extract payload from stego image using ONNX decoder.")
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX decoder model')
    parser.add_argument('--image', type=str, required=True, help='Path to stego image')
    parser.add_argument('--height', type=int, default=None, help='Input image height (optional)')
    parser.add_argument('--width', type=int, default=None, help='Input image width (optional)')
    parser.add_argument('--output', type=str, default=None, help='Path to write restored binary file (optional)')
    args = parser.parse_args()
    payload = extract_payload_from_image(Path(args.onnx), Path(args.image), args.height, args.width)
    if payload:
        if args.output:
            try:
                restored_binary = base64.b64decode(payload.encode('ascii'))
                with open(args.output, 'wb') as f:
                    f.write(restored_binary)
                logging.info(f"Restored binary written to {args.output}")
            except Exception as e:
                logging.error(f"Failed to decode or write binary payload: {e}")
                traceback.print_exc()
    else:
        print("No payload extracted.")

if __name__ == "__main__":
    main() 
 