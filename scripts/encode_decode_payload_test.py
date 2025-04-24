import base64
import logging
import traceback
from pathlib import Path
from typing import Optional
from steganogan import SteganoGAN


def configure_logging() -> None:
    """Configure root logger to output time, level and message."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def read_binary_file(path: Path) -> bytes:
    try:
        logging.info(f"Reading binary payload from {path}")
        with open(path, 'rb') as f:
            data = f.read()
        logging.info(f"Read {len(data)} bytes from {path}")
        return data
    except Exception as e:
        logging.error(f"Failed to read binary file: {e}")
        traceback.print_exc()
        raise

def write_binary_file(path: Path, data: bytes) -> None:
    try:
        logging.info(f"Writing binary payload to {path}")
        with open(path, 'wb') as f:
            f.write(data)
        logging.info(f"Wrote {len(data)} bytes to {path}")
    except Exception as e:
        logging.error(f"Failed to write binary file: {e}")
        traceback.print_exc()
        raise

def encode_payload_to_image(model: SteganoGAN, cover_path: Path, output_path: Path, payload_b64: str) -> None:
    try:
        logging.info(f"Encoding payload into image: {cover_path} -> {output_path}")
        model.encode(
            cover=str(cover_path),
            output=str(output_path),
            text=payload_b64
        )
        logging.info("Encoding completed successfully.")
    except Exception as e:
        logging.error(f"Encoding failed: {e}")
        traceback.print_exc()
        raise

def decode_payload_from_image(model: SteganoGAN, stego_path: Path) -> Optional[str]:
    try:
        logging.info(f"Decoding payload from image: {stego_path}")
        # Patch: get raw logits and bits for debug
        import imageio
        import torch
        from steganogan.utils import bits_to_bytearray
        image = imageio.imread(str(stego_path), pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(model.device)
        with torch.cuda.amp.autocast(enabled=model.cuda):
            decoded = model.decoder(image)
        print("First 20 raw logits from PyTorch:", decoded.view(-1)[:20].detach().cpu().numpy())
        bits = (decoded.view(-1) > 0).int().detach().cpu().numpy().tolist()
        print("First 100 bits from PyTorch:", bits[:100])
        # Now do the original decode logic
        decoded_b64 = model.decode(str(stego_path))
        if decoded_b64 is None:
            logging.error("Decoding returned None.")
        else:
            logging.info(f"Decoded base64 payload, length: {len(decoded_b64)} characters")
        return decoded_b64
    except Exception as e:
        logging.error(f"Decoding failed: {e}")
        traceback.print_exc()
        return None

def main() -> None:
    configure_logging()
    try:
        # Paths
        cover_path = Path('cola.jpg')
        stego_path = Path('encoded_output.png')
        payload_path = Path('payload/shell.elf')
        restored_path = Path('restored_shell.elf')

        # Load model
        logging.info("Loading SteganoGAN model (dense architecture)...")
        model = SteganoGAN.load(architecture='dense')
        logging.info("Model loaded.")

        # Read and encode binary payload
        binary_data = read_binary_file(payload_path)
        b64_data = base64.b64encode(binary_data).decode('ascii')
        logging.info(f"Base64-encoded payload length: {len(b64_data)} characters")

        # # Encode into image
        # encode_payload_to_image(model, cover_path, stego_path, b64_data)

        # Decode from image
        decoded_b64 = decode_payload_from_image(model, stego_path)
        if decoded_b64 is None:
            logging.error("No payload decoded. Exiting.")
            return

        # Restore binary
        try:
            restored_binary = base64.b64decode(decoded_b64.encode('ascii'))
            logging.info(f"Decoded binary payload length: {len(restored_binary)} bytes")
        except Exception as e:
            logging.error(f"Failed to decode base64 payload: {e}")
            traceback.print_exc()
            return
        write_binary_file(restored_path, restored_binary)
        logging.info(f"Restored binary written to {restored_path}")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()