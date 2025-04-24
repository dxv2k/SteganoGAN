# scripts/export_submodel_stage1.py
import torch
import os
from pathlib import Path
import sys
# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from steganogan.models import SteganoGAN

# --- Config ---
PYTORCH_MODEL_ARCH = "dense"
HEIGHT, WIDTH = 256, 256
OUTPUT_ONNX_PATH = "decoder_stage1.onnx"

# --- Load original decoder ---
print("Loading PyTorch model...")
steganogan = SteganoGAN.load(architecture=PYTORCH_MODEL_ARCH, cuda=False, verbose=False)
original_decoder = steganogan.decoder
if hasattr(original_decoder, '_orig_mod'):
    original_decoder = original_decoder._orig_mod
original_decoder.upgrade_legacy()
print("Original decoder loaded.")

# --- Define Stage 1 Model ---
class DecoderStage1(torch.nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        # Assuming DenseDecoder structure, copy the first conv block
        self.conv1 = original_decoder.conv1
    def forward(self, x):
        return self.conv1(x)

stage1_model = DecoderStage1(original_decoder).eval()
print("Stage 1 sub-model created.")

# --- Export ---
dummy_input = torch.randn(1, 3, HEIGHT, WIDTH)
print(f"Exporting Stage 1 model to {OUTPUT_ONNX_PATH}...")
torch.onnx.export(
    stage1_model,
    dummy_input,
    OUTPUT_ONNX_PATH,
    export_params=True,
    opset_version=13,         # Match previous exports
    do_constant_folding=False, # Match previous exports
    input_names=['input'],
    output_names=['output_stage1'], # Specific output name
    dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                  'output_stage1': {0: 'batch_size', 2: 'height', 3: 'width'}}
)
print("Export complete.")