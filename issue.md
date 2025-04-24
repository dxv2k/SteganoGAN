# Issue: ONNX Decoder Bit Errors and FEC Workaround

## Context
- The goal is to export a PyTorch-based DenseDecoder (from SteganoGAN) to ONNX for use without the full torch dependency.
- The ONNX export process completes successfully, and the exported model structure matches the PyTorch model (as confirmed by verbose logs from torch.onnx.export).
- However, when running the ONNX model, the extracted payload contains bit errors compared to the PyTorch output, breaking the steganography payload after thresholding.

## Evidence
- Verbose ONNX export logs show correct mapping of PyTorch layers (Conv2d, BatchNorm2d, LeakyReLU, etc.) to ONNX operators.
- No structural errors or warnings are present in the export logs.
- The bit errors are likely due to minor numerical differences in ONNX Runtime's execution of standard operators (especially BatchNormalization) compared to PyTorch.
- These small floating-point discrepancies are enough to cause bit errors after thresholding.

## Root Cause
- The issue is not with the export process or model structure, but with numerical precision differences between PyTorch and ONNX Runtime, especially in sensitive layers like BatchNorm2d.

## Workaround: Forward Error Correction (FEC)
- FEC can be used to add redundancy to the payload before encoding, allowing the decoder to correct a certain number of bit errors after extraction.
- Example: Use Reed-Solomon codes (via the `reedsolo` Python library) to encode the payload before hiding it, and decode/correct errors after extraction.
- This requires modifying both the encoding and decoding pipeline to include FEC steps.
- FEC increases payload size, so the image must have enough capacity.

## Next Steps
- Experiment with FEC parameters (e.g., number of parity bytes) to find a balance between error correction capability and payload size.
- Consider alternative approaches (e.g., TorchScript export) if FEC is insufficient or too costly in terms of capacity.

---
*Logged for future revision and tracking. See also TODO.md for related tasks.* 