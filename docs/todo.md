# TODO: Lightweight SteganoGAN Extractor (ONNX)

## 1. Analyze Current Extractor
- [ ] Review current SteganoGAN decoder implementation
- [ ] Identify all dependencies and heavy components

## 2. Export Model to ONNX
- [ ] Check if SteganoGAN decoder (PyTorch) can be exported to ONNX
- [ ] Identify input/output schema for the decoder
- [ ] Write export script using `torch.onnx.export`
- [ ] Test ONNX export for compatibility (custom layers, etc.)
- [ ] Validate ONNX model output matches PyTorch model

## 3. Minimal ONNX Extractor Script
- [ ] Write a minimal Python script to load ONNX model and extract payload
- [ ] Use only `onnxruntime`, `numpy`, and standard libraries
- [ ] Ensure preprocessing/postprocessing matches original model
- [ ] Add logging, error handling, and type hints

## 4. Compile with PyInstaller
- [ ] Bundle extractor script and ONNX model using PyInstaller (`--onefile`)
- [ ] Test executable on Linux and Windows

## 5. Optimize & Document
- [ ] Document input/output schema and usage
- [ ] Add tests for extractor script
- [ ] Troubleshoot and optimize for size and speed

## References
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [Exporting PyTorch Models to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [Reducing Python Executable Size](https://pyinstaller.org/en/stable/usage.html#reducing-the-size-of-the-bundled-app)
