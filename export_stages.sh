# Export Stage 1
python scripts/export_submodel.py --arch dense --stage s1 --output decoder_stage1.onnx

# Export Stage 1-2
python scripts/export_submodel.py --arch dense --stage s12 --output decoder_stage12.onnx

# Export Stage 1-2-3
python scripts/export_submodel.py --arch dense --stage s123 --output decoder_stage123.onnx

# Export Stage 1-2-3-4 (Full - should be same as decoder_dense.onnx)
python scripts/export_submodel.py --arch dense --stage s1234 --output decoder_stage1234.onnx
