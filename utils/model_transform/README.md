```python

python -m transformers.onnx --model=intfloat/e5-large onnx_e5_large/ --atol=1e-3
python -m transformers.onnx --model=intfloat/e5-base .onnx_output/onnx_e5_base/ --atol=1e-3
python -m transformers.onnx --model=intfloat/e5-small .onnx_output/onnx_e5_small/ --atol=1e-3

python -m transformers.onnx --model=sentence-transformers/gtr-t5-xl .onnx_output/gtr_t5_xl/ --atol=1e-3
```