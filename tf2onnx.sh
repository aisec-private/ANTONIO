#!/bin/zsh

# zsh tf2onnx.sh

python -m tf2onnx.convert --saved-model ./path/to/tf/model --output ./path/to/onnx/model.onnx
