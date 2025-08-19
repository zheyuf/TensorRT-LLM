#!/bin/bash

echo "Setting up TensorRT-LLM development environment..."

# Install TensorRT-LLM in development mode to ensure all dependencies
cd /code/tensorrt_llm
pip install -e . --quiet

# Verify critical dependencies
python -c "import tensorrt_llm; print('TensorRT-LLM imported successfully')"
python -c "import xgrammar; print('xgrammar imported successfully')"
python -c "from tensorrt_llm.llmapi import LLM; print('LLM API imported successfully')"

echo "Container setup complete! You can now run memory profiling."
