#!/bin/bash
# Install LLM dependencies for Windows (Git Bash/WSL)

echo "Installing LLM dependencies for Windows..."

# Install PyTorch CPU-only from official index (avoids DLL issues)
echo "Installing PyTorch CPU-only..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers (now that PyTorch is installed)
echo "Installing sentence-transformers..."
pip install sentence-transformers

# Install other LLM dependencies
echo "Installing other LLM dependencies..."
pip install -r requirements-llm.txt

echo ""
echo "Installation complete!"
echo "You can now run: make rag"
