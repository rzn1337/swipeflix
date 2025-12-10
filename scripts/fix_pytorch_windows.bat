@echo off
REM Fix PyTorch installation on Windows
REM This script uninstalls and reinstalls PyTorch CPU version to fix DLL loading issues

echo Fixing PyTorch installation on Windows...

REM Uninstall existing PyTorch and related packages
echo Uninstalling existing PyTorch packages...
pip uninstall -y torch torchvision torchaudio sentence-transformers

REM Install PyTorch CPU version from PyPI (works on Windows)
echo Installing PyTorch CPU version...
pip install torch>=2.0.0,<3.0.0 --index-url https://download.pytorch.org/whl/cpu

REM Reinstall sentence-transformers
echo Reinstalling sentence-transformers...
pip install sentence-transformers>=2.2.0,<3.0.0

echo PyTorch installation fixed! Try running 'make rag' again.
