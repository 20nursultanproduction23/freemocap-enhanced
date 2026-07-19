"""Ensure onnxruntime is loaded before PySide6 to avoid DLL conflicts."""
import importlib
import sys


def _early_import_onnxruntime():
    """Import onnxruntime first so its DLLs are loaded before Qt's."""
    try:
        import onnxruntime
    except Exception:
        pass


_early_import_onnxruntime()
