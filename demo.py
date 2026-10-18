#!/usr/bin/env python
"""RAG-Autopsy: Standalone 60-Second Interactive/Automated Demo.

Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Add src to pythonpath if run directly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rag_autopsy.demo import run_live_demo


if __name__ == "__main__":
    success = run_live_demo()
    sys.exit(0 if success else 1)
