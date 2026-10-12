"""
Exports eda.ipynb to PDF without code: python eda_assets/export_pdf.py (or the Run button in PyCharm)
(optional arguments: notebook and output name, default eda.ipynb -> 3_12.pdf).

Why not `jupyter nbconvert --to webpdf`: on Windows nbconvert switches asyncio to the
"selector" event loop, which cannot start the browser that renders the PDF
(NotImplementedError). Here the exporter is called directly with the default loop.
The notebook must be run and saved first: the PDF contains the saved outputs.
"""
import asyncio
import sys
from pathlib import Path

import nbformat
from nbconvert import WebPDFExporter

ROOT = Path(__file__).resolve().parent.parent          # repository root (where eda.ipynb is)
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eda.ipynb"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "3_12.pdf"

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nb = nbformat.read(src, as_version=4)
exporter = WebPDFExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True,
                          allow_chromium_download=True)
pdf, _ = exporter.from_notebook_node(nb, resources={"metadata": {"name": src.stem}})
out.write_bytes(pdf)
print(f"saved {out.resolve()} ({len(pdf) / 1e6:.1f} MB)")