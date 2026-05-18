"""
pdf_chunked.py
--------------
Splits a large PDF into smaller chunk files; 1 page per chunk by default.
No changes from original — included for completeness.
"""

import os
import fitz  


def split_pdf_into_chunks(
    input_pdf: str,
    out_dir: str = "chunks",
    chunk_size: int = 1,
) -> list:
    """
    Splits a PDF into files of N pages each.

    Args:
        input_pdf  : Path to the source PDF.
        out_dir    : Output directory for chunk files.
        chunk_size : Pages per chunk (default 1 = one file per page).

    Returns:
        Sorted list of paths to generated chunk PDFs.
    """
    if not os.path.isfile(input_pdf):
        raise FileNotFoundError(f"PDF not found: {input_pdf}")

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(input_pdf)
    total_pages = len(doc)
    chunk_paths = []

    try:
        start, part_idx = 0, 1
        while start < total_pages:
            end = min(start + chunk_size - 1, total_pages - 1)
            chunk_path = os.path.join(
                out_dir,
                f"part_{part_idx:03d}_{start + 1:04d}-{end + 1:04d}.pdf",
            )
            newdoc = fitz.open()
            newdoc.insert_pdf(doc, from_page=start, to_page=end)
            newdoc.save(chunk_path)
            newdoc.close()
            chunk_paths.append(chunk_path)
            start += chunk_size
            part_idx += 1
    finally:
        doc.close()

    return sorted(chunk_paths)
