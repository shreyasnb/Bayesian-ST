"""Stack the saved contact PDFs without rasterizing their text or curves.

Run with Python and PyMuPDF installed, after exporting the two notebook plots.
No numerical inputs or simulations are executed by this composition step.
"""
from pathlib import Path
import argparse

import fitz


def export(root):
    folder = root / 'results/main/contact'
    sources = [folder / 'posterior_anisotropy_risk.pdf',
               folder / 'prediction_error_sphere.pdf']
    width, gap = 6.5 * 72., 6.
    documents = [fitz.open(path) for path in sources]
    try:
        assert all(len(document) == 1 for document in documents)
        heights = [width * document[0].rect.height / document[0].rect.width
                   for document in documents]
        output = fitz.open()
        page = output.new_page(width=width, height=sum(heights) + gap)
        top = 0.
        for document, height in zip(documents, heights):
            page.show_pdf_page(fitz.Rect(0., top, width, top + height), document, 0)
            top += height + gap
        target = folder / 'contact_comparison.pdf'
        output.save(target, garbage=4, deflate=True)
        preview = page.get_pixmap(matrix=fitz.Matrix(300./72., 300./72.), alpha=False)
        preview.set_dpi(300, 300)
        preview.save(target.with_suffix('.png'))
        print(f'{target}: {width/72.:.3f} x {page.rect.height/72.:.3f} inches')
        output.close()
    finally:
        for document in documents:
            document.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    export(parser.parse_args().root.resolve())
