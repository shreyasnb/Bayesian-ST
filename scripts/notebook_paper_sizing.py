"""Source-only contact layouts for a 6.5-inch paper insertion.

The generated notebooks retain ordinary editable Matplotlib statements. Only
canvas, type, padding and export controls change; saved numerical inputs and
the plotted coordinate expressions are untouched.
"""
from __future__ import annotations

import re
from scripts.notebook_publication_style import figure_blocks


CONTACT_SIZES = {
    'posterior_anisotropy_risk.pdf': (7, 2.1),
    'prediction_error_sphere.pdf': (7, 2.4),
    'information_rotation.pdf': (7, 2.6),
    'posterior_anisotropy_spheres.pdf': (7, 3.1),
    'moving_sphere_posterior_spheres.pdf': (7, 3.2),
}


def contact_paper_layout(source):
    blocks = []
    for block in figure_blocks(source):
        filename = next((name for name in CONTACT_SIZES
                         if repr(name) in block), None)
        if filename is None:
            blocks.append(block)
            continue
        block = re.sub(r'^FIGSIZE = .*$', 'FIGSIZE = '+repr(CONTACT_SIZES[filename]),
                       block, flags=re.M)
        block = block.replace('# Font sizes are chosen for a six-inch-wide insertion in the paper.\n'
                              '# The width factor preserves that printed size when FIGSIZE is changed.',
                              '# Readable at approximately 6.5 inches in the paper.')
        for name, value in [('LABEL_FONT_SIZE', 10.5), ('TICK_FONT_SIZE', 10.),
                            ('LEGEND_FONT_SIZE', 10.)]:
            block = re.sub(r'^'+name+r' = .*$', name+' = '+repr(value), block, flags=re.M)
        block = block.replace("'axes.labelpad': 8.", "'axes.labelpad': 4.")
        block = block.replace('h_pad=.18, w_pad=.16', 'h_pad=.04, w_pad=.04')
        block = block.replace('bbox_inches=None', "bbox_inches='tight', pad_inches=.04")
        block = block.replace('ax.text(.015,.985,"(c)",transform=ax.transAxes,ha="left",va="top")\n', '')
        if filename in {'information_rotation.pdf', 'posterior_anisotropy_spheres.pdf'}:
            block = block.replace("DRAFT_SECTION = 'main'", "DRAFT_SECTION = 'supplementary'")
            # Retain the existing main-directory copies for manuscript compatibility.
            block = block.replace('    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())',
                '    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())\n'
                "    (ROOT / 'results/main/contact' / PDF_PATH.name).write_bytes(PDF_PATH.read_bytes())")
        if filename == 'posterior_anisotropy_spheres.pdf':
            block = block.replace('grid = fig.add_gridspec(1,2)',
                                  'grid = fig.add_gridspec(1,3, width_ratios=(1.,1.,.055))')
            block = block.replace('ax=spheres,label=', 'cax=fig.add_subplot(grid[0,2]),label=')
            block = block.replace('labelpad=8)', 'labelpad=2)')
        if filename == 'moving_sphere_posterior_spheres.pdf':
            block = block.replace("axes = [fig.add_subplot(1,2,index+1,projection='3d') for index in range(2)]",
                "grid = fig.add_gridspec(1,3, width_ratios=(1.,1.,.055))\n"
                "axes = [fig.add_subplot(grid[0,index],projection='3d') for index in range(2)]")
            block = block.replace('ax=axes,', 'cax=fig.add_subplot(grid[0,2]),')
            block = block.replace('labelpad=8.', 'labelpad=2.')
        if 'PNG_PATH = PDF_PATH.with_suffix' not in block:
            block = block.replace('plt.show()',
                "if SAVE_PDF:\n"
                "    PNG_PATH = PDF_PATH.with_suffix('.png')\n"
                "    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches='tight', pad_inches=.04)\n"
                "    DRAFT_PATH.with_suffix('.png').write_bytes(PNG_PATH.read_bytes())\n"
                + ("    (ROOT / 'results/main/contact' / PNG_PATH.name).write_bytes(PNG_PATH.read_bytes())\n"
                   if filename in {'information_rotation.pdf','posterior_anisotropy_spheres.pdf'} else '')
                + 'plt.show()')
        blocks.append(block)
    return ''.join(blocks)
