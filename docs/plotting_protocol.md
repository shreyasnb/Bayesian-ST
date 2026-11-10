# Plotting protocol

The implementation walkthrough and the current application notebooks use
actual LaTeX with serif text. Their earlier visual differences came from
typography and figure layout, rather than a different text renderer.

The preserved implementation notebook was produced using 11-point main and
axis text, 10-point ticks and legends, and 1.6-point lines. Its working
figures were 13–17 inches wide, usually 5 inches high per row, with tight
bounding boxes. The embedded PNGs are 2617–3423 pixels wide at 200 dpi and
are displayed at a width of 1400 pixels. This is the historical notebook's
original output, not a newly generated 300-dpi study.

The current application figures previously used 9-point text, 1.35-point
lines, and a seven-inch width. Their 300-dpi previews are 2100 pixels wide
and are also displayed at 1400 pixels. Inline width alone therefore does
not explain the difference. Source font size must be considered together
with figure width and the amount of space assigned to each panel.

The shared current typography now follows the implementation notebook:

| Setting | Current value |
| --- | --- |
| Main and axis text | 11 pt, serif LaTeX |
| Tick labels and legends | 10 pt |
| Line width | 1.6 pt |
| Default marker size | 4 pt |
| Manuscript page width | 7 in |
| Preview and raster-layer resolution | 300 dpi |

The seven-inch width preserves the manuscript insertion size. The old
13–17-inch working canvases are not copied literally: reducing an entire
17-inch figure to seven inches would also reduce its 11-point text to
about 4.5 points. Panel layout and height are adjusted instead so labels
and legends have sufficient room at the actual insertion size.

PDF text and curves remain vector objects and do not have a pixel dpi.
The 300-dpi setting controls density surfaces and other raster layers, as
well as PNG previews. Every current seven-inch preview is 2100 pixels
wide. Notebook previews are scaled to the screen; the accompanying PDF
and full-resolution PNG links retain the export quality. Existing
historical result files keep their original metadata.

Unchanged compositions in the editable notebooks preserve the author's
15-by-5-inch or 15-by-10-inch settings. Selected panels are displayed as separate
figures in the same code cell, using 7.5-by-5 inches for half-width panels and
15-by-5 inches for full-width panels. The two contact density spheres retain
their shared side-by-side figure. Every figure has its own editable size and
PDF save; the seven-inch published catalogue remains a separate set of files.

Figure changes read the saved, verified arrays. Font, layout and color
changes do not rerun inference, tune an estimator, or change a benchmark
or its numerical results. Density colorbars retain their explicit
normalization; colored display regions are not credible-region boundaries.
