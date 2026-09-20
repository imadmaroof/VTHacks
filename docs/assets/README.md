# docs/assets

## impiricus-logo.svg

The official Impiricus lockup, mark and wordmark together, in white vector,
shown in the masthead of `docs/index.html` at 28px tall.

Using the real lockup rather than pairing the mark with CSS text means the
letterforms and the spacing between mark and wordmark are theirs, not an
approximation. It is also about a seventh the size of the raster version it
replaced, and stays crisp on any display.

The `<img>` keeps an `onerror` that swaps in a plain `IMPIRICUS` text
wordmark, so a missing or broken file still leaves a readable masthead
instead of a broken-image icon.

Note the file is white artwork with a transparent ground, so it only reads
against a dark surface. Anything placed on a light background needs the dark
variant instead.
