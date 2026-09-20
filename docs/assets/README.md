# docs/assets

## impiricus-logo.png

The Impiricus mark shown beside the wordmark in the masthead of
`docs/index.html`.

It was supplied as a JPEG with the transparency checkerboard flattened into
the pixels, so it could not be used directly on a dark background. It was
converted by keying that checkerboard out: luminance 92 to 232 was stretched
onto the alpha channel, which keeps the antialiased rim of the mark smooth
instead of going jagged, then recoloured to pure white, cropped to the mark,
padded square and resized to 512x512.

The conversion was checked against the source rather than by eye: every light
run across the middle of the source maps to alpha 255 in the output and every
dark run maps to alpha 0.

If the file is ever missing, the `onerror` on the `<img>` removes it and the
`IMPIRICUS` wordmark stands on its own, so the masthead still reads correctly.
