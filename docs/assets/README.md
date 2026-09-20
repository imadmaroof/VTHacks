# docs/assets

## impiricus-logo.svg

The masthead in `docs/index.html` loads the Impiricus logo from:

```
docs/assets/impiricus-logo.svg
```

That file is **not committed**. Impiricus' site sits behind Cloudflare, which
blocks automated downloads of their assets, so it has to be saved by hand from
a normal browser session:

1. Open <https://impiricus.com/assets/wp/2025/04/logo_white_final.svg> in your
   browser. It loads fine for a real browser, just not for scripts.
2. Save it as `docs/assets/impiricus-logo.svg`.

It's the white version of the mark, which is what this dark theme needs.

Until that file exists the page degrades gracefully: the `<img>` fails, its
`onerror` removes it, and a plain `IMPIRICUS` wordmark is shown instead. No
layout shift and no broken-image icon. The only side effect is one 404 in the
console.
