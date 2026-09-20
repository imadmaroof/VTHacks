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

## hero-bg.mp4 and hero-bg.jpg (not yet generated)

The ambient hero background. **Both files are still missing**, and the page
is built to be fine with that, so there is nothing to fix until they exist.

### Why they are missing

Generation is blocked on the Higgsfield account tier. Both `gpt_image_2_5`
and `z_image` return:

```
Error: {"error_type":"only_mcp_usage_on_trial_is_available"}
```

On the current `plus` trial, generation only runs through Higgsfield's MCP
server, not the CLI. The server has been registered locally:

```bash
claude mcp add --transport http higgsfield https://mcp.higgsfield.ai/mcp
```

It shows `Needs authentication` until the OAuth handshake is completed from
inside Claude Code (`/mcp`), and MCP tools only register at session start, so
the tools become callable in a later session.

### The fallback chain, in order

The page never depends on these files existing:

1. `hero-bg.mp4` plays at 55% opacity once it can actually play
2. `hero-bg.jpg` is the `poster`, shown while the video loads and instead of
   it on slow connections
3. the CSS grid and sweep backdrop, which is always there underneath

The video is also skipped outright, before a single byte is requested, when
`prefers-reduced-motion` is set or `navigator.connection` reports `saveData`
or a 2g-class connection. `preload="none"` holds the request until that check
passes.

### Prompts to generate them

Image, at hero-banner ratio, roughly 1 credit each:

```bash
higgsfield generate create gpt_image_2_5 --aspect_ratio 16:9 --resolution 2k --wait \
  --prompt "Abstract near-black background texture for a premium dark website hero. \
Flowing topographic contour lines drifting softly across a deep charcoal void. \
Extremely low contrast, faint desaturated cyan-teal glow bleeding through one corner only. \
Cinematic, minimal, vast negative space, soft atmospheric depth. \
No text, no logos, no objects, no people. Subtle film grain."
```

Worth generating two or three variations, swapping the texture clause for
`soft drifting particle field` or `slow liquid energy ribbons`, then animating
whichever reads best.

Video, image-to-video from the chosen still, 35 credits:

```bash
higgsfield generate create seedance_2_5 --start-image <chosen-still> --wait \
  --prompt "Extremely slow ambient drift. The pattern breathes and flows gently in place. \
No camera cuts, no zoom, no flashes, no new elements entering frame. \
Seamless loop, cinematic, minimal."
```

### Compression before committing

Keep the hero under roughly 1.5MB. The video carries no audio, so strip it.

```bash
# video: ~1.5MB target, no audio, web-optimised header placement
ffmpeg -i raw-hero.mp4 -an -vf "scale=1920:-2,fps=24" \
  -c:v libx264 -crf 30 -preset slow -profile:v high -pix_fmt yuv420p \
  -movflags +faststart hero-bg.mp4

# optional smaller modern codec, add a second <source> in initHeroVideo if used
ffmpeg -i raw-hero.mp4 -an -vf "scale=1920:-2,fps=24" \
  -c:v libvpx-vp9 -crf 38 -b:v 0 hero-bg.webm

# poster: same first frame, small enough to be effectively free
ffmpeg -i hero-bg.mp4 -frames:v 1 -q:v 6 hero-bg.jpg
```

`-movflags +faststart` matters: it moves the index to the front so playback
can begin before the whole file arrives. `-an` drops the audio track, which
is both smaller and required for reliable autoplay.
