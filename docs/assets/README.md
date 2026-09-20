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

## hero-bg.mp4 and hero-bg.jpg

The ambient hero background: a slow cyan-teal ribbon drifting through a
near-black void. 706KB of video, a 12KB poster.

### The fallback chain, in order

The page never depends on these files existing, which is worth keeping true:

1. `hero-bg.mp4` plays at 55% opacity once it can actually play
2. `hero-bg.jpg` is the `poster`, shown while the video loads and instead of
   it on slow connections
3. the CSS grid and sweep backdrop, which is always there underneath

The video is also skipped outright, before a single byte is requested, when
`prefers-reduced-motion` is set or `navigator.connection` reports `saveData`
or a 2g-class connection. `preload="none"` holds the request until that check
passes.

### How they were generated

Through Higgsfield's MCP server rather than the CLI, which on the `plus` trial
returns `only_mcp_usage_on_trial_is_available` for both `gpt_image_2_5` and
`z_image`.

Three stills were generated at 16:9 / 2k, roughly 1 credit each, from one
prompt with the texture clause swapped: `flowing topographic contour lines`,
`a soft drifting particle field`, and `slow liquid energy ribbons`.

```
Abstract near-black background texture for a premium dark website hero.
<texture clause> across a deep charcoal void. Extremely low contrast, faint
desaturated cyan-teal glow bleeding through one corner only. Cinematic,
minimal, vast negative space, soft atmospheric depth. No text, no logos, no
objects, no people. Subtle film grain.
```

The ribbons variant won. The scrim over the video is
`radial-gradient(120% 90% at 20% 0%, transparent 30%, ...)`, so the only place
the media reads at full strength is the top left, and the ribbons still is the
one whose glow sits in that corner while the rest falls away into smooth
darkness. It is also the lowest-detail of the three, which matters twice: it
does not compete with the CSS grid, and it holds up under compression where
contour lines and a particle field turn into shimmer.

That still was then animated with `seedance_2_5`, 35 credits:

```
mode: omni_reference     (t2v rejects start_image)
start_image: <the ribbons still>
resolution: 720p         (1080p is 60 credits, not 35)
generate_audio: false
prompt: Extremely slow ambient drift. The pattern breathes and flows gently
in place. No camera cuts, no zoom, no flashes, no new elements entering frame.
Seamless loop, cinematic, minimal.
```

720p rather than 1080p because the result is an out-of-focus wash sitting at
55% opacity under that scrim. There is no detail in it for the extra
resolution to carry.

### Making the loop actually loop

The model holds the frame still and adds no new elements, but it does not
return to its first frame, so a plain `loop` would visibly jump. The fix is to
cross-dissolve the last second onto the first second and drop the spare second:
the seam becomes a dissolve that happens to sit at the loop point. 5.04s in,
4.04s out.

```bash
ffmpeg -i raw-hero.mp4 -filter_complex "\
[0:v]trim=0:1,setpts=PTS-STARTPTS[head];\
[0:v]trim=1:4.0417,setpts=PTS-STARTPTS[body];\
[0:v]trim=4.0417:5.0417,setpts=PTS-STARTPTS[tail];\
[tail][head]xfade=transition=fade:duration=1:offset=0[blend];\
[blend][body]concat=n=2:v=1:a=0[out]" \
  -map "[out]" -an -r 24 \
  -c:v libx264 -crf 18 -preset slow -profile:v high -pix_fmt yuv420p \
  -movflags +faststart hero-bg.mp4

# poster: the same first frame, small enough to be effectively free
ffmpeg -i hero-bg.mp4 -frames:v 1 -q:v 6 hero-bg.jpg
```

`-movflags +faststart` matters: it moves the index to the front so playback
can begin before the whole file arrives. `-an` drops the audio track, which is
both smaller and required for reliable autoplay.

### On crf 18

The budget is ~1.5MB and `crf 30` came in at 62KB, which sounds like a win and
is not. Smooth near-black gradients are the worst case for H.264: at `crf 30`
the ribbon edges posterise into visible blocks. `crf 18` is 706KB, still less
than half the budget, and holds the gradient. When the content is this dark and
this smooth, spend the headroom.

A `libvpx-vp9` `.webm` second source would save perhaps a third of that. It is
not worth another `<source>` in `initHeroVideo` for 200KB, but that is the
lever if the budget ever tightens.

### Replacing them

Kept at 1276x722, the model's native output, deliberately unscaled — the old
`scale=1920:-2` in this file would have upscaled 720p into a bigger file
carrying no more detail. If you regenerate, match the glow to the scrim's
transparent corner at 20% from the left, or the media will be dark where the
page actually shows it.
