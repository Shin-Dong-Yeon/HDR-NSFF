# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **static project website** for the ICLR 2026 paper "High Dynamic Range Neural Scene Flow Fields" (HDR-NSFF). It presents a method for reconstructing dynamic HDR radiance fields from alternating-exposure monocular videos.

- Paper: https://arxiv.org/abs/2603.08313
- Code: https://github.com/kaist-ami/HDR-NSFF

## Running Locally

No build process — open `index.html` directly in a browser or serve with any static file server:

```bash
python3 -m http.server 8000
```

## Architecture

Single-page static site adapted from the [Nerfies template](https://nerfies.github.io):

- **`index.html`** — all content: abstract, method description, results tables, BibTeX citation
- **`static/js/index.js`** — carousel controls, interpolation slider (240 frames), mobile navbar toggle
- **`static/css/index.css`** — custom styles on top of Bulma CSS framework
- **`static/videos/`** — demo videos (~550MB): teaser, HDR rendering results, 4D comparison, interpolation sequences
- **`static/images/`** — method diagrams: pipeline, flow visualization, comparison figures
- **`paper/`** — PDF of the paper (gitignored)

External dependencies are CDN-hosted (jQuery 3.5.1, Google Fonts, Font Awesome) or bundled locally (Bulma CSS/JS).

The interpolation viewer in `index.js` uses a slider to scrub through numbered image frames (e.g., `frame000.jpg` to `frame239.jpg`) stored under `static/videos/interpolation/`.
