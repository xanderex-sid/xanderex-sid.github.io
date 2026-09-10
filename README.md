# xanderex-sid.github.io

Personal site — Astro + Tailwind CSS + MDX, deployed to GitHub Pages at
<https://xanderex-sid.github.io>.

## Run locally

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # static output into dist/
npm run preview  # serve the built dist/ locally
npm run check    # TypeScript + Astro diagnostics
```

Requires Node 20+ (built and tested on Node 22).

## Add a blog post

Create one file in `src/content/blog/`. The filename becomes the URL:
`my-post.mdx` → `/blog/my-post`.

```mdx
---
title: 'Your post title'
date: 2026-09-20
description: 'One or two lines. Shown on the blog index and in search results.'
tags: ['LiDAR', 'Deep Learning'] # optional
cover: './my-cover.jpg'          # optional, relative to this file
coverAlt: 'Describe the image'   # required if cover is set
draft: false                     # true hides it from the site
---

Your prose here.
```

Reading time is calculated automatically. Posts sort newest first.

**Code blocks** support a filename label and optional line numbers:

````text
```python title="train/losses.py" showLineNumbers
def loss_fn(logits, target):
    ...
```
````

Both are optional — ` ```python ` alone works and shows the language in the header.
Every block gets a copy button automatically.

**Images with captions** — put the file in `public/images/` and use a `<figure>`:

```html
<figure>
  <img src="/images/my-figure.png" alt="Describe it" loading="lazy" />
  <figcaption>Caption text.</figcaption>
</figure>
```

## Add a project

Add an entry to `src/data/projects.json`:

```json
{
  "id": "my-project",
  "title": "My project",
  "description": "One or two sentences.",
  "videoSrc": "/videos/my-project.mp4",
  "posterSrc": "/posters/my-project.jpg",
  "posterAlt": "Describe the poster image",
  "repoUrl": "https://github.com/xanderex-sid/my-project",
  "tags": ["Python", "PyTorch"],
  "order": 0
}
```

Then drop the files in:

- **Video** → `public/videos/my-project.mp4` — 20–30s, muted, silent, loops cleanly.
  It plays on hover and is lazy-loaded (only fetched when scrolled near).
- **Poster** → `public/posters/my-project.jpg` — **required**. Shown before the video
  loads, and it's the whole block when there is no video yet.

Blocks sort by `order` (ascending). Clicking one opens `repoUrl` in a new tab; omit
`repoUrl` and the block renders as non-clickable. Set `"comingSoon": true` for a
placeholder block.

**No video yet?** Leave `videoSrc` out, or set it to `"TODO"` — anything starting with
`TODO` is treated as absent and the poster shows with a "Video coming soon" badge.

## Add a news item

Add an entry to `src/data/news.json` — the homepage sorts by date, newest first:

```json
{
  "id": "unique-slug",
  "date": "2026-09-20",
  "text": "What happened.",
  "href": "https://example.com", // optional
  "linkText": "Paper"            // optional, defaults to "Link"
}
```

## Add a publication

Add an entry to `src/data/publications.json`:

```json
{
  "id": "unique-slug",
  "title": "Full paper title",
  "authors": ["First Author", "Siddharth Mishra"],
  "highlightAuthor": "Siddharth Mishra",
  "venue": "Conference or journal name",
  "year": 2026,
  "doi": "10.1109/...",
  "links": [{ "label": "DOI", "href": "https://doi.org/10.1109/..." }],
  "note": "Optional — renders as a visible amber TODO box. Delete when resolved."
}
```

`highlightAuthor` must match one string in `authors` exactly; it gets bolded and
underlined in the citation.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds with
`withastro/action` and publishes via `actions/deploy-pages`.

**One-time setup:** in the repo on GitHub go to **Settings → Pages → Build and
deployment → Source** and select **GitHub Actions**. Without this the workflow builds
but nothing goes live.

Because this is a *user* site served from the domain root, `astro.config.mjs` sets
`site: 'https://xanderex-sid.github.io'` and `base: '/'`. Leave `base` alone unless the
site moves to a project subpath.

## Other bits

- **Editing name, tagline, nav, social links** → `src/consts.ts`.
- **Colours, glass, prose and code styling** → `src/styles/global.css`.
- **CV** → replace `public/resume.pdf`; it's linked from the homepage as `/resume.pdf`.
- **Social preview card** → `public/og-default.png`. Regenerate after changing the
  tagline with `node scripts/make-og.mjs`.
- **Outstanding TODOs** are marked `TODO(sid)` in the source and render as visible amber
  boxes on the site so you don't forget them. Search: `grep -rn "TODO(sid)" src/`.
