# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: `/Users/siddharthmishra/Downloads/CLAUDE.md` describes an unrelated point-cloud
> segmentation project. It loads into context only because this repo sits inside `Downloads/`.
> Ignore it here.

## What this repo is

Siddharth Mishra's personal site — Astro 7 + Tailwind v4, deployed to GitHub Pages at
<https://xanderex-sid.github.io>. Two pages: a research-profile homepage and a video-driven
projects page, plus a 404.

There is deliberately **no blog**. It was removed in September 2026 along with MDX, Shiki and the
code-block styling, because Sid isn't writing posts. Don't reintroduce any of it unless asked.

`README.md` is the user-facing guide (add a project / news item / publication, deploy).
This file covers things that aren't obvious from the code.

## Commands

```bash
npm run dev     # localhost:4321
npm run build   # → dist/
npm run check   # astro check (expect 0 errors, 0 warnings)
```

Node 22 was installed via `brew install node@22` and is **keg-only** — it is not on the default
PATH. Prefix commands:

```bash
export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
```

## Architecture notes

**Content is data-driven by design.** News, publications and projects are JSON in `src/data/`,
loaded through typed collections in `src/content.config.ts`. Adding one should never require
touching a component. The one exception is research interests, which live in the `research`
object at the top of `src/pages/index.astro`.

Import `z` from `astro/zod`, not from `astro:content` — the latter is deprecated and removed in
Astro 8.

**`videoSrc: "TODO"` is a deliberate sentinel.** `VideoBlock.astro` treats any `videoSrc`
starting with `TODO` as absent and renders the poster instead. This keeps the TODO visible in
`projects.json` rather than hiding it in a comment (JSON has none). Don't "fix" it by removing
the field.

**The hero is a two-column grid.** On sm+ it is
`grid-cols-[minmax(0,15rem)_minmax(0,1fr)]`: the photo fills the full height of the block on the
left, the banner strip and text sit on the right. The photo matches the text column's height
purely via the grid's default `align-items: stretch` — adding `items-start`/`items-center` there
will collapse it. The photo itself is `absolute inset-0 object-cover` inside a relative cell,
which is what lets it fill an auto height. Below sm the grid collapses to one column and the
photo gets an explicit `h-64`, since there is no sibling to stretch against.

**The hero photo uses `object-position: 57% 22%`.** `src/assets/profile.jpg` is a near-square
full-body shot, so a centred crop lands on the shirt. There is no baked crop file any more — the
tall column suits the original framing and object-position does the rest. Retune those percentages
if you replace the photo.

`scripts/make-og.mjs` generates `public/og-default.png`; re-run after changing the tagline in
`src/consts.ts`.

## Research interests — robotics-first, CV-grounded

Sid is applying to **hands-on robotics MS programs** from a pure software-AI background, so the
research-interests section leads with robotics. Each area pairs a forward-looking `interest` with a
`fromWork` line that quotes an achievement from the CV, in the CV's own wording (PointNet-EdgeConv,
RandLA-Net, Point Transformer V3, FPS / Morton code, INSID3 & DINOv3, the DECoN 2025 SoC paper).

The rule: `interest` may be aspirational; `fromWork` may not. Never add experience the CV doesn't
show — ROS, SLAM, control, embedded deployment, hardware builds. An admissions reader will check
those claims against the CV and in interview.

## Design system — readability is the explicit priority

The backdrop is a **flat navy green** (`--color-navy: #0e3b38`) — deliberately uniform, with no
light-to-dark gradient. The only tonal variation comes from the drifting haze blobs. Panels keep
the frosted-glass treatment, so body copy is still dark ink on a light surface.

Standing rule: **when aesthetics and readability conflict, readability wins.** Concretely:

- Glass goes on cards, nav and section containers — never directly behind long-form text.
- Under 640px `.glass` trades blur for opacity and `.glass-strong` drops `backdrop-filter`
  entirely — heavy blur is expensive to composite on mobile.

**Glass opacity is load-bearing, not decorative.** Because the backdrop is dark, `--glass-bg` sits
at 0.88 rather than the 0.66 that worked over a light sky. Drop it back and the composite pulls
muted text below AA. If you change the backdrop or any ink token, re-run the audit rather than
eyeballing it — translucent panels cannot be checked from CSS alone, because the effective
background is a composite. The method that works: screenshot with all glyphs set to
`color: transparent`, sample the real pixel behind each text run, then compute the ratio.

Last measured (September 2026): 0 failures across home and projects, minimum **5.6:1**; the
research-interests section alone measures 6.09:1 minimum.

The audit only samples text inside the viewport. Long pages need a tall viewport or an audit
scoped to one section — the research-interests section sits below the fold and was silently
skipped by the default 900px audit until it was scoped explicitly.

`--color-ink-faint` (#4d5a70) is the tightest and is for meta text only — don't use it for body
copy.

## Accessibility invariants

- `prefers-reduced-motion` must disable **both** the haze drift and video autoplay. `VideoBlock`
  re-checks the media query in JS before calling `play()` and exposes native `controls` instead.
  Verified under emulation.
- All pages measured 0 horizontal overflow at 390px. The haze blobs extend past the viewport but
  are inside a `position: fixed` container with `overflow: hidden`, so they don't create scroll.
- Focus rings come from a single `:focus-visible` rule in `global.css`. Don't remove outlines
  locally.

## Outstanding TODOs

Content the resume didn't cover is marked `TODO(sid)` in source and renders as visible amber
boxes on the site (news placeholder, project videos, Google Scholar link).
Find them with `grep -rn "TODO(sid)" src/`. Don't invent replacements — these exist because
the information wasn't available.

## Deployment

`.github/workflows/deploy.yml` builds via `withastro/action` and publishes with
`actions/deploy-pages` on push to `main`. The repo's Pages source must be set to **GitHub
Actions** in Settings → Pages, otherwise the workflow builds and nothing goes live.

Work so far is on the `redesign` branch, not `main`.
