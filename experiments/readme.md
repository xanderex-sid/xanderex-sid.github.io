# Experiment Template README

This README explains how to use the **experiment template** (`exp1.html`) for creating, documenting, and publishing experiments. It covers directory structure, required assets, how to include Excalidraw slides, and what to edit when creating a real experiment.

---

## 📂 Directory Structure
Every experiment should live in its own folder under `experiments/`. Example:

```
experiments/
  exp1/
    exp1.html                <-- main template (copy/duplicate per experiment)
    assets/
      images/
        hero.jpg             <-- hero image (cover)
        fig-1.png            <-- figure(s)
        fig-2.png
      excalidraw/
        slideList.json       <-- REQUIRED: slide list config
        slide1.svg           <-- exported SVG snapshot (preview)
        slide2.svg
        thumb1.png           <-- optional thumbnail
        exp1.json            <-- full Excalidraw JSON
      code/
        sample.py            <-- downloadable code snippets
```

---

## 📝 What to Edit When Creating a New Experiment

1. **Duplicate the folder**
   - Copy `exp1/` to `exp2/` (or whatever your experiment ID is).

2. **Update `expN.html`**
   - Change the `<h1>` experiment title.
   - Update **date** and **tags**.
   - Replace hero image (`assets/images/hero.jpg`).
   - Update figure gallery images if needed.
   - Insert text sections for:
     - **Abstract**
     - **Methodology**
     - **Results**
     - **Discussion / Next steps**

3. **Update `assets/excalidraw/slideList.json`**
   - Define your slides (SVG previews + JSON interactive version).

4. **Add code snippets**
   - Place code in `assets/code/`.
   - Reference them in the sidebar or body. The template already adds **Copy** and **Download** buttons.

5. **Replace gallery images**
   - Put experiment figures into `assets/images/`.
   - Add thumbnails and links in the gallery section.

6. **Adjust sidebar links**
   - Add links to datasets, checkpoints, or related experiments.

---

## 🎨 Using Excalidraw Slides

### Step 1: Create diagrams
- Make your diagram in [Excalidraw](https://excalidraw.com/).

### Step 2: Export files
- Export a **preview** of each slide as **SVG** or **PNG** → place under `assets/excalidraw/`.
- Save the **full JSON** (`File → Save to...`) → also place under `assets/excalidraw/`.

### Step 3: Update `slideList.json`
Example:
```json
[
  {"type":"svg","src":"assets/excalidraw/slide1.svg","title":"Overview","thumb":"assets/excalidraw/thumb1.png"},
  {"type":"svg","src":"assets/excalidraw/slide2.svg","title":"Mapping"},
  {"type":"json","src":"assets/excalidraw/exp1.json","title":"Interactive (Open in Excalidraw)"}
]
```

- `type`: one of `svg`, `png`, `jpg`, or `json`
- `src`: relative path to file
- `title`: label shown under viewer
- `thumb`: optional thumbnail image

### Step 4: Test viewer
- Open `expN.html` in browser.
- Use arrows/controls to navigate.
- For JSON slides: click **Open in Excalidraw** → diagram opens online.

---

## 💻 Code Snippets

- Code placed in `assets/code/` can be embedded inline:
  ```html
  <pre><code class="language-python"># example code
print("Hello Experiment")
</code></pre>
  ```
- Template provides **Copy** + **Download** buttons automatically.

---

## 🖼 Figures & Gallery

- Put images in `assets/images/`.
- Add thumbnails in the gallery:
```html
<div class="gallery">
  <a href="assets/images/fig-1.png" target="_blank">
    <img src="assets/images/fig-1.png" alt="Figure 1">
  </a>
</div>
```

---

## 📖 Features Already Included in Template
- Responsive layout (article + sidebar)
- Excalidraw slide viewer:
  - Prev/Next/Play controls
  - Thumbnails
  - Open JSON in Excalidraw
- Code blocks with Copy + Download
- Auto-calculated reading time
- Figure gallery

---

## ⚡ Best Practices

- Keep **hero image** consistent style across experiments.
- Always export **SVG previews** for diagrams → faster to load.
- Large assets (models, datasets) → host externally (S3, GDrive, HuggingFace) and link in sidebar.
- Maintain a short **README.md inside each experiment folder** summarizing:
  - Dataset
  - Commands to reproduce
  - External asset links

---

## ✅ Quick Checklist (Before Publishing a New Experiment)
- [ ] Duplicate template folder → rename
- [ ] Update title, date, tags in HTML
- [ ] Replace hero image
- [ ] Add figures in `assets/images/`
- [ ] Export Excalidraw slides and update `slideList.json`
- [ ] Add code snippets to `assets/code/`
- [ ] Update sidebar with downloads/resources
- [ ] Write experiment text (abstract, methodology, results, etc.)
- [ ] Test HTML in browser (slides, code copy/download, images)

