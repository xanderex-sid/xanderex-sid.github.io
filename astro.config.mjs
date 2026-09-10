// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

/**
 * Shiki transformer that surfaces the code fence's metadata to the DOM.
 *
 * Writing ```python title="losses.py" showLineNumbers puts `data-filename`,
 * `data-lang` and `data-line-numbers` on the <pre>. CodeBlockEnhancer.astro
 * reads those attributes at runtime to build the header bar and copy button,
 * and global.css reads data-line-numbers to switch on the line gutter.
 */
/** @type {import('shiki').ShikiTransformer} */
const codeMetaTransformer = {
  name: 'code-meta',
  pre(node) {
    const raw = /** @type {{ __raw?: string } | undefined} */ (this.options.meta)?.__raw ?? '';
    const title = raw.match(/(?:title|filename)="([^"]+)"/);
    if (title) node.properties['data-filename'] = title[1];
    if (/\bshowLineNumbers\b/.test(raw)) node.properties['data-line-numbers'] = 'true';
    node.properties['data-lang'] = this.options.lang ?? 'text';
  },
};

export default defineConfig({
  // GitHub Pages *user* site: served from the domain root, so base stays '/'.
  site: 'https://xanderex-sid.github.io',
  base: '/',
  trailingSlash: 'ignore',
  integrations: [mdx(), sitemap()],
  vite: {
    // Cast: @tailwindcss/vite and Astro's bundled Vite ship slightly different
    // Plugin types. Runtime is fine; this only quiets `astro check`.
    plugins: [/** @type {any} */ (tailwindcss())],
  },
  markdown: {
    shikiConfig: {
      // Solid dark panel against the light sky page — high contrast, no blur.
      theme: 'github-dark-default',
      wrap: false,
      transformers: [codeMetaTransformer],
    },
  },
});
