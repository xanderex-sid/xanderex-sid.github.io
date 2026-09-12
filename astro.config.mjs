// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // GitHub Pages *user* site: served from the domain root, so base stays '/'.
  site: 'https://xanderex-sid.github.io',
  base: '/',
  trailingSlash: 'ignore',
  integrations: [sitemap()],
  vite: {
    // Cast: @tailwindcss/vite and Astro's bundled Vite ship slightly different
    // Plugin types. Runtime is fine; this only quiets `astro check`.
    plugins: [/** @type {any} */ (tailwindcss())],
  },
});
