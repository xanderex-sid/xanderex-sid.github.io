import { defineCollection, z } from 'astro:content';
import { glob, file } from 'astro/loaders';

/** Blog posts: one .mdx file per post in src/content/blog/. */
const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      date: z.coerce.date(),
      description: z.string(),
      tags: z.array(z.string()).default([]),
      /** Optional card/hero image. Lives in src/content/blog/ next to the post. */
      cover: image().optional(),
      coverAlt: z.string().optional(),
      draft: z.boolean().default(false),
    }),
});

/** Dated "recent news" entries, newest first. */
const news = defineCollection({
  loader: file('./src/data/news.json'),
  schema: z.object({
    id: z.string(),
    date: z.coerce.date(),
    text: z.string(),
    href: z.string().optional(),
    linkText: z.string().optional(),
  }),
});

/** Academic publications. */
const publications = defineCollection({
  loader: file('./src/data/publications.json'),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    authors: z.array(z.string()),
    /** Highlight this exact string in the author list. */
    highlightAuthor: z.string().optional(),
    venue: z.string(),
    year: z.number(),
    doi: z.string().optional(),
    links: z
      .array(z.object({ label: z.string(), href: z.string() }))
      .default([]),
    note: z.string().optional(),
  }),
});

/** Project blocks on /projects. */
const projects = defineCollection({
  loader: file('./src/data/projects.json'),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    description: z.string(),
    /** Path under public/, e.g. "/videos/foo.mp4". Omit until you have one. */
    videoSrc: z.string().optional(),
    /** Required fallback image, shown when there is no video. */
    posterSrc: z.string(),
    posterAlt: z.string(),
    repoUrl: z.string().optional(),
    tags: z.array(z.string()).default([]),
    comingSoon: z.boolean().default(false),
    order: z.number().default(0),
  }),
});

export const collections = { blog, news, publications, projects };
