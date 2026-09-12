/** Site-wide constants. Edit these rather than hunting through components. */

export const SITE = {
  title: 'Siddharth Mishra',
  tagline: 'Data Scientist — Spatial AI and point clouds, moving toward robotics.',
  description:
    'Siddharth Mishra — Data Scientist at Deepmatrix working on LiDAR point-cloud segmentation and geospatial object detection, now moving toward robotics.',
  url: 'https://xanderex-sid.github.io',
  author: 'Siddharth Mishra',
  locale: 'en',
  /** Used for og:image; generated at public/og-default.png by scripts/make-og.mjs. */
  ogImage: '/og-default.png',
} as const;

export const NAV_LINKS = [
  { label: 'Home', href: '/' },
  { label: 'Projects', href: '/projects' },
] as const;

/**
 * Social links. URLs below were taken from the hyperlinks embedded in
 * Resume.pdf, so they match the resume exactly.
 */
export const SOCIALS = [
  { label: 'GitHub', href: 'https://github.com/xanderex-sid', icon: 'github' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/iamsiddharthmishra/', icon: 'linkedin' },
  { label: 'LeetCode', href: 'https://leetcode.com/XANDEREX/', icon: 'leetcode' },
  { label: 'Email', href: 'mailto:siddharthmishra.work@gmail.com', icon: 'mail' },
  // TODO(sid): replace with your real Google Scholar profile URL, or delete this
  // entry if you do not have one yet. Left visible on purpose so you notice it.
  { label: 'Google Scholar', href: '#TODO-google-scholar', icon: 'scholar', todo: true },
] as const;

export const RESUME_URL = '/resume.pdf';
