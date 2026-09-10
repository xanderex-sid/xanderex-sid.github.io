/**
 * Crops the full-body profile photo down to a face-centred square avatar.
 *
 * src/assets/profile.jpg is a 1280x1333 full-body shot, so a plain object-cover
 * square shows almost the whole frame and the face ends up tiny. This bakes a
 * proper headshot crop instead. Re-run after replacing profile.jpg:
 *   node scripts/make-avatar.mjs
 */
import sharp from 'sharp';

const SRC = 'src/assets/profile.jpg';
const OUT = 'src/assets/profile-avatar.jpg';

// Face sits around (735, 400) in the 1280x1333 original; this square keeps the
// head plus a little shoulder.
const CROP = { left: 405, top: 90, width: 660, height: 660 };

const meta = await sharp(SRC).metadata();
if (
  CROP.left + CROP.width > (meta.width ?? 0) ||
  CROP.top + CROP.height > (meta.height ?? 0)
) {
  throw new Error(
    `Crop ${JSON.stringify(CROP)} falls outside ${meta.width}x${meta.height}. ` +
      'Adjust CROP for the new photo.'
  );
}

await sharp(SRC)
  .extract(CROP)
  .resize(512, 512, { fit: 'cover' })
  .jpeg({ quality: 90, mozjpeg: true })
  .toFile(OUT);

console.log(`wrote ${OUT} (512x512 from ${JSON.stringify(CROP)})`);
