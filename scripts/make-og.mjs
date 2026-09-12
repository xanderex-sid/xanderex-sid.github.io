import sharp from 'sharp';

const W = 1200;
const H = 630;

const overlay = Buffer.from(`
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <rect width="${W}" height="${H}" fill="#0e3b38" fill-opacity="0.88"/>
  <rect x="64" y="64" width="${W - 128}" height="${H - 128}" rx="28"
        fill="#ffffff" fill-opacity="0.94" stroke="#ffffff" stroke-width="2"/>
  <text x="112" y="268" font-family="Helvetica, Arial, sans-serif" font-size="66" font-weight="700" fill="#0f172a">Siddharth Mishra</text>
  <text x="112" y="330" font-family="Helvetica, Arial, sans-serif" font-size="31" fill="#2c3e4c">Data Scientist — Spatial AI and point clouds,</text>
  <text x="112" y="374" font-family="Helvetica, Arial, sans-serif" font-size="31" fill="#2c3e4c">moving toward robotics.</text>
  <rect x="112" y="424" width="120" height="6" rx="3" fill="#115e59"/>
  <text x="112" y="500" font-family="Helvetica, Arial, sans-serif" font-size="26" fill="#3d5160">xanderex-sid.github.io</text>
</svg>
`);

await sharp('src/assets/banner.jpg')
  .resize(W, H, { fit: 'cover', position: 'centre' })
  .blur(8)
  .composite([{ input: overlay, top: 0, left: 0 }])
  .png({ quality: 90, compressionLevel: 9 })
  .toFile('public/og-default.png');

console.log('wrote public/og-default.png');
