// -------------------------------------------------------------
// File: next.config.ts (optional)
/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "image.tmdb.org" },
      { protocol: "https", hostname: "m.media-amazon.com" },
    ],
  },
};
export default nextConfig;
