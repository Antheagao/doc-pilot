import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enables the .next/standalone output layout consumed by frontend/Dockerfile.
  // Additive: does not change `npm run dev` or a plain `npm run build`.
  output: "standalone",
};

export default nextConfig;
