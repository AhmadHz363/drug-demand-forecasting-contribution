import type { NextConfig } from "next";

/** Training and receipt ingestion can run for a long time on large catalogs. */
const PROXY_TIMEOUT_MS = 2 * 60 * 60 * 1000;

const nextConfig: NextConfig = {
  experimental: {
    proxyTimeout: PROXY_TIMEOUT_MS,
    /** Default 10MB truncates large spreadsheet uploads proxied through Next. */
    proxyClientMaxBodySize: "500mb",
  },
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
