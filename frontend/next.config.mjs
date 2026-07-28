/** @type {import('next').NextConfig} */
const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig = {
  transpilePackages: [],
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backendUrl}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
