/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    // Override at deploy time if FastAPI runs on a different host.
    BACKEND_URL: process.env.BACKEND_URL || "http://localhost:8000",
  },
};
module.exports = nextConfig;
