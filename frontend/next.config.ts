import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Use environment variable for backend URL (works for both local and production)
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
    
    return [
      {
        source: "/api/chat",
        destination: `${backendUrl}/api/chat`,
      },
    ];
  },
};

export default nextConfig;
