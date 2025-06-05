/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: true,
    serverComponentsExternalPackages: ['@solana/web3.js'],
  },
  
  // Environment variables
  env: {
    NEXT_PUBLIC_SOLANA_NETWORK: process.env.NEXT_PUBLIC_SOLANA_NETWORK || 'devnet',
    NEXT_PUBLIC_RPC_ENDPOINT: process.env.NEXT_PUBLIC_RPC_ENDPOINT,
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:3001',
    NEXT_PUBLIC_IPFS_GATEWAY: process.env.NEXT_PUBLIC_IPFS_GATEWAY || 'https://gateway.pinata.cloud/ipfs',
    NEXT_PUBLIC_ANALYTICS_ID: process.env.NEXT_PUBLIC_ANALYTICS_ID,
    NEXT_PUBLIC_SENTRY_DSN: process.env.NEXT_PUBLIC_SENTRY_DSN,
  },

  // Webpack configuration for Solana and crypto dependencies
  webpack: (config, { isServer }) => {
    // Handle Node.js modules in browser
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        crypto: require.resolve('crypto-browserify'),
        stream: require.resolve('stream-browserify'),
        http: require.resolve('stream-http'),
        https: require.resolve('https-browserify'),
        zlib: require.resolve('browserify-zlib'),
        path: require.resolve('path-browserify'),
        os: require.resolve('os-browserify/browser'),
        url: require.resolve('url'),
        assert: require.resolve('assert'),
        buffer: require.resolve('buffer'),
        process: require.resolve('process/browser'),
      };
    }

    // Add support for WebAssembly
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    // Optimize bundle size for Solana libraries
    config.module.rules.push({
      test: /\.m?js$/,
      resolve: {
        fullySpecified: false,
      },
    });

    return config;
  },

  // Image optimization
  images: {
    domains: [
      'gateway.pinata.cloud',
      'ipfs.io',
      'cloudflare-ipfs.com',
      'arweave.net',
    ],
    formats: ['image/webp', 'image/avif'],
  },

  // Headers for security and performance
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
      {
        source: '/api/(.*)',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: process.env.NODE_ENV === 'production' 
              ? 'https://stratoshub.com' 
              : '*',
          },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET, POST, PUT, DELETE, OPTIONS',
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'Content-Type, Authorization, X-Requested-With',
          },
        ],
      },
    ];
  },

  // Redirects for SEO and user experience
  async redirects() {
    return [
      {
        source: '/marketplace',
        destination: '/agents',
        permanent: true,
      },
      {
        source: '/docs',
        destination: '/documentation',
        permanent: true,
      },
    ];
  },

  // Rewrites for API proxying
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/v1/:path*`,
      },
    ];
  },

  // Compiler options
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Output configuration
  output: 'standalone',
  
  // Performance optimizations
  poweredByHeader: false,
  generateEtags: false,
  
  // Internationalization
  i18n: {
    locales: ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh'],
    defaultLocale: 'en',
  },
};

module.exports = nextConfig; 