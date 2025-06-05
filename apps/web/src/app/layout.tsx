import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Providers } from './providers';
import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';
import { Toaster } from '@/components/ui/Toaster';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: {
    default: 'StratosHub - Decentralized AI Agent Marketplace',
    template: '%s | StratosHub',
  },
  description: 'Enterprise-grade decentralized marketplace for AI agents on Solana blockchain. Deploy, discover, and execute AI models with guaranteed performance and transparent pricing.',
  keywords: [
    'AI agents',
    'artificial intelligence',
    'blockchain',
    'Solana',
    'decentralized',
    'marketplace',
    'machine learning',
    'smart contracts',
    'Web3',
    'DeFi'
  ],
  authors: [{ name: 'StratosHub Team' }],
  creator: 'StratosHub',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://stratoshub.com',
    title: 'StratosHub - Decentralized AI Agent Marketplace',
    description: 'Enterprise-grade decentralized marketplace for AI agents on Solana blockchain.',
    siteName: 'StratosHub',
    images: [
      {
        url: '/banner.jpeg',
        width: 1200,
        height: 630,
        alt: 'StratosHub Platform',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'StratosHub - Decentralized AI Agent Marketplace',
    description: 'Enterprise-grade decentralized marketplace for AI agents on Solana blockchain.',
    images: ['/banner.jpeg'],
    creator: '@stratoshub',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_VERIFICATION,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link rel="dns-prefetch" href="https://api.mainnet-beta.solana.com" />
        <link rel="dns-prefetch" href="https://gateway.pinata.cloud" />
        <meta name="theme-color" content="#000000" />
        <meta name="color-scheme" content="dark light" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        
        {/* Structured Data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'WebApplication',
              name: 'StratosHub',
              description: 'Enterprise-grade decentralized marketplace for AI agents on Solana blockchain.',
              url: 'https://stratoshub.com',
              applicationCategory: 'BusinessApplication',
              operatingSystem: 'Web',
              offers: {
                '@type': 'Offer',
                category: 'AI Services',
              },
              author: {
                '@type': 'Organization',
                name: 'StratosHub',
              },
            }),
          }}
        />
      </head>
      <body className={`${inter.className} min-h-screen bg-background antialiased`}>
        <Providers>
          <div className="relative flex min-h-screen flex-col">
            <Header />
            <main className="flex-1">
              {children}
            </main>
            <Footer />
          </div>
          <Toaster />
        </Providers>
        
        {/* Analytics */}
        {process.env.NEXT_PUBLIC_ANALYTICS_ID && (
          <>
            <script
              async
              src={`https://www.googletagmanager.com/gtag/js?id=${process.env.NEXT_PUBLIC_ANALYTICS_ID}`}
            />
            <script
              dangerouslySetInnerHTML={{
                __html: `
                  window.dataLayer = window.dataLayer || [];
                  function gtag(){dataLayer.push(arguments);}
                  gtag('js', new Date());
                  gtag('config', '${process.env.NEXT_PUBLIC_ANALYTICS_ID}', {
                    page_title: document.title,
                    page_location: window.location.href,
                  });
                `,
              }}
            />
          </>
        )}
      </body>
    </html>
  );
} 