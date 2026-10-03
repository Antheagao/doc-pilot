import type { Metadata } from "next";
import Link from "next/link";
import ReviewQueueLink from "@/components/ReviewQueueLink";
import "./globals.css";

export const metadata: Metadata = {
  title: "doc-pilot",
  description:
    "Upload a document, watch a VLM extract structured data from it, then search and ask questions with citations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <Link href="/" className="site-title">
            doc-pilot
          </Link>
          <nav className="site-nav" aria-label="Main">
            <Link href="/search" className="nav-link">
              Search
            </Link>
            <Link href="/ask" className="nav-link">
              Ask
            </Link>
            <Link href="/dashboard" className="nav-link">
              Dashboard
            </Link>
            <ReviewQueueLink />
          </nav>
        </header>
        <main className="site-main">{children}</main>
      </body>
    </html>
  );
}
