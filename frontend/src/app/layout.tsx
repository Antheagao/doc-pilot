import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "doc-pilot",
  description: "Upload a document, watch a VLM extract structured data from it.",
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
        </header>
        <main className="site-main">{children}</main>
      </body>
    </html>
  );
}
