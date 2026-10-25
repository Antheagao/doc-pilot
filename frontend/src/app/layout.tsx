import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "doc-pilot",
  description:
    "Upload a document, watch a VLM extract structured data from it, then search and ask questions with citations.",
};

// html and body only: the header belongs to (app)/layout.tsx, so the sign-in
// page (outside that group) renders without it and polls nothing.
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
