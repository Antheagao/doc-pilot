import Link from "next/link";
import AccountMenu from "@/components/AccountMenu";
import ReviewQueueLink from "@/components/ReviewQueueLink";

// Every signed-in page: the header nav, the account menu, and <main>. The
// real sign-in check is the API's 401 on each fetch (lib/api.ts apiFetch).
export default function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
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
        <AccountMenu />
      </header>
      <main className="site-main">{children}</main>
    </>
  );
}
