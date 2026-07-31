"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { getReviewQueueCount } from "@/lib/api";

/** Header link to the review queue with a pending-count badge.
 *
 * Refetches on every route change rather than polling: the count only
 * moves when an extraction lands or a reviewer resolves a field, and both
 * of those involve navigation soon after. A fetch failure just hides the
 * badge — the link itself must never depend on the API being up. */
export default function ReviewQueueLink() {
  const [count, setCount] = useState<number | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    let cancelled = false;
    getReviewQueueCount()
      .then((c) => {
        if (!cancelled) setCount(c);
      })
      .catch(() => {
        if (!cancelled) setCount(null);
      });
    return () => {
      cancelled = true;
    };
  }, [pathname]);

  return (
    <Link href="/review" className="review-nav-link">
      Review
      {count !== null && count > 0 && (
        <span className="nav-badge">{count}</span>
      )}
    </Link>
  );
}
