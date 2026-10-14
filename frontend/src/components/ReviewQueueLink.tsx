"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { getReviewQueueCount, REVIEW_QUEUE_CHANGED } from "@/lib/api";

/** Header link to the review queue with a pending-count badge.
 *
 * Refetches on every route change and whenever a field is resolved
 * (REVIEW_QUEUE_CHANGED) rather than polling: the count only moves when an
 * extraction lands, which involves navigation soon after, or when a
 * reviewer resolves a field. A fetch failure just hides the badge. The link
 * itself must never depend on the API being up. */
export default function ReviewQueueLink() {
  const [count, setCount] = useState<number | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      getReviewQueueCount()
        .then((c) => {
          if (!cancelled) setCount(c);
        })
        .catch(() => {
          if (!cancelled) setCount(null);
        });
    };
    load();
    window.addEventListener(REVIEW_QUEUE_CHANGED, load);
    return () => {
      cancelled = true;
      window.removeEventListener(REVIEW_QUEUE_CHANGED, load);
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
