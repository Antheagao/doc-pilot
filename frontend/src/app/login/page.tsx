import type { Metadata } from "next";
import { Suspense } from "react";
import LoginForm from "@/components/LoginForm";

export const metadata: Metadata = {
  title: "Sign in - doc-pilot",
};

// Outside the (app) group: no header, no account menu, and nothing that polls
// the API. LoginForm reads ?next= with useSearchParams, which on a prerendered
// page needs a Suspense boundary.
export default function LoginPage() {
  return (
    <main className="auth-main">
      <div className="auth-brand">doc-pilot</div>
      <Suspense fallback={<div className="auth-card" aria-busy="true" />}>
        <LoginForm />
      </Suspense>
    </main>
  );
}
