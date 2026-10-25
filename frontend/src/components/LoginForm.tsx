"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ApiError,
  EMAIL_MAX_LENGTH,
  PASSWORD_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  getMe,
  login,
  safeNextPath,
} from "@/lib/api";

/** "42 seconds" up to a minute and a half, then whole minutes. */
function waitPhrase(seconds: number): string {
  if (seconds <= 90) return `${seconds} ${seconds === 1 ? "second" : "seconds"}`;
  return `${Math.ceil(seconds / 60)} minutes`;
}

/** The sentence for a failed sign-in. Switches on the status, not on the
 * API's wording; the 401 is deliberately one message for every cause. */
function failureMessage(err: unknown): string {
  if (!(err instanceof ApiError)) return "Could not reach the API. Is the backend running?";
  switch (err.status) {
    case 401:
      return "The email or password is incorrect.";
    case 403:
      return "The API refused this sign-in because this page's address is not an allowed origin.";
    case 422:
      return `Enter your email and a password of ${PASSWORD_MIN_LENGTH} to ${PASSWORD_MAX_LENGTH} characters.`;
    case 429:
      return err.retryAfterSeconds === null
        ? "Too many sign-in attempts. Wait a minute and try again."
        : `Too many sign-in attempts. Try again in ${waitPhrase(err.retryAfterSeconds)}.`;
    default:
      return `API error: ${err.message}`;
  }
}

export default function LoginForm() {
  const router = useRouter();
  const next = safeNextPath(useSearchParams().get("next"));
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Already signed in (a bookmark of /login, or Back to it): go on. apiFetch
  // does not redirect from this page, so a 401 here just leaves the form.
  useEffect(() => {
    let cancelled = false;
    getMe()
      .then(() => {
        if (!cancelled) router.replace(next);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [router, next]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError(null);
    try {
      await login(email, password);
      // Stay pending: the page is about to be replaced.
      router.replace(next);
    } catch (err) {
      // The input stays as typed so a typo is one fix away.
      setError(failureMessage(err));
      setPending(false);
    }
  }

  return (
    <div className="auth-card">
      <h1 className="page-title">Sign in</h1>
      <p className="auth-hint">Accounts are created by the person who runs this instance.</p>

      <form onSubmit={onSubmit} className="auth-form">
        <div className="form-field">
          <label htmlFor="login-email">Email</label>
          <input
            id="login-email"
            name="email"
            type="email"
            inputMode="email"
            autoComplete="username"
            autoCapitalize="none"
            spellCheck={false}
            required
            maxLength={EMAIL_MAX_LENGTH}
            className="review-input"
            value={email}
            aria-describedby={error ? "login-error" : undefined}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>
        <div className="form-field">
          <label htmlFor="login-password">Password</label>
          <input
            id="login-password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
            minLength={PASSWORD_MIN_LENGTH}
            maxLength={PASSWORD_MAX_LENGTH}
            className="review-input"
            value={password}
            aria-describedby={error ? "login-error" : undefined}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>

        {/* Mounted before it has text, so a screen reader announces the change. */}
        <div role="alert" id="login-error">
          {error && <p className="error-banner">{error}</p>}
        </div>

        <button type="submit" className="btn btn-primary auth-submit" disabled={pending}>
          {pending ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
