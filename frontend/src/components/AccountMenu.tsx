"use client";

import { useEffect, useId, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { getMe, logout, switchOrganization, type Me } from "@/lib/api";

/** The header's account control: who is signed in and where, the way to the
 * API keys page, and Sign out.
 *
 * A disclosure, not an ARIA menu: a button that shows a panel of ordinary
 * controls, so Tab walks it and Escape closes it. The organization switcher
 * appears only when the user belongs to more than one. The header must never
 * depend on the API being up, so a failed load renders no control at all
 * (a 401 sends the browser to the sign-in page from apiFetch). */
export default function AccountMenu() {
  const [me, setMe] = useState<Me | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  // The organization picked in the switcher; nothing changes until Switch, so
  // arrowing through the options does not move the session (WCAG 3.2.2).
  const [picked, setPicked] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const pathname = usePathname();

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      getMe()
        .then((fetched) => {
          if (!cancelled) setMe(fetched);
        })
        .catch(() => {
          if (!cancelled) setFailed(true);
        });
    };
    load();
    // A page restored from the back/forward cache is the page as it was
    // left, data and all; ask the API whether that session still exists (a
    // 401 sends the browser to the sign-in page).
    const onPageShow = (event: PageTransitionEvent) => {
      if (event.persisted) load();
    };
    window.addEventListener("pageshow", onPageShow);
    return () => {
      cancelled = true;
      window.removeEventListener("pageshow", onPageShow);
    };
  }, []);

  // Close on navigation. Adjusting state while rendering, keyed on the
  // previous pathname, avoids an effect that only sets state.
  const [lastPathname, setLastPathname] = useState(pathname);
  if (pathname !== lastPathname) {
    setLastPathname(pathname);
    setOpen(false);
  }

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  if (failed) return null;
  if (!me) return <span className="account-placeholder" aria-hidden="true" />;
  const choice = picked ?? me.organization.id;

  async function signOut() {
    setBusy(true);
    setError(null);
    try {
      await logout();
      // A full navigation, replacing this entry: nothing signed-in stays in
      // memory, and Back does not return to a page of someone's data.
      window.location.replace("/login");
    } catch {
      setError("Could not sign out. Try again.");
      setBusy(false);
    }
  }

  async function switchTo(organizationId: string) {
    setBusy(true);
    setError(null);
    try {
      await switchOrganization(organizationId);
      // Every page's data belongs to the old organization.
      window.location.assign("/");
    } catch {
      setError("Could not switch organization. Try again.");
      setBusy(false);
    }
  }

  return (
    <div
      className="account-menu"
      ref={rootRef}
      onBlur={(event) => {
        // Tabbing out of the panel closes it; a click on its plain text
        // (relatedTarget null) is left to the pointerdown handler.
        const next = event.relatedTarget as Node | null;
        if (next && !event.currentTarget.contains(next)) setOpen(false);
      }}
    >
      <button
        type="button"
        ref={triggerRef}
        className="account-trigger"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="account-avatar" aria-hidden="true">
          {me.email.charAt(0).toUpperCase()}
        </span>
        <span className="account-trigger-label">
          <span className="visually-hidden">Account: </span>
          {me.email}
        </span>
      </button>

      <div className="account-panel" id={panelId} hidden={!open}>
        <div className="account-identity">
          <div className="account-email">{me.email}</div>
          <div className="account-org">
            {me.organization.name}
            <span className="account-role">{me.organization.role}</span>
          </div>
        </div>

        {me.organizations.length > 1 && (
          <div className="account-switcher">
            <label htmlFor={`${panelId}-org`}>Organization</label>
            <div className="account-switcher-row">
              <select
                id={`${panelId}-org`}
                className="review-input"
                value={choice}
                disabled={busy}
                onChange={(event) => setPicked(event.target.value)}
              >
                {me.organizations.map((organization) => (
                  <option key={organization.id} value={organization.id}>
                    {organization.name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="btn"
                disabled={busy || choice === me.organization.id}
                onClick={() => void switchTo(choice)}
              >
                Switch
              </button>
            </div>
          </div>
        )}

        <Link href="/settings/api-keys" className="account-item" onClick={() => setOpen(false)}>
          API keys
        </Link>
        <button type="button" className="account-item" disabled={busy} onClick={() => void signOut()}>
          Sign out
        </button>
        <div role="alert">{error && <p className="account-error">{error}</p>}</div>
      </div>
    </div>
  );
}
