"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import {
  API_KEY_NAME_MAX_LENGTH,
  ApiError,
  createApiKey,
  getMe,
  listApiKeys,
  revokeApiKey,
  type ApiKey,
  type CreatedApiKey,
  type Me,
} from "@/lib/api";
import { formatTimestamp } from "@/lib/format";

function errorText(err: unknown): string {
  return err instanceof ApiError
    ? `API error: ${err.message}`
    : "Could not reach the API. Is the backend running?";
}

/** The one-time secret of a key just created: read-only text to copy, shown
 * until Done, and held in this component's state only. */
function NewKeySecret({ created, onDone }: { created: CreatedApiKey; onDone: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [copy, setCopy] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    input.select();
    // Selecting scrolls a long value to its end; show the "dpk_" start.
    input.scrollLeft = 0;
  }, []);

  async function copySecret() {
    try {
      await navigator.clipboard.writeText(created.secret);
      setCopy("copied");
    } catch {
      // No clipboard permission: leave the text selected for Ctrl+C.
      inputRef.current?.focus();
      inputRef.current?.select();
      setCopy("failed");
    }
  }

  return (
    <section className="review-row key-secret" aria-labelledby="key-secret-title">
      <h2 id="key-secret-title" className="key-secret-title">
        Key created: {created.api_key.name}
      </h2>
      <p className="key-secret-warning">Store it now; it is not shown again.</p>
      <div className="key-secret-row">
        <label htmlFor="key-secret-value" className="visually-hidden">
          New API key
        </label>
        <input
          id="key-secret-value"
          ref={inputRef}
          className="review-input key-secret-input"
          value={created.secret}
          readOnly
          spellCheck={false}
          autoComplete="off"
          onFocus={(event) => event.currentTarget.select()}
        />
        <button type="button" className="btn" onClick={() => void copySecret()}>
          Copy
        </button>
        <button type="button" className="btn btn-primary" onClick={onDone}>
          Done
        </button>
      </div>
      <p className="key-secret-status" role="status">
        {copy === "copied" && "Copied to the clipboard."}
        {copy === "failed" && "Could not copy. The key is selected: press Ctrl+C or Cmd+C."}
      </p>
    </section>
  );
}

/** A key's last-used time, or the plain fact that it never was. */
function lastUsed(key: ApiKey): string {
  return key.last_used_at ? formatTimestamp(key.last_used_at) : "Never";
}

export default function ApiKeysManager() {
  const [me, setMe] = useState<Me | null>(null);
  const [keys, setKeys] = useState<ApiKey[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  // The key whose Revoke is awaiting its confirmation, and the key being revoked.
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [revokingId, setRevokingId] = useState<string | null>(null);
  const [revokeError, setRevokeError] = useState<string | null>(null);
  const [revokedNote, setRevokedNote] = useState("");
  const nameRef = useRef<HTMLInputElement>(null);
  const listHeadingRef = useRef<HTMLHeadingElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);

  const reload = useCallback(async () => {
    try {
      setKeys(await listApiKeys());
      setLoadError(null);
    } catch (err) {
      setLoadError(errorText(err));
    }
  }, []);

  useEffect(() => {
    // Loads after the awaited fetches resolve, not synchronously in the effect
    // body; see the same pattern in app/(app)/page.tsx.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void reload();
    getMe()
      .then(setMe)
      .catch((err) => setLoadError(errorText(err)));
  }, [reload]);

  // Moving into the confirmation puts the keyboard on the safe choice.
  useEffect(() => {
    if (confirmingId !== null) cancelRef.current?.focus();
  }, [confirmingId]);

  async function onCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCreating(true);
    setCreateError(null);
    try {
      const result = await createApiKey(name.trim());
      setCreated(result);
      setName("");
      void reload();
    } catch (err) {
      // The name stays as typed.
      setCreateError(
        err instanceof ApiError && err.status === 403
          ? "Only an owner can create API keys."
          : errorText(err)
      );
    } finally {
      setCreating(false);
    }
  }

  function dismissSecret() {
    setCreated(null);
    // The name field is disabled until the secret panel is gone.
    requestAnimationFrame(() => nameRef.current?.focus());
  }

  async function onRevoke(key: ApiKey) {
    setRevokingId(key.id);
    setRevokeError(null);
    setRevokedNote("");
    try {
      await revokeApiKey(key.id);
      setRevokedNote(`Revoked ${key.name}.`);
    } catch (err) {
      // A 404 means it is already gone (revoked elsewhere); the reload shows it.
      if (!(err instanceof ApiError && err.status === 404)) setRevokeError(errorText(err));
    }
    setRevokingId(null);
    setConfirmingId(null);
    await reload();
    // The Revoke button this focus was on is gone.
    listHeadingRef.current?.focus();
  }

  const isOwner = me?.organization.role === "owner";
  const isMember = me !== null && !isOwner;
  const hasKeys = keys !== null && keys.length > 0;
  const columns = isOwner
    ? ["name", "key", "created", "used", "status", "actions"]
    : ["name", "key", "created", "used", "status"];

  return (
    <div className="keys-page">
      <h1 className="page-title">API keys</h1>
      <p className="review-intro">
        {me ? `Keys for ${me.organization.name}. ` : ""}
        An API key lets a script or the MCP server act in this organization without a browser
        sign-in. It works until you revoke it.
      </p>

      {loadError && (
        <div role="alert">
          <p className="error-banner">{loadError}</p>
        </div>
      )}

      {created && <NewKeySecret created={created} onDone={dismissSecret} />}

      {isOwner && (
        <form className="query-form key-create" onSubmit={onCreate}>
          <div className="form-field key-create-field">
            <label htmlFor="key-name">Key name</label>
            <input
              id="key-name"
              ref={nameRef}
              className="review-input"
              placeholder="e.g. Claude Code on my laptop"
              required
              maxLength={API_KEY_NAME_MAX_LENGTH}
              autoComplete="off"
              value={name}
              disabled={created !== null}
              aria-describedby={createError ? "key-create-error" : undefined}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={creating || created !== null || !name.trim()}
          >
            {creating ? "Creating…" : "Create key"}
          </button>
        </form>
      )}
      {isOwner && created && (
        <p className="auth-hint">Press Done above to create another key.</p>
      )}
      {isOwner && (
        <div role="alert" id="key-create-error">
          {createError && <p className="error-banner">{createError}</p>}
        </div>
      )}

      {isMember && (
        <p className="key-note">
          You are a member of {me.organization.name}. Only an owner can create or revoke API keys;
          you can see the list.
        </p>
      )}

      <h2 className="section-heading" ref={listHeadingRef} tabIndex={-1}>
        Keys
      </h2>

      <p className="key-status-note" role="status">
        {revokedNote}
      </p>

      {revokeError && (
        <div role="alert">
          <p className="error-banner">{revokeError}</p>
        </div>
      )}

      {keys === null && !loadError && <div className="empty-state">Loading…</div>}

      {keys !== null && !hasKeys && (
        <div className="empty-state">
          {isMember
            ? "No API keys yet. An owner can create one."
            : "No API keys yet. Name a key above and choose Create key; its secret is shown once."}
        </div>
      )}

      {hasKeys && (
        <table className="key-table">
          {/* Fixed column widths, so the row does not reflow while it asks
              whether to revoke. */}
          <colgroup>
            {columns.map((column) => (
              <col key={column} className={`key-col-${column}${isOwner ? "" : " member"}`} />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Key</th>
              <th scope="col">Created</th>
              <th scope="col">Last used</th>
              <th scope="col">Status</th>
              {isOwner && (
                <th scope="col">
                  <span className="visually-hidden">Actions</span>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => {
              const revoked = key.revoked_at !== null;
              return (
                <tr key={key.id} className={revoked ? "key-revoked" : undefined}>
                  <th scope="row" data-label="Name" className="key-name">
                    {key.name}
                  </th>
                  <td data-label="Key" className="key-prefix">
                    dpk_{key.prefix}…
                  </td>
                  <td data-label="Created">{formatTimestamp(key.created_at)}</td>
                  <td data-label="Last used">{lastUsed(key)}</td>
                  <td data-label="Status">
                    <span className={`status-chip ${revoked ? "status-chip--revoked" : "status-chip--extracted"}`}>
                      {revoked ? "Revoked" : "Active"}
                    </span>
                  </td>
                  {isOwner && (
                    <td className="key-actions">
                      {!revoked && confirmingId !== key.id && (
                        <button
                          type="button"
                          className="btn"
                          id={`revoke-${key.id}`}
                          aria-label={`Revoke ${key.name}`}
                          disabled={revokingId !== null}
                          onClick={() => {
                            setRevokeError(null);
                            setConfirmingId(key.id);
                          }}
                        >
                          Revoke
                        </button>
                      )}
                      {!revoked && confirmingId === key.id && (
                        <div className="key-confirm" role="group" aria-label={`Revoke ${key.name}`}>
                          <span className="key-confirm-text">
                            Anything using this key stops working at once.
                          </span>
                          <button
                            type="button"
                            className="btn btn-danger"
                            disabled={revokingId !== null}
                            onClick={() => void onRevoke(key)}
                          >
                            {revokingId === key.id ? "Revoking…" : "Revoke key"}
                          </button>
                          <button
                            type="button"
                            className="btn"
                            ref={cancelRef}
                            disabled={revokingId !== null}
                            onClick={() => {
                              setConfirmingId(null);
                              // Back to the button that opened the question.
                              requestAnimationFrame(() => document.getElementById(`revoke-${key.id}`)?.focus());
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      )}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
