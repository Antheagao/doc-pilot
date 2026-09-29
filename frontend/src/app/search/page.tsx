"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import {
  searchDocuments,
  ApiError,
  type SearchHit,
  type SearchMode,
  type SearchResponse,
} from "@/lib/api";

const MODES: { value: SearchMode; label: string }[] = [
  { value: "hybrid", label: "Hybrid" },
  { value: "dense", label: "Meaning" },
  { value: "lexical", label: "Keywords" },
];

const EXAMPLES = ["a light for my workspace", "receipts from Asheville", "27,82 EUR", "dry-erase pens"];

/** Why a hit ranked where it did: its rank on each side of hybrid search
 * (absent when that side didn't return it). */
function RankChips({ hit }: { hit: SearchHit }) {
  return (
    <span className="rank-chips">
      {hit.dense_rank !== null && <span className="rank-chip">meaning #{hit.dense_rank}</span>}
      {hit.lexical_rank !== null && <span className="rank-chip">keywords #{hit.lexical_rank}</span>}
    </span>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Mode and example clicks can fire a new search while one is in flight;
  // only the latest request may update the page.
  const latest = useRef(0);

  async function run(q: string, m: SearchMode) {
    if (!q.trim()) return;
    const request = ++latest.current;
    setLoading(true);
    setError(null);
    try {
      const result = await searchDocuments(q.trim(), m);
      if (request === latest.current) setResponse(result);
    } catch (err) {
      if (request !== latest.current) return;
      setError(
        err instanceof ApiError
          ? `API error: ${err.message}`
          : "Could not reach the API. Is the backend running?"
      );
    } finally {
      if (request === latest.current) setLoading(false);
    }
  }

  return (
    <div>
      <h1 className="page-title">Search</h1>
      <p className="review-intro">
        Plain-language search over every indexed document. Each result is the exact
        passage that matched, with the page it came from.
      </p>

      <form
        className="query-form"
        onSubmit={(e) => {
          e.preventDefault();
          run(query, mode);
        }}
      >
        <label htmlFor="search-query" className="visually-hidden">
          Search query
        </label>
        <input
          id="search-query"
          className="review-input query-input"
          placeholder="e.g. a light for my workspace"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="mode-toggle" role="radiogroup" aria-label="Search mode">
          {MODES.map((m) => (
            <button
              key={m.value}
              type="button"
              role="radio"
              aria-checked={mode === m.value}
              className={`mode-option${mode === m.value ? " active" : ""}`}
              onClick={() => {
                setMode(m.value);
                if (query.trim()) run(query, m.value);
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="example-row">
        <span className="example-label">Try:</span>
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="example-chip"
            onClick={() => {
              setQuery(example);
              run(example, mode);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {response && (
        <>
          <h2 className="section-heading">
            {response.results.length} result{response.results.length === 1 ? "" : "s"} ·{" "}
            <span className="muted-inline">{response.embedding_model}</span>
          </h2>
          {response.results.length === 0 ? (
            <div className="empty-state">
              Nothing matched. Documents become searchable once their index job finishes.
            </div>
          ) : (
            <ol className="hit-list">
              {response.results.map((hit) => (
                <li key={hit.chunk_id} className="review-row hit-row">
                  <div className="hit-head">
                    <Link href={`/documents/${hit.document_id}`} className="hit-doc">
                      {hit.filename}
                    </Link>
                    <span className="hit-page">page {hit.page_number}</span>
                    <RankChips hit={hit} />
                  </div>
                  <pre className="passage">{hit.text}</pre>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </div>
  );
}
