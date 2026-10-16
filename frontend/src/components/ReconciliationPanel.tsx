"use client";

import Link from "next/link";
import type {
  CheckStatus,
  Evidence,
  Reconciliation,
  ReconciliationCheck,
  ReconciliationOperand,
} from "@/lib/api";
import { formatAmount, formatPage, formatQuote } from "@/lib/format";
import {
  amountText,
  equationView,
  humanize,
  includedNote,
  isComputedBase,
  pathLabel,
  quotePath,
  suggestionWhy,
  type EquationView,
} from "@/lib/reconciliation";

const STATUS_WORD: Record<CheckStatus, string> = {
  pass: "Passes",
  fail: "Fails",
  skipped: "Skipped",
};

// The mark is decoration: the word beside it carries the meaning.
const STATUS_MARK: Record<CheckStatus, string> = {
  pass: "✓",
  fail: "✕",
  skipped: "–",
};

// Per-line checks ("line_math[3]") are summarized, not listed.
const LINE_CHECK = /^line_math\[\d+\]$/;

// What the quieter checks are called when a pass is summarized.
function quietName(id: string): string {
  if (id.startsWith("non_negative:")) return "no negative amounts";
  if (id === "tax_rate") return "tax rate";
  if (id === "lines_present") return "line items present";
  if (id === "due_date_order") return "due date not before the document date";
  if (id.startsWith("quote:")) return "values match their quotes";
  return id.replace(/[_:]/g, " ");
}

// The quote checks have their own group; the arithmetic is everything else.
function isQuote(check: ReconciliationCheck): boolean {
  return check.kind === "quote";
}

// The report's own rule for a set of checks: any failure fails it, else any
// pass passes it, else nothing was judged.
function statusOf(checks: ReconciliationCheck[]): CheckStatus {
  if (checks.some((c) => c.status === "fail")) return "fail";
  return checks.some((c) => c.status === "pass") ? "pass" : "skipped";
}

function StatusMark({ status }: { status: CheckStatus }) {
  return (
    <span className={`recon-mark ${status}`}>
      <span aria-hidden="true">{STATUS_MARK[status]}</span> {STATUS_WORD[status]}
    </span>
  );
}

/** A tag for a number the document did not print as such. An extracted or
 * reviewed amount carries none. */
function SourceTag({ source }: { source: ReconciliationOperand["source"] }) {
  if (source !== "computed" && source !== "assumed") return null;
  return <span className="computed-tag">{source}</span>;
}

interface PickProps {
  /** The report path of the selected operand; its field lights up above. */
  picked: string | null;
  onPick: (path: string) => void;
}

/** An operand of a check: a button that highlights its field in the
 * extraction above and scrolls to it. `label` is shown beside the number
 * when the number alone would not say what it is. */
function Operand({
  operand,
  label,
  picked,
  onPick,
}: PickProps & { operand: ReconciliationOperand; label?: string }) {
  const text = amountText(operand);
  const named = label ?? pathLabel(operand.path);
  // A missing amount has nothing to point at: it is said, not selectable.
  if (operand.value === null) {
    return (
      <span className="operand missing">
        {text}
        <span className="operand-label">{named}</span>
      </span>
    );
  }
  // A base worked out for a tax line that printed none has no cell in the
  // tax lines table: said, not selectable.
  if (isComputedBase(operand)) {
    return (
      <>
        <span className="operand static">
          {text}
          <span className="visually-hidden"> {named}</span>
        </span>
        <SourceTag source={operand.source} />
      </>
    );
  }
  const pressed = picked === operand.path;
  const tagged = operand.source === "computed" || operand.source === "assumed";
  return (
    <>
      <button
        type="button"
        className={`operand${pressed ? " picked" : ""}`}
        aria-pressed={pressed}
        aria-label={`${text} ${named}${tagged ? `, ${operand.source}` : ""}`}
        onClick={() => onPick(operand.path)}
      >
        {text}
        {label && <span className="operand-label">{label}</span>}
      </button>
      <SourceTag source={operand.source} />
    </>
  );
}

function Equation({ view, ...pick }: PickProps & { view: EquationView }) {
  return (
    <p className="recon-equation">
      {view.collapsed !== null ? (
        <span className="recon-sum">{view.collapsed}</span>
      ) : (
        view.terms.map((term, i) => (
          <span key={`${term.operand.path}-${i}`} className="recon-term">
            {(i > 0 || term.sign === "-") && <span className="recon-sign">{term.sign}</span>}
            {term.coef !== null && (
              <span className="recon-sign">
                {term.coef} {term.rate ? "of" : "×"}
              </span>
            )}
            <Operand operand={term.operand} {...pick} />
          </span>
        ))
      )}
      {view.included.map((operand) => (
        <span key={operand.path} className="recon-term">
          <span className="recon-sign">(</span>
          <Operand
            operand={operand}
            label={`${pathLabel(operand.path)} ${includedNote(operand)}`}
            {...pick}
          />
          <span className="recon-sign">)</span>
        </span>
      ))}
      {/* Each sign stays on the line of the number it introduces. */}
      {view.relation === "=" ? (
        <span className="recon-term">
          <span className="recon-sign">=</span>
          <Operand operand={view.rhs} label={pathLabel(view.rhs.path)} {...pick} />
        </span>
      ) : (
        <>
          <span className="recon-term">
            <span className="recon-sign">=</span>
            <span className="recon-expected">{view.expected}</span>
          </span>
          <span className="recon-term">
            <span className="recon-sign">
              <span aria-hidden="true">{view.relation}</span>
              <span className="visually-hidden">
                {view.relation === "≠" ? "does not equal" : "is about"}
              </span>
            </span>
            <Operand operand={view.rhs} label={pathLabel(view.rhs.path)} {...pick} />
          </span>
        </>
      )}
      {view.off !== null && view.relation === "≠" && (
        <span className="recon-off">off by {view.off}</span>
      )}
    </p>
  );
}

function CheckRow({ check, ...pick }: PickProps & { check: ReconciliationCheck }) {
  const view = equationView(check);
  const failing = check.status === "fail";
  const operands = view === null && failing ? check.terms : [];
  return (
    <li className={`recon-check ${check.status}`}>
      <div className="recon-check-head">
        <StatusMark status={check.status} />
        <span className="recon-formula">{humanize(check.formula)}</span>
      </div>
      {view !== null && <Equation view={view} {...pick} />}
      {view === null && failing && check.detail && (
        <p className="recon-detail">{humanize(check.detail)}</p>
      )}
      {operands.length > 0 && (
        <p className="recon-operands">
          {operands.map(({ operand }) => (
            <span key={operand.path} className="recon-term">
              <Operand operand={operand} label={pathLabel(operand.path)} {...pick} />
            </span>
          ))}
        </p>
      )}
      {view !== null && view.collapsed !== null && (
        <details className="recon-terms">
          <summary>Show the {view.terms.length} amounts</summary>
          <p className="recon-operands">
            {view.terms.map(({ operand }) => (
              <span key={operand.path} className="recon-term">
                <Operand operand={operand} label={pathLabel(operand.path)} {...pick} />
              </span>
            ))}
          </p>
        </details>
      )}
      {failing && check.all_operands_verified && (
        <p className="recon-note recon-reviewed">
          Every amount in this check was reviewed, and the document still does not add up.
        </p>
      )}
    </li>
  );
}

function SkippedChecks({ checks }: { checks: ReconciliationCheck[] }) {
  if (checks.length === 0) return null;
  const items = checks.map((check) => (
    <li key={check.id}>
      {humanize(check.formula)}
      {check.detail && ` - ${humanize(check.detail)}`}
    </li>
  ));
  const noun = `${checks.length} check${checks.length === 1 ? "" : "s"} skipped`;
  // A long run of skipped line checks stays one line until it is asked for.
  if (checks.length > 3) {
    return (
      <details className="recon-muted recon-skipped">
        <summary>{noun}</summary>
        <ul>{items}</ul>
      </details>
    );
  }
  return (
    <div className="recon-muted recon-skipped">
      <p>{noun}:</p>
      <ul>{items}</ul>
    </div>
  );
}

function QuoteCheckRow({
  check,
  evidence,
  ...pick
}: PickProps & { check: ReconciliationCheck; evidence: Evidence | null | undefined }) {
  const path = quotePath(check);
  const operand = check.terms[0]?.operand;
  return (
    <li className="recon-check fail">
      <div className="recon-check-head">
        <StatusMark status="fail" />
        <span className="recon-formula">{humanize(check.formula)}</span>
      </div>
      <p className="recon-equation">
        {operand && <Operand operand={operand} label={pathLabel(path)} {...pick} />}
        <span className="recon-detail">{humanize(check.detail ?? check.formula)}</span>
      </p>
      {evidence && (
        <p className="recon-quote">
          <span className="recon-quote-page">{formatPage(evidence.page)}</span>{" "}
          {formatQuote(evidence.quote)}
        </p>
      )}
    </li>
  );
}

/** Whether each value agrees with the number printed in the quote it was read
 * from: the failures with the value, the printed number and the quote, the
 * passes counted. */
function EvidenceGroup({
  checks,
  evidenceOf,
  ...pick
}: PickProps & {
  checks: ReconciliationCheck[];
  evidenceOf: (path: string) => Evidence | null | undefined;
}) {
  if (checks.length === 0) return null;
  const failing = checks.filter((c) => c.status === "fail");
  const passing = checks.filter((c) => c.status === "pass").length;
  const skipped = checks.filter((c) => c.status === "skipped").length;
  const judged = passing + failing.length;
  return (
    <div className="recon-evidence" role="group" aria-labelledby="recon-evidence-heading">
      <h3 className="recon-heading" id="recon-evidence-heading">
        Evidence <StatusMark status={statusOf(checks)} />
      </h3>
      <p className="recon-summary">
        {judged === 0
          ? "No quoted value could be compared with its quote."
          : failing.length === 0
            ? passing === 1
              ? "The quoted value matches its quote."
              : `All ${passing} quoted values match their quotes.`
            : `${passing} of ${judged} quoted values match their quotes.`}
        {skipped > 0 &&
          ` ${skipped} not compared (no quote, or a zero said in words).`}
      </p>
      {failing.length > 0 && (
        <ul className="recon-checks">
          {failing.map((check) => (
            <QuoteCheckRow
              key={check.id}
              check={check}
              evidence={evidenceOf(quotePath(check))}
              {...pick}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

/** Whether the document's amounts add up, from the stored report. It sits
 * under the line items; selecting an operand highlights that amount above.
 * The suggestion is display-only here: it is applied from the review queue. */
export default function ReconciliationPanel({
  reconciliation,
  picked,
  onPick,
  evidenceOf,
}: PickProps & {
  reconciliation: Reconciliation | null;
  /** The page and quote a value at a report path was read from, for the
   * evidence group (undefined or null when it cites none). */
  evidenceOf: (path: string) => Evidence | null | undefined;
}) {
  if (reconciliation === null) {
    return (
      <section className="recon recon-muted" aria-label="Arithmetic check">
        <p>Arithmetic check: not run for this extraction.</p>
      </section>
    );
  }
  const pick = { picked, onPick };
  const { suggestion, derived, assumptions } = reconciliation;
  const quoteChecks = reconciliation.checks.filter(isQuote);
  const checks = reconciliation.checks.filter((c) => !isQuote(c));
  // Skipped line checks (no quantity or unit price printed) are listed with
  // the other skipped checks, not counted as lines that fail to add up.
  const lineChecks = checks.filter((c) => LINE_CHECK.test(c.id) && c.status !== "skipped");
  const linesPassing = lineChecks.filter((c) => c.status === "pass").length;
  const shown = checks.filter(
    (c) =>
      c.status === "fail" ||
      (c.status === "pass" && c.kind === "equation" && !LINE_CHECK.test(c.id))
  );
  const skipped = checks.filter((c) => c.status === "skipped");
  const quietPasses = [
    ...new Set(
      checks
        .filter((c) => c.status === "pass" && c.kind !== "equation")
        .map((c) => quietName(c.id))
    ),
  ];
  const passCount = checks.filter((c) => c.status === "pass" && c.kind !== "equation").length;

  return (
    <section className="recon" aria-labelledby="recon-heading">
      <h2 className="recon-heading" id="recon-heading">
        Arithmetic check <StatusMark status={statusOf(checks)} />
      </h2>

      {lineChecks.length > 0 && (
        <p className="recon-summary">
          {linesPassing} of {lineChecks.length}{" "}
          {lineChecks.length === 1 ? "line adds up" : "lines add up"}
        </p>
      )}

      {shown.length > 0 && (
        <ul className="recon-checks">
          {shown.map((check) => (
            <CheckRow key={check.id} check={check} {...pick} />
          ))}
        </ul>
      )}

      {passCount > 0 && (
        <p className="recon-summary">
          {passCount} other check{passCount === 1 ? " passes" : "s pass"} ({quietPasses.join(", ")})
        </p>
      )}

      <SkippedChecks checks={skipped} />

      <EvidenceGroup checks={quoteChecks} evidenceOf={evidenceOf} {...pick} />

      {derived.length > 0 && (
        <div className="recon-muted">
          <p>Not printed, so computed from the document:</p>
          <ul>
            {derived.map((d) => (
              <li key={d.path}>
                {pathLabel(d.path)} <span className="computed-tag">computed</span>{" "}
                {formatAmount(d.value)} - {humanize(d.formula)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {assumptions.length > 0 && (
        <div className="recon-muted">
          <p>Assumptions:</p>
          <ul>
            {assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </div>
      )}

      {suggestion && (
        <div className="recon-suggestion" role="note">
          <p>
            <strong>
              Suggested: {pathLabel(suggestion.path)} {formatAmount(suggestion.value)}
            </strong>{" "}
            - {suggestionWhy(suggestion)}
            {suggestion.current !== null && ` (now ${formatAmount(suggestion.current)})`}.
          </p>
          <p className="recon-muted">
            Nothing is changed here. Apply it from the{" "}
            <Link href="/review">review queue</Link>.
          </p>
        </div>
      )}

      <p className="recon-version">
        {reconciliation.rules_version}
        {reconciliation.currency && ` - ${reconciliation.currency}`}
      </p>
    </section>
  );
}
