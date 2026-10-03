"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ApiError,
  getDailyMetrics,
  getEvalHistory,
  type DailyPoint,
  type EvalHistory,
  type EvalRun,
  type EvalSuite,
} from "@/lib/api";
import {
  ChartCard,
  Legend,
  LineChart,
  StackedColumns,
  StatTile,
  niceDomain,
  type Domain,
  type Series,
  decimalsFor,
} from "@/components/charts";

// ---- formatting ---------------------------------------------------------------

function usd(value: number): string {
  if (value === 0) return "$0";
  if (Math.abs(value) < 0.01) return `$${value.toFixed(4)}`;
  if (Math.abs(value) < 1) return `$${value.toFixed(3)}`;
  return `$${value.toFixed(2)}`;
}

function usdAxis(value: number, domain: Domain): string {
  return `$${value.toFixed(decimalsFor(domain.step))}`;
}

function ms(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`;
}

/** One unit for every tick: seconds once the scale reaches a second. */
function msAxis(value: number, domain: Domain): string {
  if (domain.max < 1000) return `${Math.round(value)}ms`;
  return `${(value / 1000).toFixed(decimalsFor(domain.step / 1000))}s`;
}

function pct(value: number): string {
  return `${(value * 100).toFixed(value > 0.995 || value < 0.005 ? 0 : 1)}%`;
}

function pctAxis(value: number, domain: Domain): string {
  return `${(value * 100).toFixed(decimalsFor(domain.step * 100))}%`;
}

// Rate charts span at least 10 points, so a small move reads as small.
const RATE_SPAN = 0.1;

const UTC_DAY = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const UTC_RUN = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: "UTC",
});

function dayLabel(date: string): string {
  return UTC_DAY.format(new Date(`${date}T00:00:00Z`));
}

function runLabel(run: EvalRun): string {
  return UTC_RUN.format(new Date(run.started_at));
}

function dash(value: number | null, format: (v: number) => string): string {
  return value === null ? "—" : format(value);
}

// ---- eval runs ----------------------------------------------------------------

const SUITES: { key: EvalSuite; label: string; item: string; items: string; empty: ReactNode }[] = [
  {
    key: "extraction",
    label: "Extraction",
    item: "document",
    items: "documents",
    empty: (
      <>
        No extraction eval runs committed yet: <code>python evals/run.py</code> writes one.
      </>
    ),
  },
  {
    key: "agent",
    label: "Ask agent",
    item: "question",
    items: "questions",
    empty: (
      <>
        No agent eval runs committed yet: <code>python evals/run_agent.py</code> with{" "}
        <code>ANTHROPIC_API_KEY</code> set writes one (an estimated $1-2 a run, hard-capped by{" "}
        <code>--max-cost</code>).
      </>
    ),
  },
  {
    key: "retrieval",
    label: "Retrieval",
    item: "query",
    items: "queries",
    empty: (
      <>
        No retrieval eval runs committed yet: <code>python evals/run_retrieval.py</code> writes one.
      </>
    ),
  },
];

/** One series per model (or embedder), slotted in order of first
 * appearance across every run of the suite -- so a model keeps its color. */
function runSeries(runs: EvalRun[], value: (run: EvalRun) => number | null): Series[] {
  const names = [...new Set(runs.map((r) => r.series))];
  return names.map((name, i) => ({
    key: name,
    name,
    slot: i + 1,
    values: runs.map((run) => (run.series === name ? value(run) : null)),
  }));
}

function previousOf(runs: EvalRun[], run: EvalRun): EvalRun | undefined {
  return runs.slice(0, runs.indexOf(run)).reverse().find((r) => r.series === run.series);
}

function EvalSection({ history }: { history: EvalHistory }) {
  const [suite, setSuite] = useState<EvalSuite>("extraction");
  const meta = SUITES.find((s) => s.key === suite)!;
  const runs = history[suite];
  const latest = runs.at(-1);
  const labels = runs.map(runLabel);
  const accuracy = runSeries(runs, (r) => r.accuracy);
  const cost = runSeries(runs, (r) => r.mean_cost_usd);
  const latency = runSeries(runs, (r) => r.latency_p50_ms);
  const metric = latest?.accuracy_metric ?? "accuracy";
  const hasCost = runs.some((r) => r.mean_cost_usd !== null);
  const previous = latest ? previousOf(runs, latest) : undefined;
  const accuracyDelta =
    latest?.accuracy != null && previous?.accuracy != null
      ? latest.accuracy - previous.accuracy
      : null;

  const runColumns = ["Run (UTC)", "Model", "Version", "n"];
  const runCells = (r: EvalRun) => [runLabel(r), r.series, r.detail, String(r.n)];

  return (
    <section className="dash-section" aria-labelledby="evals-heading">
      <h2 id="evals-heading" className="section-heading">
        Eval runs
      </h2>
      <p className="dash-intro">
        Offline, against known answers: every committed run of each eval suite, oldest to newest.
      </p>
      <div className="dash-filters">
        <div className="mode-toggle" role="group" aria-label="Eval suite">
          {SUITES.map((s) => (
            <button
              key={s.key}
              type="button"
              className={`mode-option${suite === s.key ? " active" : ""}`}
              aria-pressed={suite === s.key}
              onClick={() => setSuite(s.key)}
            >
              {s.label} <span className="mode-count">{history[s.key].length}</span>
            </button>
          ))}
        </div>
      </div>

      {!latest ? (
        <div className="results-panel dash-empty">{meta.empty}</div>
      ) : (
        <>
          <div className="stat-row">
            <StatTile
              label={`Latest ${metric}`}
              value={dash(latest.accuracy, pct)}
              delta={
                accuracyDelta === null
                  ? undefined
                  : {
                      text: `${accuracyDelta >= 0 ? "▲" : "▼"} ${(Math.abs(accuracyDelta) * 100).toFixed(1)} pts vs this model's previous run`,
                      good: accuracyDelta === 0 ? null : accuracyDelta > 0,
                    }
              }
              note={`${latest.series} · ${latest.n} ${meta.items}`}
            />
            <StatTile
              label={`Cost per ${meta.item}`}
              value={dash(latest.mean_cost_usd, usd)}
              note={latest.total_cost_usd !== null ? `${usd(latest.total_cost_usd)} for the run` : "no model cost"}
            />
            <StatTile
              label={`Latency per ${meta.item}`}
              value={dash(latest.latency_p50_ms, ms)}
              note={latest.latency_p95_ms !== null ? `p50 · p95 ${ms(latest.latency_p95_ms)}` : "p50"}
            />
            <StatTile label="Runs" value={String(runs.length)} note={`latest ${runLabel(latest)} UTC`} />
          </div>

          <div className="chart-grid three">
            <ChartCard
              title={`Accuracy: ${metric}`}
              subtitle="per run"
              legend={<Legend series={accuracy} kind="line" />}
              table={{
                columns: [...runColumns, metric],
                rows: runs.map((r) => [...runCells(r), dash(r.accuracy, pct)]),
              }}
            >
              <LineChart
                labels={labels}
                series={accuracy}
                format={pct}
                axisFormat={pctAxis}
                domain={niceDomain(
                  runs.flatMap((r) => (r.accuracy === null ? [] : [r.accuracy])),
                  { zero: false, cap: 1, minSpan: RATE_SPAN }
                )}
                connectGaps
                ariaLabel={`${metric} per ${meta.label.toLowerCase()} eval run`}
              />
            </ChartCard>

            <ChartCard
              title={`Cost per ${meta.item}`}
              subtitle="mean, per run"
              legend={<Legend series={cost} kind="line" />}
              table={{
                columns: [...runColumns, `$ / ${meta.item}`, "Run total"],
                rows: runs.map((r) => [
                  ...runCells(r),
                  dash(r.mean_cost_usd, usd),
                  dash(r.total_cost_usd, usd),
                ]),
              }}
            >
              {hasCost ? (
                <LineChart
                  labels={labels}
                  series={cost}
                  format={usd}
                  axisFormat={usdAxis}
                  domain={niceDomain(runs.flatMap((r) => (r.mean_cost_usd === null ? [] : [r.mean_cost_usd])))}
                  connectGaps
                  ariaLabel={`cost per ${meta.item} per eval run`}
                />
              ) : (
                <p className="dash-note">
                  Retrieval runs on a local embedding model: there is no model cost per query.
                </p>
              )}
            </ChartCard>

            <ChartCard
              title={`Latency per ${meta.item}`}
              subtitle="p50 per run; p95 in the table"
              legend={<Legend series={latency} kind="line" />}
              table={{
                columns: [...runColumns, "p50", "p95"],
                rows: runs.map((r) => [
                  ...runCells(r),
                  dash(r.latency_p50_ms, ms),
                  dash(r.latency_p95_ms, ms),
                ]),
              }}
            >
              <LineChart
                labels={labels}
                series={latency}
                format={ms}
                axisFormat={msAxis}
                domain={niceDomain(runs.flatMap((r) => (r.latency_p50_ms === null ? [] : [r.latency_p50_ms])))}
                connectGaps
                ariaLabel={`p50 latency per ${meta.item} per eval run`}
              />
            </ChartCard>
          </div>
        </>
      )}
    </section>
  );
}

// ---- production ---------------------------------------------------------------

const RANGES = [7, 30, 90];
// Quality rates are trailing sums over this many days: a day's handful of
// graded answers makes a raw daily rate swing between 0% and 100%.
const TRAILING_DAYS = 7;

function sum(values: number[]): number {
  return values.reduce((a, b) => a + b, 0);
}

function ProductionSection() {
  const [days, setDays] = useState(30);
  const [data, setData] = useState<DailyPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // The extra days feed the trailing rates of the window's first days.
    getDailyMetrics(days + TRAILING_DAYS - 1)
      .then((points) => {
        if (cancelled) return;
        setData(points);
        setError(null);
      })
      .catch((err) => {
        if (!cancelled)
          setError(err instanceof ApiError ? `API error: ${err.message}` : "Could not reach the API.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [days]);

  const view = useMemo(() => {
    if (!data) return null;
    const lead = TRAILING_DAYS - 1;
    const rated = (d: DailyPoint) => d.answers.feedback_up + d.answers.feedback_down;
    const groundedCount = (d: DailyPoint) => (d.answers.grounded_rate ?? 0) * d.answers.judged;
    // Trailing rate ending on each shown day: sum(hits) / sum(base) over the last 7 days.
    const trailing = (hits: (d: DailyPoint) => number, base: (d: DailyPoint) => number) =>
      data.slice(lead).map((_, i) => {
        const span = data.slice(i, i + TRAILING_DAYS);
        const denominator = sum(span.map(base));
        return denominator ? sum(span.map(hits)) / denominator : null;
      });
    const groundedTrail = trailing(groundedCount, (d) => d.answers.judged);
    const upTrail = trailing((d) => d.answers.feedback_up, rated);
    const shown = data.slice(lead);
    const labels = shown.map((d) => dayLabel(d.date));
    const judged = sum(shown.map((d) => d.answers.judged));
    const grounded = sum(shown.map(groundedCount));
    const up = sum(shown.map((d) => d.answers.feedback_up));
    const ratedTotal = sum(shown.map(rated));
    return {
      labels,
      runs: sum(shown.map((d) => d.answers.runs)),
      answered: sum(shown.map((d) => d.answers.answered)),
      judged,
      groundedRate: judged ? grounded / judged : null,
      ratedTotal,
      upRate: ratedTotal ? up / ratedTotal : null,
      spend: sum(shown.map((d) => d.spend.total_usd)),
      documents: sum(shown.map((d) => d.extractions.documents)),
      spendSeries: [
        { key: "documents", name: "Documents (extract + transcribe)", slot: 1, values: shown.map((d) => d.spend.documents_usd) },
        { key: "answers", name: "Answers (Ask + chat)", slot: 2, values: shown.map((d) => d.spend.answers_usd) },
        { key: "grading", name: "Grading (groundedness)", slot: 3, values: shown.map((d) => d.spend.grading_usd) },
      ] satisfies Series[],
      qualitySeries: [
        { key: "grounded", name: "Grounded (automatic check)", slot: 1, values: groundedTrail },
        { key: "marked-right", name: "Marked right (people)", slot: 2, values: upTrail },
      ] satisfies Series[],
      costSeries: [
        { key: "cost", name: "Cost per answer", slot: 1, values: shown.map((d) => d.answers.mean_cost_usd) },
      ] satisfies Series[],
      latencySeries: [
        { key: "p50", name: "p50", slot: 1, values: shown.map((d) => d.answers.latency_p50_ms) },
        { key: "p95", name: "p95", slot: 2, values: shown.map((d) => d.answers.latency_p95_ms) },
      ] satisfies Series[],
    };
  }, [data]);

  const muted = loading && data !== null;
  const tableOf = (series: Series[], format: (v: number) => string) => ({
    columns: ["Day (UTC)", ...series.map((s) => s.name)],
    rows: (view?.labels ?? []).map((label, i) => [
      label,
      ...series.map((s) => dash(s.values[i], format)),
    ]),
  });

  return (
    <section className="dash-section" aria-labelledby="production-heading">
      <h2 id="production-heading" className="section-heading">
        Production
      </h2>
      <p className="dash-intro">
        Live traffic per UTC day: what the model calls cost, how good the answers were (the sampled
        automatic groundedness check and people&apos;s <em>was this right?</em>), and how long they took.
      </p>
      <div className="dash-filters">
        <div className="mode-toggle" role="group" aria-label="Date range">
          {RANGES.map((range) => (
            <button
              key={range}
              type="button"
              className={`mode-option${days === range ? " active" : ""}`}
              aria-pressed={days === range}
              onClick={() => {
                if (range === days) return;
                setLoading(true);
                setDays(range);
              }}
            >
              Last {range} days
            </button>
          ))}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {!view && !error && <div className="empty-state">Loading…</div>}

      {view && (
        <>
          <div className={`stat-row${muted ? " muted" : ""}`}>
            <StatTile label="Answers" value={view.runs.toLocaleString()} note={`${view.answered} answered · Ask + chat`} />
            <StatTile
              label="Grounded"
              value={dash(view.groundedRate, pct)}
              note={view.judged ? `of ${view.judged} graded` : "none graded: set ASK_JUDGE_SAMPLE_RATE"}
            />
            <StatTile
              label="Marked right"
              value={dash(view.upRate, pct)}
              note={view.ratedTotal ? `of ${view.ratedTotal} rated` : "no ratings yet"}
            />
            <StatTile label="Model spend" value={usd(view.spend)} note={`${view.documents} documents extracted`} />
          </div>

          <div className="chart-grid three">
            <ChartCard
              title="Spend by stage"
              subtitle="USD per day"
              legend={<Legend series={view.spendSeries} kind="rect" />}
              table={tableOf(view.spendSeries, usd)}
              muted={muted}
              wide
            >
              <StackedColumns
                labels={view.labels}
                series={view.spendSeries}
                format={usd}
                axisFormat={usdAxis}
                ariaLabel="Model spend per day by stage"
              />
            </ChartCard>

            <ChartCard
              title="Answer quality"
              subtitle={`trailing ${TRAILING_DAYS}-day rate`}
              legend={<Legend series={view.qualitySeries} kind="line" />}
              table={tableOf(view.qualitySeries, pct)}
              muted={muted}
            >
              <LineChart
                labels={view.labels}
                series={view.qualitySeries}
                format={pct}
                axisFormat={pctAxis}
                domain={niceDomain([1], { zero: true, cap: 1 })}
                ariaLabel="Grounded and marked-right rates per day"
              />
            </ChartCard>

            <ChartCard
              title="Cost per answer"
              subtitle="mean per day"
              table={tableOf(view.costSeries, usd)}
              muted={muted}
            >
              <LineChart
                labels={view.labels}
                series={view.costSeries}
                format={usd}
                axisFormat={usdAxis}
                domain={niceDomain(view.costSeries[0].values.flatMap((v) => (v === null ? [] : [v])))}
                ariaLabel="Mean cost per answer per day"
              />
            </ChartCard>

            <ChartCard
              title="Answer latency"
              subtitle="per day"
              legend={<Legend series={view.latencySeries} kind="line" />}
              table={tableOf(view.latencySeries, ms)}
              muted={muted}
            >
              <LineChart
                labels={view.labels}
                series={view.latencySeries}
                format={ms}
                axisFormat={msAxis}
                domain={niceDomain(
                  view.latencySeries.flatMap((s) => s.values.flatMap((v) => (v === null ? [] : [v])))
                )}
                ariaLabel="p50 and p95 answer latency per day"
              />
            </ChartCard>
          </div>
        </>
      )}
    </section>
  );
}

export default function DashboardPage() {
  const [history, setHistory] = useState<EvalHistory | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvalHistory()
      .then(setHistory)
      .catch((err) =>
        setError(err instanceof ApiError ? `API error: ${err.message}` : "Could not reach the API.")
      );
  }, []);

  return (
    <div>
      <h1 className="page-title">Monitoring</h1>
      <p className="review-intro">
        How good, how expensive, and how slow the AI parts of doc-pilot are -- measured offline by
        the eval suites, and in production from every stored answer and extraction.
      </p>
      {error && <div className="error-banner">{error}</div>}
      {history ? (
        <EvalSection history={history} />
      ) : (
        !error && <div className="empty-state">Loading…</div>
      )}
      <ProductionSection />
    </div>
  );
}
