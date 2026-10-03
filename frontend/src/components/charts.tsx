"use client";

// Small SVG charts for the monitoring dashboard, no chart library: a
// multi-series line chart and a stacked column chart, each in a card with
// a legend and a table view (every value is readable without hovering).
// Marks follow one spec: 2px lines, 8px markers with a 2px surface ring,
// columns at most 24px wide with a 4px rounded top and a 2px surface gap
// between stacked segments, hairline grid. Series colors are the
// --series-N tokens in globals.css (validated for color-vision deficiency
// in both themes); a series keeps its slot whatever else is shown.

import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from "react";

export interface Series {
  key: string;
  name: string;
  /** 1-4 picks --series-N; anything else draws in the neutral "other" gray. */
  slot: number;
  values: (number | null)[];
}

export interface TableData {
  columns: string[];
  rows: string[][];
}

const MARGIN = { top: 14, right: 56, bottom: 26, left: 56 };

function seriesColor(slot: number): string {
  return slot >= 1 && slot <= 4 ? `var(--series-${slot})` : "var(--series-other)";
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function niceStep(range: number, target: number): number {
  const raw = range / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
}

export interface Domain {
  min: number;
  max: number;
  ticks: number[];
  step: number;
}

/** Decimal places that show every tick of `step` exactly (0.025 -> 3). */
export function decimalsFor(step: number): number {
  for (let d = 0; d < 8; d++) {
    if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d;
  }
  return 8;
}

/** A y-domain on clean tick values. Lines may start above zero (zero:
 * false) so small changes in a rate stay visible; columns never do. */
export function niceDomain(
  values: number[],
  { zero = true, cap, minSpan = 0 }: { zero?: boolean; cap?: number; minSpan?: number } = {}
): Domain {
  const finite = values.filter((v) => Number.isFinite(v));
  let low = zero ? 0 : Math.min(...finite, cap ?? Infinity);
  let high = Math.max(...finite, 0);
  if (!finite.length) {
    low = 0;
    high = cap ?? 1;
  }
  // A span floor keeps a small change from filling the plot: a 0.7-point
  // move in a rate shouldn't look like a cliff.
  if (high - low < minSpan) low = Math.max(0, high - minSpan);
  if (high - low < 1e-9) {
    const pad = Math.abs(high) * 0.1 || (cap ?? 1) * 0.1;
    high += pad;
    low = zero ? 0 : low - pad;
  }
  const step = niceStep(high - low, 4);
  let min = Math.floor(low / step) * step;
  let max = Math.ceil(high / step) * step;
  if (cap !== undefined) max = Math.min(max, cap);
  if (zero || min < 0) min = Math.max(0, min);
  const ticks: number[] = [];
  for (let t = min; t <= max + step / 1e6; t += step) ticks.push(Number(t.toFixed(10)));
  return { min, max, ticks, step };
}

function xTickIndexes(count: number, width: number): number[] {
  const fit = Math.max(1, Math.floor(width / 90));
  if (count <= fit) return Array.from({ length: count }, (_, i) => i);
  const every = Math.ceil(count / fit);
  const indexes = [];
  // Anchor on the newest point, which is the one people look for.
  for (let i = count - 1; i >= 0; i -= every) indexes.unshift(i);
  return indexes;
}

/** The legend: always shown for two or more series; a line key for line
 * charts, a swatch for columns. */
export function Legend({
  series,
  kind,
}: {
  series: Pick<Series, "key" | "name" | "slot">[];
  kind: "line" | "rect";
}) {
  if (series.length < 2) return null;
  return (
    <ul className="chart-legend">
      {series.map((s) => (
        <li key={s.key}>
          <svg width="16" height="10" aria-hidden="true">
            {kind === "line" ? (
              <line x1="1" y1="5" x2="15" y2="5" stroke={seriesColor(s.slot)} strokeWidth="2" strokeLinecap="round" />
            ) : (
              <rect x="3" y="0" width="10" height="10" rx="2" fill={seriesColor(s.slot)} />
            )}
          </svg>
          {s.name}
        </li>
      ))}
    </ul>
  );
}

/** A chart in a card: title, subtitle, legend, and a table view toggle.
 * `muted` holds the previous render at reduced opacity while it reloads. */
export function ChartCard({
  title,
  subtitle,
  legend,
  table,
  muted = false,
  wide = false,
  children,
}: {
  title: string;
  subtitle?: string;
  legend?: ReactNode;
  table: TableData;
  muted?: boolean;
  wide?: boolean;
  children: ReactNode;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className={`chart-card${wide ? " wide" : ""}${muted ? " muted" : ""}`}>
      <figcaption className="chart-head">
        <span>
          <span className="chart-title">{title}</span>
          {subtitle && <span className="chart-subtitle">{subtitle}</span>}
        </span>
        <button
          type="button"
          className="chart-toggle"
          aria-pressed={asTable}
          onClick={() => setAsTable((t) => !t)}
        >
          {asTable ? "Chart" : "Table"}
        </button>
      </figcaption>
      {!asTable && legend}
      {asTable ? (
        <div className="chart-table-wrap">
          <table className="chart-table">
            <thead>
              <tr>
                {table.columns.map((c) => (
                  <th key={c} scope="col">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        children
      )}
    </figure>
  );
}

function Tooltip({
  x,
  width,
  title,
  rows,
}: {
  x: number;
  width: number;
  title: string;
  rows: { key: string; name: string; slot: number; value: string; kind: "line" | "rect" }[];
}) {
  const flip = x > width / 2;
  return (
    <div
      className="chart-tooltip"
      style={flip ? { right: width - x + 10 } : { left: x + 10 }}
      role="status"
    >
      <div className="chart-tooltip-title">{title}</div>
      {rows.map((row) => (
        <div key={row.key} className="chart-tooltip-row">
          <svg width="12" height="10" aria-hidden="true">
            {row.kind === "line" ? (
              <line x1="1" y1="5" x2="11" y2="5" stroke={seriesColor(row.slot)} strokeWidth="2" strokeLinecap="round" />
            ) : (
              <rect x="1" y="0" width="10" height="10" rx="2" fill={seriesColor(row.slot)} />
            )}
          </svg>
          <strong>{row.value}</strong>
          <span>{row.name}</span>
        </div>
      ))}
    </div>
  );
}

/** Pointer and arrow-key tracking of the data position nearest the
 * pointer: the crosshair finds the x, so nobody has to land on a 2px line. */
function useActiveIndex(count: number, band: number) {
  const [active, setActive] = useState<number | null>(null);
  const onPointerMove = (event: PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const index = Math.floor((event.clientX - box.left) / band);
    setActive(Math.max(0, Math.min(count - 1, index)));
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    setActive((current) => {
      const start = current ?? count - 1;
      const next = start + (event.key === "ArrowRight" ? 1 : -1);
      return Math.max(0, Math.min(count - 1, next));
    });
  };
  return {
    active,
    onPointerMove,
    onPointerLeave: () => setActive(null),
    onKeyDown,
    onBlur: () => setActive(null),
  };
}

/** Multi-series lines over ordered positions (eval runs, days). A null
 * is a gap; with connectGaps a series' line bridges positions that
 * belong to other series (eval runs of other models). */
export function LineChart({
  labels,
  series,
  format,
  axisFormat = format,
  domain,
  connectGaps = false,
  height = 200,
  ariaLabel,
}: {
  labels: string[];
  series: Series[];
  format: (value: number) => string;
  axisFormat?: (value: number, domain: Domain) => string;
  domain: Domain;
  /** Bridge nulls with the line, and leave series with no value at a
   * position out of its tooltip: for positions that belong to one series
   * each (an eval run is one model's). */
  connectGaps?: boolean;
  height?: number;
  ariaLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const count = labels.length;
  const plotWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const band = count ? plotWidth / count : plotWidth;
  const hover = useActiveIndex(count, band);
  const x = (i: number) => MARGIN.left + (i + 0.5) * band;
  const y = (v: number) =>
    MARGIN.top + plotHeight - ((v - domain.min) / (domain.max - domain.min || 1)) * plotHeight;

  const lines = series.map((s) => {
    const segments: string[] = [];
    let current = "";
    s.values.forEach((value, i) => {
      if (value === null) {
        if (!connectGaps && current) {
          segments.push(current);
          current = "";
        }
        return;
      }
      current += `${current ? "L" : "M"}${x(i).toFixed(1)},${y(value).toFixed(1)}`;
    });
    if (current) segments.push(current);
    const points = s.values.flatMap((value, i) => (value === null ? [] : [i]));
    return { s, segments, points };
  });

  // Markers on every point when there are few, else only where a point
  // stands alone (it would otherwise be invisible) and at each end.
  const denseMarkers = count <= 16;
  // Direct labels: each series' latest value, skipped where it would
  // collide with another (the legend and tooltip still carry it).
  const placed: { x: number; y: number }[] = [];
  const endLabels = series.length <= 4
    ? lines.flatMap(({ s, points }) => {
        const last = points.at(-1);
        if (last === undefined) return [];
        const lx = x(last) + 8;
        const ly = y(s.values[last] as number) + 4;
        if (placed.some((p) => Math.abs(p.x - lx) < 60 && Math.abs(p.y - ly) < 13)) return [];
        placed.push({ x: lx, y: ly });
        return [{ key: s.key, x: lx, y: ly, text: format(s.values[last] as number) }];
      })
    : [];

  return (
    <div
      ref={ref}
      className="chart-plot"
      tabIndex={0}
      role="img"
      aria-label={ariaLabel}
      onKeyDown={hover.onKeyDown}
      onBlur={hover.onBlur}
    >
      {width > 0 && (
        <svg width={width} height={height} aria-hidden="true">
          {domain.ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={MARGIN.left}
                x2={MARGIN.left + plotWidth}
                y1={y(tick)}
                y2={y(tick)}
                className={tick === domain.min ? "chart-baseline" : "chart-gridline"}
              />
              <text x={MARGIN.left - 8} y={y(tick) + 4} textAnchor="end" className="chart-axis">
                {axisFormat(tick, domain)}
              </text>
            </g>
          ))}
          {xTickIndexes(count, plotWidth).map((i) => (
            <text key={i} x={x(i)} y={height - 6} textAnchor="middle" className="chart-axis">
              {labels[i]}
            </text>
          ))}
          {hover.active !== null && (
            <line
              x1={x(hover.active)}
              x2={x(hover.active)}
              y1={MARGIN.top}
              y2={MARGIN.top + plotHeight}
              className="chart-crosshair"
            />
          )}
          {lines.map(({ s, segments }) =>
            segments.map((d, i) => (
              <path
                key={`${s.key}-${i}`}
                d={d}
                fill="none"
                stroke={seriesColor(s.slot)}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))
          )}
          {lines.map(({ s, points }) =>
            points
              .filter(
                (i, n) =>
                  denseMarkers ||
                  hover.active === i ||
                  n === points.length - 1 ||
                  ((s.values[i - 1] ?? null) === null && (s.values[i + 1] ?? null) === null)
              )
              .map((i) => (
                <circle
                  key={`${s.key}-${i}`}
                  cx={x(i)}
                  cy={y(s.values[i] as number)}
                  r="4"
                  fill={seriesColor(s.slot)}
                  className="chart-marker"
                />
              ))
          )}
          {endLabels.map((label) => (
            <text key={label.key} x={label.x} y={label.y} className="chart-value">
              {label.text}
            </text>
          ))}
          <rect
            x={MARGIN.left}
            y={MARGIN.top}
            width={plotWidth}
            height={plotHeight}
            fill="transparent"
            onPointerMove={hover.onPointerMove}
            onPointerLeave={hover.onPointerLeave}
          />
        </svg>
      )}
      {hover.active !== null && (
        <Tooltip
          x={x(hover.active)}
          width={width}
          title={labels[hover.active]}
          rows={series
            .filter((s) => !connectGaps || s.values[hover.active as number] !== null)
            .map((s) => ({
              key: s.key,
              name: s.name,
              slot: s.slot,
              kind: "line",
              value:
                s.values[hover.active as number] === null
                  ? "—"
                  : format(s.values[hover.active as number] as number),
            }))}
        />
      )}
    </div>
  );
}

/** Stacked columns, one per position: parts of a whole per day. */
export function StackedColumns({
  labels,
  series,
  format,
  axisFormat = format,
  height = 200,
  ariaLabel,
}: {
  labels: string[];
  series: Series[];
  format: (value: number) => string;
  axisFormat?: (value: number, domain: Domain) => string;
  height?: number;
  ariaLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const count = labels.length;
  const plotWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const band = count ? plotWidth / count : plotWidth;
  const barWidth = Math.max(2, Math.min(24, band * 0.7));
  const hover = useActiveIndex(count, band);
  const totals = labels.map((_, i) => series.reduce((sum, s) => sum + (s.values[i] ?? 0), 0));
  const domain = niceDomain(totals, { zero: true });
  const y = (v: number) => MARGIN.top + plotHeight - (v / (domain.max || 1)) * plotHeight;
  const GAP = 2;
  const RADIUS = 4;
  const peak = totals.reduce((best, t, i) => (t > (totals[best] ?? -1) ? i : best), 0);

  return (
    <div
      ref={ref}
      className="chart-plot"
      tabIndex={0}
      role="img"
      aria-label={ariaLabel}
      onKeyDown={hover.onKeyDown}
      onBlur={hover.onBlur}
    >
      {width > 0 && (
        <svg width={width} height={height} aria-hidden="true">
          {domain.ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={MARGIN.left}
                x2={MARGIN.left + plotWidth}
                y1={y(tick)}
                y2={y(tick)}
                className={tick === 0 ? "chart-baseline" : "chart-gridline"}
              />
              <text x={MARGIN.left - 8} y={y(tick) + 4} textAnchor="end" className="chart-axis">
                {axisFormat(tick, domain)}
              </text>
            </g>
          ))}
          {xTickIndexes(count, plotWidth).map((i) => (
            <text
              key={i}
              x={MARGIN.left + (i + 0.5) * band}
              y={height - 6}
              textAnchor="middle"
              className="chart-axis"
            >
              {labels[i]}
            </text>
          ))}
          {labels.map((_, i) => {
            const left = MARGIN.left + (i + 0.5) * band - barWidth / 2;
            const visible = series.filter((s) => (s.values[i] ?? 0) > 0);
            let base = 0;
            return (
              <g key={i} className={hover.active === i ? "chart-column active" : "chart-column"}>
                {visible.map((s, n) => {
                  const value = s.values[i] as number;
                  const top = y(base + value);
                  // A 2px surface gap under every segment but the first.
                  const bottom = y(base) - (n > 0 ? GAP : 0);
                  base += value;
                  const h = bottom - top;
                  if (h <= 0) return null;
                  const isTop = n === visible.length - 1;
                  const r = isTop ? Math.min(RADIUS, h, barWidth / 2) : 0;
                  const d = `M${left},${bottom} V${top + r} Q${left},${top} ${left + r},${top} H${left + barWidth - r} Q${left + barWidth},${top} ${left + barWidth},${top + r} V${bottom} Z`;
                  return <path key={s.key} d={d} fill={seriesColor(s.slot)} />;
                })}
              </g>
            );
          })}
          {totals[peak] > 0 && (
            <text
              x={MARGIN.left + (peak + 0.5) * band}
              y={y(totals[peak]) - 6}
              textAnchor="middle"
              className="chart-value"
            >
              {format(totals[peak])}
            </text>
          )}
          <rect
            x={MARGIN.left}
            y={MARGIN.top}
            width={plotWidth}
            height={plotHeight}
            fill="transparent"
            onPointerMove={hover.onPointerMove}
            onPointerLeave={hover.onPointerLeave}
          />
        </svg>
      )}
      {hover.active !== null && (
        <Tooltip
          x={MARGIN.left + (hover.active + 0.5) * band}
          width={width}
          title={`${labels[hover.active]} · ${format(totals[hover.active])}`}
          rows={series.map((s) => ({
            key: s.key,
            name: s.name,
            slot: s.slot,
            kind: "rect",
            value: format(s.values[hover.active as number] ?? 0),
          }))}
        />
      )}
    </div>
  );
}

/** A headline number: label, value, and an optional delta or note. */
export function StatTile({
  label,
  value,
  note,
  delta,
}: {
  label: string;
  value: string;
  note?: string;
  delta?: { text: string; good: boolean | null };
}) {
  return (
    <div className="stat-tile">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {delta && (
        <div
          className={`stat-delta${delta.good === true ? " good" : delta.good === false ? " bad" : ""}`}
        >
          {delta.text}
        </div>
      )}
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}
