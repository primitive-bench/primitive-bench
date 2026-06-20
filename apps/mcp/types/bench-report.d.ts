/* GENERATED from bench-report.schema.json by 'bench results schema' + json2ts. Do not edit. */

/**
 * The infrastructure primitives Primitive Bench certifies, one vertical each.
 */
export type Primitive =
  | "ocr"
  | "websearch"
  | "vectordb"
  | "reranker"
  | "retrieval"
  | "extraction"
  | "chunking"
  | "crawl"
  | "memory";
export type RunId = string;
export type Status = "published" | "no_published_results";
export type Slice = string;
export type MetricName = string;
/**
 * Slice size (max n across adapters).
 */
export type N = number;
export type Status1 = "published" | "thin_data" | "saturated";
export type Winner = string | null;
/**
 * TIE group, leader first.
 */
export type Band = string[];
export type Thin = boolean;
export type Saturated = boolean;
export type Citation = string | null;
export type RunId1 = string;
/**
 * Slice/constraint key, e.g. 'doc_type:invoice'
 */
export type Slice1 = string;
export type Adapter = string;
/**
 * Number of items in this slice for this adapter.
 */
export type N1 = number;
/**
 * Primary metric value (accuracy, nDCG, ...).
 */
export type PointEstimate = number;
export type MetricName1 = string;
export type Method = "mcnemar" | "wilson" | "bootstrap" | "bradley_terry";
export type Statistic = number | null;
export type PValue = number | null;
export type CiLow = number | null;
export type CiHigh = number | null;
export type N2 = number | null;
/**
 * Fixed seed for bootstrap reproducibility.
 */
export type Seed = number | null;
/**
 * Whether this adapter is statistically separable from the runner-up on this slice.
 */
export type Separable = boolean | null;
export type Rank = number | null;
/**
 * Per-adapter, ranked.
 */
export type Results = SliceResult[];
export type Slices = SliceReport[];

/**
 * All published slices for one primitive — the unit a seed file holds.
 */
export interface PrimitiveReport {
  primitive: Primitive;
  run_id: RunId;
  status?: Status;
  slices?: Slices;
}
/**
 * Slice-level view: the frozen per-adapter rows plus the derived call.
 *
 * `winner` is set only when the leader's Wilson interval clears the runner-up's
 * (D-10 separability gate). Otherwise `band` is the TIE group and `winner` is None.
 * `thin`/`saturated` are honesty flags the recommender uses to refuse a call.
 */
export interface SliceReport {
  slice: Slice;
  metric_name: MetricName;
  n: N;
  status?: Status1;
  winner?: Winner;
  band?: Band;
  thin?: Thin;
  saturated?: Saturated;
  citation?: Citation;
  results?: Results;
}
/**
 * Per-slice aggregate for one adapter. The unit the leaderboard ranks.
 *
 * `separable` is the trust gate: if False (overlapping CIs / high McNemar p at
 * this n), the leaderboard MUST NOT publish a single winner for the slice.
 */
export interface SliceResult {
  run_id: RunId1;
  primitive: Primitive;
  slice: Slice1;
  adapter: Adapter;
  n: N1;
  point_estimate: PointEstimate;
  metric_name?: MetricName1;
  /**
   * Wilson/bootstrap CI for the estimate.
   */
  ci?: StatTest | null;
  separable?: Separable;
  rank?: Rank;
}
/**
 * Output of a bench-stats test, carried on SliceResult for separability badges.
 */
export interface StatTest {
  method: Method;
  statistic?: Statistic;
  p_value?: PValue;
  ci_low?: CiLow;
  ci_high?: CiHigh;
  n?: N2;
  seed?: Seed;
}
