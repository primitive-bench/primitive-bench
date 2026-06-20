/**
 * Runtime validation for the bundled seed JSON.
 *
 * The TS types in `types/bench-report.d.ts` are *compile-time* only — they can't
 * catch a corrupted or hand-edited seed at runtime. This zod schema validates each
 * seed on load so a malformed file fails fast and loudly instead of producing
 * silently-wrong recommendations. It is intentionally lenient on extra keys
 * (`passthrough`) so additive schema changes don't break the server.
 */
import { z } from "zod";

const StatTest = z
  .object({
    method: z.string(),
    statistic: z.number().nullable().optional(),
    p_value: z.number().nullable().optional(),
    ci_low: z.number().nullable().optional(),
    ci_high: z.number().nullable().optional(),
    n: z.number().nullable().optional(),
    seed: z.number().nullable().optional(),
  })
  .passthrough();

const SliceResult = z
  .object({
    run_id: z.string(),
    primitive: z.string(),
    slice: z.string(),
    adapter: z.string(),
    n: z.number(),
    point_estimate: z.number(),
    metric_name: z.string(),
    ci: StatTest.nullable().optional(),
    separable: z.boolean().nullable().optional(),
    rank: z.number().nullable().optional(),
  })
  .passthrough();

const SliceReport = z
  .object({
    slice: z.string(),
    metric_name: z.string(),
    n: z.number(),
    status: z.string(),
    winner: z.string().nullable().optional(),
    band: z.array(z.string()).optional(),
    thin: z.boolean().optional(),
    saturated: z.boolean().optional(),
    citation: z.string().nullable().optional(),
    results: z.array(SliceResult).optional(),
  })
  .passthrough();

export const PrimitiveReportSchema = z
  .object({
    primitive: z.string(),
    run_id: z.string(),
    status: z.string(),
    as_of: z.string().nullable().optional(),
    slices: z.array(SliceReport).optional(),
  })
  .passthrough();
