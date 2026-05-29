import { prom } from "@/lib/prometheus";

/**
 * Shared GPU metrics helpers for Prometheus value extraction and recording rule checks.
 * Used by /api/gpu, /api/gpu/node, and /api/gpu/report routes.
 */

const normalizePromLabelFilter = (filter: string) =>
  filter.trim().replace(/^\{/, "").replace(/\}$/, "").trim();

const escapePromString = (value: string) =>
  value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

export type PrometheusSampleValue =
  | [number | string, string]
  | { value: string | number }
  | string
  | number
  | null
  | undefined;

type PrometheusLabels = Record<string, string | undefined>;
type PrometheusMetric = PrometheusLabels & { labels?: PrometheusLabels };

interface PrometheusSeriesItem {
  metric?: PrometheusMetric;
  value?: PrometheusSampleValue;
}

interface PrometheusInstantResult {
  result?: PrometheusSeriesItem[];
}

const getResultItems = (result: unknown): PrometheusSeriesItem[] => {
  const maybeResult = result as PrometheusInstantResult | null | undefined;
  return Array.isArray(maybeResult?.result) ? maybeResult.result : [];
};

export const getGpuMetricsCluster = () =>
  process.env.GPU_METRICS_CLUSTER?.trim() || "";

export const getGpuMetricsLabelFilter = () => {
  const configuredFilter = normalizePromLabelFilter(
    process.env.GPU_METRICS_LABEL_FILTER || ""
  );

  if (configuredFilter) return configuredFilter;

  const cluster = getGpuMetricsCluster();
  return cluster ? `cluster="${escapePromString(cluster)}"` : "";
};

export const buildGpuMetricsFilter = (filters: string[] = []) => {
  const labelFilter = getGpuMetricsLabelFilter();
  return [...filters, labelFilter].filter(Boolean).join(", ");
};

export const promSelector = (metricName: string, filter?: string) =>
  filter ? `${metricName}{${filter}}` : metricName;

/**
 * Extract a numeric value from a Prometheus result item's value field.
 * Handles all formats returned by the prometheus-query library:
 * - Array: [timestamp, "value"] (raw Prometheus API format)
 * - Object: { time: Date, value: number } (prometheus-query SampleValue)
 * - Primitive: number or string
 */
export const extractNumericValue = (value: PrometheusSampleValue): number => {
  if (Array.isArray(value) && value.length > 1) {
    return parseFloat(value[1]);
  }
  if (typeof value === "object" && value !== null && !Array.isArray(value) && value.value !== undefined) {
    return parseFloat(String(value.value));
  }
  return parseFloat(String(value));
};

/**
 * Extract a single scalar value from a Prometheus instant query result.
 * Returns null if no result or parsing fails.
 */
export const extractValue = (result: unknown): number | null => {
  try {
    const item = getResultItems(result)[0];
    if (!item) return null;
    if (item.value) {
      const parsed = extractNumericValue(item.value);
      return isNaN(parsed) ? null : parsed;
    }
    return null;
  } catch {
    return null;
  }
};

/**
 * Extract labeled series from a Prometheus instant query result.
 * Returns an array of objects with jobId, labels, value, hostname, gpuModel, and timestamp.
 */
export const extractSeries = (result: unknown): PrometheusSeries[] => {
  try {
    return getResultItems(result).map((item) => {
      const metric = item.metric || {};
      const labels = metric.labels || {};
      const jobId = labels.hpc_job || metric.hpc_job || "unknown";

      let value = 0;
      let timestamp = Date.now() / 1000;

      if (item.value) {
        if (Array.isArray(item.value) && item.value.length > 1) {
          value = parseFloat(item.value[1]);
          if (typeof item.value[0] === "number") {
            timestamp = item.value[0];
          }
        } else {
          value = extractNumericValue(item.value);
        }
      }

      return {
        jobId,
        labels,
        value: isNaN(value) ? 0 : value,
        hostname: labels.Hostname || metric.Hostname || "unknown",
        gpuModel: labels.modelName || metric.modelName || "unknown",
        cluster: labels.cluster || metric.cluster || "unknown",
        timestamp,
      };
    });
  } catch {
    return [];
  }
};

export interface PrometheusSeries {
  jobId: string;
  labels: Record<string, string | undefined>;
  value: number;
  hostname: string;
  gpuModel: string;
  cluster: string;
  timestamp: number;
}

/**
 * Check if Prometheus recording rules for GPU metrics are available.
 * Optionally test with a specific job ID.
 */
export const checkRecordingRulesAvailable = async (jobId?: string, filter?: string): Promise<boolean> => {
  if (!prom) return false;
  try {
    const queryFilter = filter || buildGpuMetricsFilter(jobId ? [`hpc_job="${jobId}"`] : []);
    const query = promSelector("job:gpu_utilization:current_avg", queryFilter);
    const result = await prom.instantQuery(query);
    return result && Array.isArray(result.result) && result.result.length > 0;
  } catch {
    return false;
  }
};

/**
 * Count the number of results in a Prometheus query response.
 */
export const extractCount = (result: unknown): number => {
  try {
    return getResultItems(result).length;
  } catch {
    return 0;
  }
};
