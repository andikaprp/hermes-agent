import { useCallback, useEffect, useState } from "react";
import { Activity, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { usePageHeader } from "@/contexts/usePageHeader";
import { errorMessage } from "@/lib/api-error";

interface JevDecision {
  ts?: number;
  kind?: string;
  tier?: string;
  model?: string;
  confidence?: number | null;
  latency_ms?: number | null;
  reason?: string;
  content_hash?: string;
  mode?: string;
}

interface JevDecisionsResponse {
  mode: string;
  enabled: boolean;
  limit: number;
  decisions: JevDecision[];
  path: string;
}

function fmtTs(ts?: number): string {
  if (!ts) return "—";
  try {
    return new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19);
  } catch {
    return String(ts);
  }
}

export default function JevPage() {
  const [data, setData] = useState<JevDecisionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await api.getJevDecisions(200);
      setData(resp);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  usePageHeader({
    title: "Jev decisions",
    description: "Local observability for per-turn Jev routing (metadata only)",
  });

  return (
    <div className="flex h-full flex-col gap-4 p-4 md:p-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Activity className="h-4 w-4" />
          <span>
            mode: <code>{data?.mode ?? "…"}</code>
            {data?.path ? (
              <>
                {" "}
                · <code className="text-xs">{data.path}</code>
              </>
            ) : null}
          </span>
        </div>
        <Button variant="outline" size="sm" onClick={() => void load()} disabled={loading}>
          <RefreshCw className={`mr-2 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : null}

      {loading && !data ? (
        <div className="flex flex-1 items-center justify-center">
          <Spinner />
        </div>
      ) : data?.mode === "off" ? (
        <p className="text-sm text-muted-foreground">
          Observability is off. Set{" "}
          <code>gateway.jev_observability.mode</code> to <code>shadow</code> or{" "}
          <code>on</code> in config.yaml (default off). <code>logs/jev-decisions.jsonl</code> is
          the decision store. This page is a read-only local view — nothing is sent off-box.
        </p>
      ) : (
        <div className="overflow-auto border border-border">
          <table className="w-full min-w-[640px] text-left text-xs font-courier">
            <thead className="sticky top-0 bg-background-base border-b border-border">
              <tr>
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="px-3 py-2 font-medium">Kind</th>
                <th className="px-3 py-2 font-medium">Tier</th>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Confidence</th>
                <th className="px-3 py-2 font-medium">Latency</th>
                <th className="px-3 py-2 font-medium">Reason</th>
                <th className="px-3 py-2 font-medium">Hash</th>
              </tr>
            </thead>
            <tbody>
              {(data?.decisions || []).length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-3 py-6 text-muted-foreground">
                    No decisions recorded yet.
                  </td>
                </tr>
              ) : (
                (data?.decisions || []).map((row, i) => (
                  <tr key={`${row.ts}-${row.kind}-${i}`} className="border-b border-border/60">
                    <td className="px-3 py-1.5 whitespace-nowrap">{fmtTs(row.ts)}</td>
                    <td className="px-3 py-1.5">{row.kind || "—"}</td>
                    <td className="px-3 py-1.5">{row.tier || "—"}</td>
                    <td className="px-3 py-1.5">{row.model || "—"}</td>
                    <td className="px-3 py-1.5">
                      {row.confidence == null ? "—" : row.confidence.toFixed(3)}
                    </td>
                    <td className="px-3 py-1.5">
                      {row.latency_ms == null ? "—" : `${row.latency_ms} ms`}
                    </td>
                    <td className="px-3 py-1.5">{row.reason || "—"}</td>
                    <td className="px-3 py-1.5 text-muted-foreground">{row.content_hash || "—"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
