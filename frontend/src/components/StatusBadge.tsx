import type { JobStatus } from "../types/api";

const labels: Record<JobStatus, string> = {
  queued: "排队中",
  running: "分析中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" />
      {labels[status]}
    </span>
  );
}
