export function formatDuration(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const total = Math.max(0, value);
  const minutes = Math.floor(total / 60);
  const seconds = total - minutes * 60;
  return `${minutes}:${seconds.toFixed(seconds < 10 ? 1 : 0).padStart(4, "0")}`;
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatScore(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function shouldStopSegment(
  currentTime: number,
  stopAt: number | null,
): boolean {
  return stopAt !== null && currentTime >= stopAt;
}
