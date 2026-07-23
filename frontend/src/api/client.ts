import type {
  Asset,
  Job,
  JobListItem,
  LocalizationParameters,
  Page,
} from "../types/api";

interface ApiErrorEnvelope {
  error?: {
    code?: string;
    message?: string;
  };
}

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let payload: ApiErrorEnvelope = {};
    try {
      payload = (await response.json()) as ApiErrorEnvelope;
    } catch {
      // Keep the fallback message for non-JSON errors.
    }
    throw new ApiError(
      payload.error?.message ?? `请求失败（HTTP ${response.status}）`,
      payload.error?.code ?? "HTTP_ERROR",
      response.status,
    );
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function listAssets(
  kind: "image" | "video",
  search = "",
  page = 1,
): Promise<Page<Asset>> {
  const params = new URLSearchParams({
    kind,
    page: String(page),
    page_size: "24",
  });
  if (search.trim()) params.set("search", search.trim());
  return request(`/api/assets?${params}`);
}

export function uploadAsset(file: File): Promise<Asset> {
  const data = new FormData();
  data.append("file", file);
  return request("/api/assets/upload", {
    method: "POST",
    body: data,
  });
}

export function scanAssets(scanRootId: string) {
  return request<{
    discovered: number;
    created: number;
    skipped: number;
    failed: number;
  }>("/api/assets/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scan_root_id: scanRootId,
      relative_dir: "",
      recursive: true,
    }),
  });
}

export function createJob(body: {
  query_image_id: string;
  target_video_id: string;
  parameters: LocalizationParameters;
}): Promise<Job> {
  return request("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function listJobs(
  status = "",
  page = 1,
): Promise<Page<JobListItem>> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: "30",
  });
  if (status) params.set("status", status);
  return request(`/api/jobs?${params}`);
}

export function getJob(id: string): Promise<Job> {
  return request(`/api/jobs/${id}`);
}

export function cancelJob(id: string): Promise<Job> {
  return request(`/api/jobs/${id}/cancel`, { method: "POST" });
}

export function retryJob(id: string): Promise<Job> {
  return request(`/api/jobs/${id}/retry`, { method: "POST" });
}
