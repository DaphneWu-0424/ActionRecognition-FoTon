export type AssetKind = "image" | "video" | "clip" | "frame";
export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface Page<T> {
  items: T[];
  page: number;
  page_size: number;
  total: number;
}

export interface Asset {
  id: string;
  kind: AssetKind;
  name: string;
  original_filename: string;
  mime_type: string;
  file_size: number;
  duration_sec: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  created_at: string;
  content_url: string;
  thumbnail_url: string | null;
}

export interface AssetSummary {
  id: string;
  name: string;
  kind: AssetKind;
  duration_sec: number | null;
  content_url: string;
  thumbnail_url: string | null;
}

export interface LocalizationParameters {
  sample_fps: number;
  window_sec: number;
  stride_sec: number;
  top_frame_ratio: number;
  nms_iou: number;
}

export interface LocalizationResult {
  id: string;
  start_sec: number;
  end_sec: number;
  score: number;
  mean_score: number;
  max_score: number;
  best_frame_sec: number;
  clip_url: string;
  best_frame_url: string;
}

export interface Job {
  id: string;
  retry_of_job_id: string | null;
  model_key: string;
  status: JobStatus;
  stage: string | null;
  progress: number;
  parameters: LocalizationParameters;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  query_image: AssetSummary;
  target_video: AssetSummary;
  result: LocalizationResult | null;
}

export interface JobListItem {
  id: string;
  status: JobStatus;
  stage: string | null;
  progress: number;
  created_at: string;
  finished_at: string | null;
  query_image: AssetSummary;
  target_video: AssetSummary;
}
