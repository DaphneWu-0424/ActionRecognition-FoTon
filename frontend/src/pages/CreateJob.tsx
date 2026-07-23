import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createJob, listAssets } from "../api/client";
import { AssetCard } from "../components/AssetCard";
import type { LocalizationParameters } from "../types/api";

const defaults: LocalizationParameters = {
  sample_fps: 2,
  window_sec: 2,
  stride_sec: 0.5,
  top_frame_ratio: 0.5,
  nms_iou: 0.3,
};

export function CreateJob() {
  const navigate = useNavigate();
  const [queryId, setQueryId] = useState("");
  const [videoId, setVideoId] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [params, setParams] = useState(defaults);
  const images = useQuery({
    queryKey: ["assets", "image", ""],
    queryFn: () => listAssets("image"),
  });
  const videos = useQuery({
    queryKey: ["assets", "video", ""],
    queryFn: () => listAssets("video"),
  });
  const create = useMutation({
    mutationFn: createJob,
    onSuccess: (job) => navigate(`/jobs/${job.id}`),
  });
  const canSubmit = queryId && videoId && !create.isPending;

  const numberField = (
    key: keyof LocalizationParameters,
    label: string,
    step: number,
  ) => (
    <label>
      <span>{label}</span>
      <input
        type="number"
        step={step}
        min={step}
        value={params[key]}
        onChange={(event) =>
          setParams((current) => ({
            ...current,
            [key]: Number(event.target.value),
          }))
        }
      />
    </label>
  );

  return (
    <section className="page">
      <div className="page-heading">
        <span className="eyebrow">02 / NEW SEARCH</span>
        <h1>新建检索</h1>
        <p>选择一张动作参考图，再选择一段完整目标视频。</p>
      </div>

      <div className="selection-layout">
        <div className="selection-panel">
          <div className="panel-title">
            <span>QUERY</span>
            <h2>参考图片</h2>
          </div>
          <div className="compact-assets">
            {images.data?.items.map((asset) => (
              <AssetCard
                key={asset.id}
                asset={asset}
                selected={asset.id === queryId}
                onClick={() => setQueryId(asset.id)}
              />
            ))}
          </div>
        </div>
        <div className="selection-panel">
          <div className="panel-title">
            <span>TARGET</span>
            <h2>目标视频</h2>
          </div>
          <div className="compact-assets">
            {videos.data?.items.map((asset) => (
              <AssetCard
                key={asset.id}
                asset={asset}
                selected={asset.id === videoId}
                onClick={() => setVideoId(asset.id)}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="job-config">
        <div>
          <span className="config-label">MODEL</span>
          <strong>OpenCLIP · ViT-B/32</strong>
          <small>CPU · Top-1 · 固定模型</small>
        </div>
        <button
          className="text-button"
          onClick={() => setAdvanced((value) => !value)}
        >
          {advanced ? "收起参数" : "高级参数"}
        </button>
        {advanced && (
          <div className="parameter-grid">
            {numberField("sample_fps", "采样 FPS", 0.5)}
            {numberField("window_sec", "窗口长度（秒）", 0.5)}
            {numberField("stride_sec", "窗口步长（秒）", 0.1)}
            {numberField("top_frame_ratio", "高分帧比例", 0.1)}
            {numberField("nms_iou", "NMS IoU", 0.1)}
          </div>
        )}
        {create.error && (
          <div className="inline-error">
            {create.error instanceof Error
              ? create.error.message
              : "创建任务失败"}
          </div>
        )}
        <button
          className="button primary launch-button"
          disabled={!canSubmit}
          onClick={() =>
            create.mutate({
              query_image_id: queryId,
              target_video_id: videoId,
              parameters: params,
            })
          }
        >
          {create.isPending ? "正在创建…" : "开始识别 →"}
        </button>
      </div>
    </section>
  );
}
