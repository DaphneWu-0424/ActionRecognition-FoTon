import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { cancelJob, getJob, retryJob } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";
import {
  formatDuration,
  formatScore,
  shouldStopSegment,
} from "../lib/format";

const stageLabels: Record<string, string> = {
  probing_video: "读取视频",
  encoding_query: "编码参考图",
  encoding_frames: "编码视频帧",
  scoring_windows: "计算窗口分数",
  selecting_results: "筛选最佳区间",
  exporting_results: "导出结果",
  saving_results: "保存结果",
};

export function JobDetail() {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const stopAtRef = useRef<number | null>(null);
  const [playerMessage, setPlayerMessage] = useState("");
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      query.state.data &&
      ["queued", "running"].includes(query.state.data.status)
        ? 1000
        : false,
  });
  const action = useMutation({
    mutationFn: (type: "cancel" | "retry") =>
      type === "cancel" ? cancelJob(jobId) : retryJob(jobId),
    onSuccess: async (updated) => {
      if (updated.id !== jobId) {
        navigate(`/jobs/${updated.id}`);
      } else {
        await queryClient.invalidateQueries({ queryKey: ["job", jobId] });
      }
    },
  });
  const data = job.data;

  const playResult = () => {
    if (!data?.result || !videoRef.current) return;
    videoRef.current.currentTime = data.result.start_sec;
    stopAtRef.current = data.result.end_sec;
    setPlayerMessage(
      `正在播放 ${formatDuration(data.result.start_sec)}–${formatDuration(data.result.end_sec)}`,
    );
    void videoRef.current.play();
  };

  if (job.isLoading) {
    return <section className="page loading-page">正在读取任务…</section>;
  }
  if (!data || job.error) {
    return (
      <section className="page">
        <div className="notice error">
          {job.error instanceof Error ? job.error.message : "任务不存在"}
        </div>
      </section>
    );
  }

  const duration = data.target_video.duration_sec ?? 0;
  const left = data.result && duration
    ? (data.result.start_sec / duration) * 100
    : 0;
  const width = data.result && duration
    ? ((data.result.end_sec - data.result.start_sec) / duration) * 100
    : 0;

  return (
    <section className="page detail-page">
      <div className="detail-topline">
        <Link to="/jobs">← 返回任务记录</Link>
        <StatusBadge status={data.status} />
      </div>
      <div className="result-heading">
        <div>
          <span className="eyebrow">JOB / {data.id.slice(0, 8)}</span>
          <h1>{data.status === "succeeded" ? "识别完成" : "正在处理"}</h1>
          <p>
            {data.stage
              ? stageLabels[data.stage] ?? data.stage
              : data.error_message ?? "等待 Worker 领取任务"}
          </p>
        </div>
        {["queued", "running"].includes(data.status) && (
          <button
            className="button secondary"
            onClick={() => action.mutate("cancel")}
          >
            取消任务
          </button>
        )}
        {["failed", "cancelled"].includes(data.status) && (
          <button
            className="button primary"
            onClick={() => action.mutate("retry")}
          >
            重新运行
          </button>
        )}
      </div>

      {data.status !== "succeeded" ? (
        <div className="working-card">
          <div className="working-percent">
            {Math.round(data.progress * 100)}
            <span>%</span>
          </div>
          <div className="working-track">
            <span style={{ width: `${data.progress * 100}%` }} />
          </div>
          {data.error_message && (
            <div className="notice error">{data.error_message}</div>
          )}
        </div>
      ) : data.result ? (
        <>
          <div className="result-workspace">
            <aside className="query-aside">
              <span className="workspace-label">QUERY IMAGE</span>
              <img src={data.query_image.content_url} alt="Query" />
              <strong>{data.query_image.name}</strong>
            </aside>
            <div className="video-stage">
              <span className="workspace-label">TARGET VIDEO</span>
              <video
                ref={videoRef}
                controls
                preload="metadata"
                src={data.target_video.content_url}
                onSeeking={() => {
                  stopAtRef.current = null;
                  setPlayerMessage("");
                }}
                onTimeUpdate={(event) => {
                  const stop = stopAtRef.current;
                  if (shouldStopSegment(event.currentTarget.currentTime, stop)) {
                    event.currentTarget.pause();
                    stopAtRef.current = null;
                    setPlayerMessage("Top-1 区间播放完毕");
                  }
                }}
              />
              <div className="player-caption">
                <span>{data.target_video.name}</span>
                <span>{playerMessage}</span>
              </div>
            </div>
          </div>

          <div className="timeline-card">
            <div className="timeline-labels">
              <span>0:00</span>
              <strong>TOP-1 TIMELINE</strong>
              <span>{formatDuration(duration)}</span>
            </div>
            <button className="timeline" onClick={playResult}>
              <span
                className="timeline-hit"
                style={{
                  left: `${left}%`,
                  width: `${Math.max(width, 1.5)}%`,
                }}
              >
                TOP-1
              </span>
            </button>
          </div>

          <div className="result-summary">
            <div className="score-panel">
              <span>CONFIDENCE</span>
              <strong>{formatScore(data.result.score)}</strong>
              <small>OpenCLIP similarity</small>
            </div>
            <button className="interval-panel" onClick={playResult}>
              <span>BEST INTERVAL</span>
              <strong>
                {formatDuration(data.result.start_sec)}
                <i>→</i>
                {formatDuration(data.result.end_sec)}
              </strong>
              <small>点击在原视频中播放</small>
            </button>
            <a
              className="frame-panel"
              href={data.result.best_frame_url}
              target="_blank"
              rel="noreferrer"
            >
              <img src={data.result.best_frame_url} alt="最佳帧" />
              <span>BEST FRAME · {formatDuration(data.result.best_frame_sec)}</span>
            </a>
            <a
              className="clip-link"
              href={data.result.clip_url}
              target="_blank"
              rel="noreferrer"
            >
              单独播放 Top-1 片段 ↗
            </a>
          </div>
        </>
      ) : null}
    </section>
  );
}
