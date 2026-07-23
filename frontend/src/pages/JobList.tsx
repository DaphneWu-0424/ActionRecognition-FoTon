import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { cancelJob, listJobs, retryJob } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";
import { formatDate } from "../lib/format";

export function JobList() {
  const queryClient = useQueryClient();
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => listJobs(),
    refetchInterval: (query) =>
      query.state.data?.items.some((job) =>
        ["queued", "running"].includes(job.status),
      )
        ? 1000
        : false,
  });
  const action = useMutation({
    mutationFn: ({ id, type }: { id: string; type: "cancel" | "retry" }) =>
      type === "cancel" ? cancelJob(id) : retryJob(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  return (
    <section className="page">
      <div className="page-heading split-heading">
        <div>
          <span className="eyebrow">03 / HISTORY</span>
          <h1>任务记录</h1>
          <p>任务独立于页面运行。关闭浏览器不会中断 Worker。</p>
        </div>
        <Link className="button primary" to="/jobs/new">
          新建检索
        </Link>
      </div>
      <div className="job-table">
        <div className="job-row job-header">
          <span>QUERY / VIDEO</span>
          <span>STATUS</span>
          <span>PROGRESS</span>
          <span>CREATED</span>
          <span />
        </div>
        {jobs.data?.items.map((job) => (
          <div className="job-row" key={job.id}>
            <Link to={`/jobs/${job.id}`} className="job-media">
              {job.query_image.thumbnail_url && (
                <img src={job.query_image.thumbnail_url} alt="" />
              )}
              <span>
                <strong>{job.query_image.name}</strong>
                <small>→ {job.target_video.name}</small>
              </span>
            </Link>
            <StatusBadge status={job.status} />
            <div className="progress-cell">
              <div>
                <span style={{ width: `${job.progress * 100}%` }} />
              </div>
              <small>{Math.round(job.progress * 100)}%</small>
            </div>
            <time>{formatDate(job.created_at)}</time>
            <div className="row-actions">
              {["queued", "running"].includes(job.status) && (
                <button
                  onClick={() =>
                    action.mutate({ id: job.id, type: "cancel" })
                  }
                >
                  取消
                </button>
              )}
              {["failed", "cancelled"].includes(job.status) && (
                <button
                  onClick={() =>
                    action.mutate({ id: job.id, type: "retry" })
                  }
                >
                  重试
                </button>
              )}
              <Link to={`/jobs/${job.id}`}>查看</Link>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
