import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { listAssets, scanAssets, uploadAsset } from "../api/client";
import { AssetCard } from "../components/AssetCard";
import { EmptyState } from "../components/EmptyState";

export function AssetLibrary() {
  const [kind, setKind] = useState<"image" | "video">("image");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [notice, setNotice] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const assets = useQuery({
    queryKey: ["assets", kind, search, page],
    queryFn: () => listAssets(kind, search, page),
  });
  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: ["assets"] });
  const upload = useMutation({
    mutationFn: uploadAsset,
    onSuccess: async () => {
      setNotice("文件已入库");
      await refresh();
    },
  });
  const scan = useMutation({
    mutationFn: scanAssets,
    onSuccess: async (data) => {
      setNotice(`扫描完成：新增 ${data.created}，已有 ${data.skipped}`);
      await refresh();
    },
  });
  const error = upload.error ?? scan.error ?? assets.error;

  return (
    <section className="page">
      <div className="page-heading split-heading">
        <div>
          <span className="eyebrow">01 / MEDIA</span>
          <h1>媒体库</h1>
          <p>登记 Query 图片和目标视频。扫描目录时只建立引用，不复制原视频。</p>
        </div>
        <div className="heading-actions">
          <input
            ref={inputRef}
            hidden
            type="file"
            accept={kind === "image" ? "image/*" : "video/*"}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) upload.mutate(file);
              event.target.value = "";
            }}
          />
          <button
            className="button secondary"
            onClick={() => scan.mutate("inputs")}
            disabled={scan.isPending}
          >
            扫描 inputs
          </button>
          <button
            className="button primary"
            onClick={() => inputRef.current?.click()}
            disabled={upload.isPending}
          >
            {upload.isPending ? "上传中…" : "上传文件"}
          </button>
        </div>
      </div>

      {(notice || error) && (
        <div className={`notice ${error ? "error" : ""}`}>
          {error instanceof Error ? error.message : notice}
        </div>
      )}

      <div className="filter-bar">
        <div className="segmented">
          <button
            className={kind === "image" ? "active" : ""}
            onClick={() => {
              setKind("image");
              setPage(1);
            }}
          >
            图片
          </button>
          <button
            className={kind === "video" ? "active" : ""}
            onClick={() => {
              setKind("video");
              setPage(1);
            }}
          >
            视频
          </button>
        </div>
        <input
          className="search-input"
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            setPage(1);
          }}
          placeholder="搜索文件名"
        />
        <span className="result-count">{assets.data?.total ?? 0} ITEMS</span>
      </div>

      {assets.isLoading ? (
        <div className="loading-grid">正在读取媒体库…</div>
      ) : assets.data?.items.length ? (
        <div className="asset-grid">
          {assets.data.items.map((asset) => (
            <AssetCard key={asset.id} asset={asset} />
          ))}
        </div>
      ) : (
        <EmptyState
          title="这里还没有媒体"
          body="上传文件，或扫描项目 inputs 目录开始使用。"
        />
      )}
      {(assets.data?.total ?? 0) > 24 && (
        <div className="pagination">
          <button
            className="button secondary"
            disabled={page === 1}
            onClick={() => setPage((value) => value - 1)}
          >
            ← 上一页
          </button>
          <span>
            {page} / {Math.ceil((assets.data?.total ?? 0) / 24)}
          </span>
          <button
            className="button secondary"
            disabled={page * 24 >= (assets.data?.total ?? 0)}
            onClick={() => setPage((value) => value + 1)}
          >
            下一页 →
          </button>
        </div>
      )}
    </section>
  );
}
