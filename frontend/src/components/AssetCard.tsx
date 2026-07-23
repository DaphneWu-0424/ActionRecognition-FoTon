import { formatBytes, formatDuration } from "../lib/format";
import type { Asset } from "../types/api";

interface Props {
  asset: Asset;
  selected?: boolean;
  onClick?: () => void;
}

export function AssetCard({ asset, selected = false, onClick }: Props) {
  return (
    <button
      type="button"
      className={`asset-card ${selected ? "is-selected" : ""}`}
      onClick={onClick}
    >
      <div className="asset-visual">
        {asset.thumbnail_url ? (
          <img src={asset.thumbnail_url} alt="" loading="lazy" />
        ) : (
          <div className="asset-placeholder">NO PREVIEW</div>
        )}
        {asset.kind === "video" && (
          <span className="duration-chip">
            {formatDuration(asset.duration_sec)}
          </span>
        )}
        {selected && <span className="selected-mark">✓</span>}
      </div>
      <div className="asset-copy">
        <strong title={asset.original_filename}>{asset.name}</strong>
        <span>
          {asset.width} × {asset.height} · {formatBytes(asset.file_size)}
        </span>
      </div>
    </button>
  );
}
