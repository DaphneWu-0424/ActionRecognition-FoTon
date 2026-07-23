# Action Recognition Web App Design

> 文档状态：Draft for review
> 更新时间：2026-07-23
> 实施清单：[APP_TODO.md](APP_TODO.md)

## 1. 背景与结论

当前仓库已经实现最关键的 OpenCLIP 时序检索链路：

```text
Query 图片 + 完整视频
        ↓
视频采样与图像编码
        ↓
滑动窗口打分
        ↓
Temporal NMS
        ↓
Top-K 区间、片段、最佳帧、JSON/CSV
```

已有核心位于 `scripts/search_image_in_video.py`，并由 THUMOS14 和 HMDB51 脚本用于演示和评测。仓库目前没有后端、数据库、Worker 或前端。

本设计建议采用：

```text
React + TypeScript + Vite
FastAPI + SQLAlchemy + SQLite
本地文件存储
独立单进程 Worker
OpenCLIP 推理适配器
```

首版面向 Windows、CPU、单机、本地单用户，仅绑定 `127.0.0.1`，不实现登录。架构边界保留未来迁移 Linux/GPU、PostgreSQL、对象存储和专业任务队列的可能，但不提前引入这些基础设施。

Web MVP 只开放 OpenCLIP，支持单张 Query 图片检索单个目标视频，并且只长期保存 Top-1。现有研究脚本仍保留 Top-K 能力，但其他候选不进入 Web 应用的数据库、API 和页面。

## 2. 产品范围

### 2.1 核心用户流程

1. 用户上传文件，或让系统扫描已配置目录。
2. 用户在媒体库选择一张 Query 图片和一个目标视频。
3. 用户使用默认参数或调整滑窗参数后创建任务。
4. API 立即返回任务 ID，Worker 异步执行检索。
5. 用户在任务页查看状态、阶段和进度。
6. 成功后，结果页展示完整视频上的 Top-1 时间区间。
7. 用户点击该区间跳转播放，并查看最佳帧或导出片段。

### 2.2 MVP 成功标准

- 单用户可通过浏览器完成端到端操作。
- 长时间推理不占用 HTTP 请求。
- 任务、参数和结果可复现、可追踪。
- 页面刷新或 API 重启不会丢失任务记录。
- 源媒体始终与派生结果分离。
- 当前 CLI 和研究脚本继续可用。

### 2.3 非目标

MVP 不解决多租户、权限、分布式调度、实时视频流、训练/微调和复杂动作流程推理。

## 3. 关键设计决策

| 主题 | 决策 | 原因 |
| --- | --- | --- |
| 后端 | FastAPI | Python 推理代码可直接复用，且自动提供 OpenAPI。 |
| 前端 | React + TypeScript + Vite | 适合媒体管理和交互式时间轴单页应用。 |
| 数据库 | SQLite | 单机、单 Worker 下部署成本最低；ORM 隔离未来迁移。 |
| 媒体存储 | 本地文件系统 | Worker 可直接读取，适合现有数据和 Windows 环境。 |
| 异步任务 | 独立 Worker + 数据库轮询 | 隔离重计算，不在 MVP 引入 Redis。 |
| 进度更新 | 1 秒轮询 | 实现简单；后续可无损升级 SSE。 |
| 推理调用 | Python 模块接口 | 模型只加载一次，便于测试和未来 GPU 化。 |
| CLI | 保留为薄封装 | 保持当前实验和调试工作流兼容。 |
| 结果保存 | 仅持久化 Top-1 | 满足当前使用需求，并控制数据库和派生媒体规模。 |
| 结果播放 | 原视频跳转为主，Top-1 导出片段为辅 | 时间轴体验更连贯，同时保留可复现产物。 |

### 3.1 关于 subprocess

为了尽快验证 API，可以短期用 `subprocess` 桥接现有脚本，但它不是目标架构。正式 MVP 应先将检索核心抽成可导入模块，再由 Worker 直接调用。

直接调用的收益：

- Worker 启动时只加载一次 OpenCLIP。
- 可用回调报告抽帧和编码进度。
- 可在安全检查点响应取消。
- 可为推理逻辑注入 fake locator 做测试。
- 异常与结果不必通过 stdout/JSON 文件二次解析。

## 4. 系统架构

```text
┌──────────────────────────────────────────────────────┐
│ React Web App                                        │
│ 媒体库 │ 创建任务 │ 任务列表 │ 结果播放器             │
└───────────────────────┬──────────────────────────────┘
                        │ HTTP JSON / Range
                        ▼
┌──────────────────────────────────────────────────────┐
│ FastAPI                                              │
│ Assets API │ Jobs API │ Detections API │ Media API   │
└──────────────┬───────────────────────┬───────────────┘
               │ SQL                   │ storage key
               ▼                       ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ SQLite                  │  │ Local StorageAdapter    │
│ assets/jobs/results     │  │ sources/derived/tmp    │
└──────────────▲──────────┘  └──────────────▲──────────┘
               │ claim/update/result         │ read/write
               │                             │
┌──────────────┴─────────────────────────────┴──────────┐
│ Single Inference Worker                              │
│ claim → localize → export → persist                  │
│ OpenClipTemporalLocator loaded once                  │
└──────────────────────────────────────────────────────┘
```

### 4.1 进程边界

开发环境运行三个进程：

```text
frontend dev server
FastAPI server
inference worker
```

重计算不放在 FastAPI 进程，也不使用 FastAPI `BackgroundTasks`。API 负责校验、持久化和媒体响应；Worker 独占模型与 CPU 推理。

### 4.2 推荐目录

```text
ActionRecognition/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── db/
│   │   ├── inference/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── config.py
│   │   ├── main.py
│   │   └── worker.py
│   ├── migrations/
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   └── types/
│   └── package.json
├── scripts/
├── data/
│   └── app_storage/
├── APP_TODO.md
└── DESIGN.md
```

现有 `data/`、`inputs/`、`outputs/`、`cache/` 和研究脚本不移动。应用通过扫描白名单登记现有媒体，不把研究目录重组为应用目录。

## 5. 推理模块

### 5.1 接口

```python
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalizationParams:
    sample_fps: float = 2.0
    window_sec: float = 2.0
    stride_sec: float = 0.5
    top_frame_ratio: float = 0.5
    top_k: int = 1
    nms_iou: float = 0.3


@dataclass(frozen=True)
class DetectionResult:
    rank: int
    start_sec: float
    end_sec: float
    score: float
    mean_score: float
    max_score: float
    best_frame_sec: float
    clip_path: Path
    best_frame_path: Path


ProgressCallback = Callable[[str, float], None]
CancelCheck = Callable[[], bool]


class TemporalActionLocator:
    def load(self) -> None: ...

    def localize(
        self,
        query_image: Path,
        target_video: Path,
        output_dir: Path,
        params: LocalizationParams,
        on_progress: ProgressCallback,
        is_cancelled: CancelCheck,
    ) -> list[DetectionResult]: ...
```

### 5.2 模型生命周期

```text
Worker 启动
  → 创建 locator
  → load() 一次
  → 循环领取任务
  → localize() 多次
```

若后续启用多个模型，MVP 阶段一个 Worker 只配置一个启用模型。模型注册表可以存在，但不做动态并发加载和内存淘汰。

### 5.3 进度映射

| 阶段 | 进度范围 |
| --- | ---: |
| `loading_model` | Worker 启动，不属于单任务进度 |
| `probing_video` | 0%–5% |
| `encoding_query` | 5%–10% |
| `encoding_frames` | 10%–75% |
| `scoring_windows` | 75%–85% |
| `selecting_results` | 85%–90% |
| `exporting_results` | 90%–98% |
| `saving_results` | 98%–100% |

帧编码进度按“已采样帧数/预计采样帧数”计算。其他阶段使用粗粒度固定范围，前端应将其理解为近似进度。

### 5.4 取消语义

取消是协作式的，不强杀进程。API 写入 `cancel_requested_at`，Worker 在批次编码之间、打分前和每个结果导出前检查。取消后：

- 状态设为 `cancelled`。
- 临时产物被清理。
- 已提交的源媒体不删除。
- 不创建部分结果记录。

## 6. 数据与存储设计

### 6.1 存储布局

```text
data/app_storage/
├── sources/
│   ├── images/
│   └── videos/
├── derived/
│   ├── thumbnails/
│   └── jobs/{job_id}/
│       ├── query.png
│       ├── rank_01_clip.mp4
│       ├── rank_01_best_frame.png
│       ├── frame_scores.csv
│       └── window_scores.csv
└── tmp/
```

上传文件由应用复制到 `sources/`。扫描已有目录时默认采用“引用模式”，即资产记录引用白名单目录中的文件，不复制大视频。两种来源均统一为 `StorageAdapter` 可解析的存储键。

数据库禁止保存任意绝对路径作为可直接读取的用户输入。解析过程必须：

1. 根据受控 storage key 找到配置的根目录。
2. 解析规范绝对路径。
3. 验证结果仍位于允许的根目录内。

### 6.2 `media_assets`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | UUID/string PK | 公开 ID |
| `kind` | enum | `image/video/clip/frame` |
| `name` | string | 显示名称 |
| `original_filename` | string | 原文件名 |
| `storage_provider` | string | MVP 为 `local` |
| `storage_key` | string | 受控相对键 |
| `mime_type` | string | 探测后的类型 |
| `file_size` | integer | 字节数 |
| `sha256` | string | 去重和变更检测 |
| `duration_sec` | float nullable | 视频时长 |
| `width/height` | integer nullable | 媒体尺寸 |
| `fps` | float nullable | 视频 FPS |
| `thumbnail_asset_id` | FK nullable | 缩略图 |
| `source_asset_id` | FK nullable | 派生资源对应的源视频 |
| `metadata_json` | JSON/text | 编解码等扩展信息 |
| `created_at` | datetime | 创建时间 |
| `deleted_at` | datetime nullable | 软删除 |

唯一性建议为 `(storage_provider, storage_key)`。SHA-256 用于提示重复，不单独作为全局唯一键，因为同一内容可能需要不同业务名称或来源记录。

### 6.3 `inference_jobs`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | UUID/string PK | 任务 ID |
| `retry_of_job_id` | FK nullable | 重试来源任务 |
| `query_image_id` | FK | 必须是 image |
| `target_video_id` | FK | 必须是 video |
| `model_key` | string | 如 `openclip-vit-b32-openai` |
| `status` | enum | 任务状态 |
| `stage` | enum/string nullable | 当前阶段 |
| `progress` | float | 0 到 1 |
| `parameters_json` | JSON/text | 完整参数快照 |
| `error_code` | string nullable | 稳定错误码 |
| `error_message` | text nullable | 用户可读摘要 |
| `error_log_key` | string nullable | 详细日志 |
| `created_at` | datetime | 创建时间 |
| `started_at` | datetime nullable | 开始时间 |
| `heartbeat_at` | datetime nullable | Worker 心跳 |
| `cancel_requested_at` | datetime nullable | 取消请求 |
| `finished_at` | datetime nullable | 终态时间 |

### 6.4 `localization_results`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | UUID/string PK | 结果 ID |
| `job_id` | FK unique | 所属任务，一对一 |
| `start_sec/end_sec` | float | 半开区间 `[start, end)` |
| `score` | float | 当前综合分数 |
| `mean_score/max_score` | float | 可解释分数 |
| `best_frame_sec` | float | 最佳帧位置 |
| `clip_asset_id` | FK | 导出片段 |
| `best_frame_asset_id` | FK | 最佳帧 |
| `metadata_json` | JSON/text | 模型扩展结果 |

只有 Top-1 推理和导出全部成功后，才在一个事务中提交唯一的 `localization_results` 记录并将任务改为 `succeeded`。其他候选不写数据库，也不长期保留对应片段或最佳帧。

### 6.5 模型配置

MVP 的可用模型由服务端配置文件定义，不开放任意模型路径给前端。Web 应用固定 `top_k=1`，不把它作为前端可调参数。任务保存 `model_key` 和最终参数快照，保证后续默认值变化时仍可知道当时实际配置。

暂不创建可由前端修改的 `model_configs` 数据表；当需要管理员动态启停模型时再引入，避免首版出现任意本地路径加载风险。

## 7. 任务状态机

```text
queued ───────────────→ running ───────────────→ succeeded
  │                       │  │
  │ cancel                │  ├─ cancel request → cancelled
  ▼                       │  └─ exception ─────→ failed
cancelled                 │
                          └─ stale heartbeat ──→ failed

failed/succeeded/cancelled --retry--> 新建 queued 任务
```

约束：

- 终态为 `succeeded`、`failed`、`cancelled`。
- Retry 不复用原任务 ID；新任务保存 `retry_of_job_id`。
- 排队任务取消可直接进入 `cancelled`。
- 运行中任务取消先记录请求，由 Worker 最终写入 `cancelled`。
- API 不允许客户端直接修改 status、stage 或 progress。

### 7.1 SQLite 领取任务

MVP 只运行一个 Worker。领取操作仍需使用短事务，选择最早的 `queued` 任务并条件更新：

```sql
UPDATE inference_jobs
SET status = 'running', started_at = :now, heartbeat_at = :now
WHERE id = :id AND status = 'queued';
```

只有受影响行数为 1 才算领取成功。未来启用多 Worker 或 PostgreSQL 时，再切换为更严格的行锁/跳过锁定方案。

## 8. API 设计

统一前缀 `/api`，JSON 字段使用 `snake_case`。列表接口统一返回：

```json
{
  "items": [],
  "page": 1,
  "page_size": 20,
  "total": 0
}
```

错误统一返回：

```json
{
  "error": {
    "code": "INVALID_MEDIA_TYPE",
    "message": "Query asset must be an image.",
    "details": {}
  }
}
```

### 8.1 媒体

```http
GET    /api/assets
POST   /api/assets/upload
POST   /api/assets/scan
GET    /api/assets/{asset_id}
GET    /api/assets/{asset_id}/thumbnail
GET    /api/assets/{asset_id}/content
DELETE /api/assets/{asset_id}
```

`POST /assets/scan` 不接收任意路径。请求只接收服务端配置好的 `scan_root_id`，可选相对目录和递归开关。

视频 `content` 必须支持单区间 HTTP Range，返回 `206`、`Content-Range` 和 `Accept-Ranges: bytes`，以支持浏览器拖动。

### 8.2 任务

```http
POST /api/jobs
GET  /api/jobs
GET  /api/jobs/{job_id}
POST /api/jobs/{job_id}/cancel
POST /api/jobs/{job_id}/retry
GET  /api/jobs/{job_id}/result
```

创建请求：

```json
{
  "query_image_id": "asset-id",
  "target_video_id": "asset-id",
  "parameters": {
    "sample_fps": 2.0,
    "window_sec": 2.0,
    "stride_sec": 0.5,
    "top_frame_ratio": 0.5,
    "nms_iou": 0.3
  }
}
```

服务端固定使用 `openclip-vit-b32-openai` 和 `top_k=1`。两者不由前端提交，避免出现界面允许选择、Worker 实际无法执行的配置。

成功响应为 `202 Accepted`：

```json
{
  "id": "job-id",
  "status": "queued",
  "created_at": "2026-07-23T17:30:00+08:00"
}
```

任务详情：

```json
{
  "id": "job-id",
  "status": "succeeded",
  "stage": "saving_results",
  "progress": 1.0,
  "query_image": {
    "id": "query-id",
    "content_url": "/api/assets/query-id/content"
  },
  "target_video": {
    "id": "video-id",
    "duration_sec": 149.2,
    "content_url": "/api/assets/video-id/content"
  },
  "parameters": {
    "sample_fps": 2.0,
    "window_sec": 2.0,
    "stride_sec": 0.5,
    "top_frame_ratio": 0.5,
    "top_k": 1,
    "nms_iou": 0.3
  },
  "result": {
    "id": "result-id",
    "start_sec": 48.5,
    "end_sec": 50.5,
    "score": 0.814,
    "best_frame_sec": 49.5,
    "clip_url": "/api/assets/clip-id/content",
    "best_frame_url": "/api/assets/frame-id/content"
  }
}
```

任务详情内嵌唯一的 Top-1 结果，`/result` 接口用于单独读取该资源。任务未成功时 `result` 为 `null`。

### 8.3 健康检查

```http
GET /api/health
```

检查数据库可连接、存储目录可写以及 FFmpeg/ffprobe 是否可用。模型真实加载检查不应放在每次健康请求中。MVP 不额外维护空闲 Worker 心跳；运行中任务只通过 `heartbeat_at` 判断是否失联。后续若需要独立的 Worker 在线状态，再增加 Worker 注册/心跳表。

## 9. 前端设计

### 9.1 路由

```text
/assets
/jobs/new
/jobs
/jobs/:jobId
```

任务 ID 在 URL 中，页面刷新后仍可恢复状态。

### 9.2 媒体库

- 图片/视频 Tab。
- 搜索、分页、类型和来源筛选。
- 卡片显示缩略图、文件名、时长、尺寸。
- 上传入口和扫描已配置目录入口。
- 损坏或源文件丢失的资产明确标记，不允许用于创建任务。

### 9.3 创建任务

```text
┌─────────────────────┬────────────────────────┐
│ Query 图片选择/预览  │ 目标视频选择/预览       │
└─────────────────────┴────────────────────────┘
┌──────────────────────────────────────────────┐
│ 模型：OpenCLIP ViT-B/32（固定）             │
│ 推荐参数 | 高级参数                         │
│                              [创建检索任务]  │
└──────────────────────────────────────────────┘
```

默认只显示模型和“创建”按钮；滑窗参数放在可折叠高级设置中，减少普通用户误配。

### 9.4 任务列表

显示状态、阶段、进度、Query 缩略图、视频名、创建/完成时间和操作。运行任务每秒轮询，终态停止轮询。

### 9.5 结果页

```text
┌──────────────┬────────────────────────────────┐
│ Query 图片   │ 完整视频播放器                 │
├──────────────┴────────────────────────────────┤
│ 0s ──────────────[Top-1]─────────── 149.2s   │
├───────────────────────────────────────────────┤
│ 区间         分数     最佳帧       片段        │
│ 48.5–50.5    .814     [图]         [播放]      │
└───────────────────────────────────────────────┘
```

交互：

- 点击时间轴标记或结果行，将 `video.currentTime` 设为 `start_sec` 并播放。
- 通过 `timeupdate` 监听，到达 `end_sec` 时暂停。
- 用户手动拖动后取消当前“区间自动停止”状态。
- 时间轴区间按视频总时长换算百分比。
- Top-1 片段 URL 用于单独播放/下载，主要视图仍以完整视频为准。

## 10. 可靠性与安全

### 10.1 文件安全

- 上传文件名不参与实际存储路径生成。
- 使用 UUID/哈希生成存储键。
- 扫描只允许预配置根目录。
- 所有解析后的路径必须验证仍在允许根目录内。
- 限制上传大小、类型和数量。
- MIME/媒体探测结果优先于文件扩展名。
- 不向前端返回服务器绝对路径。

### 10.2 数据一致性

- 源媒体、临时产物和已提交派生资产分目录。
- 推理先写 `tmp/job_id`，全部成功后再移动并提交数据库。
- 唯一的 Top-1 result 和 `succeeded` 状态在一个数据库事务中提交。
- 删除采用软删除；有任务引用的源资产不可物理删除。
- 后续清理器只清除无数据库引用且超过保留期的临时文件。

### 10.3 故障恢复

- Worker 定期写 `heartbeat_at`。
- 启动时将超时的 `running` 任务标为 `failed`，错误码为 `WORKER_LOST`。
- 失败任务不自动重跑，避免重复高成本工作或覆盖产物。
- 用户显式 Retry 会创建新任务并复制参数。
- 错误响应保存短摘要；详细 traceback 写日志文件，不直接暴露给前端。

### 10.4 SQLite 约束

SQLite 适用于当前单机、单 Worker。开启 WAL 和 busy timeout，数据库事务保持短小。以下条件任一出现时应评估 PostgreSQL：

- 多个 Worker 并发领取任务。
- 多用户频繁写入。
- 需要远程部署 API 与 Worker。
- 任务与资产规模导致查询/备份明显变慢。

## 11. 测试策略

### 11.1 单元测试

- 时序窗口、IoU、NMS 和参数校验。
- 存储键规范化与路径逃逸。
- 状态机合法/非法迁移。
- 播放器时间区间控制与时间格式化。

### 11.2 API/Worker 集成测试

- 使用临时 SQLite 和临时存储目录。
- 使用小图片/视频夹具。
- Worker 注入 fake locator，快速覆盖成功、失败、取消和空结果。
- 验证 Range 响应头和部分内容。

### 11.3 真实模型 smoke test

使用仓库中一个已验证的 Query/视频组合运行 OpenCLIP，检查：

- 任务最终成功。
- Top-1 区间有效。
- Top-1 片段与最佳帧可读取。
- CLI 与 Worker 结果在容差内一致。

真实模型测试标记为 slow，不作为每次普通单元测试的必要条件。

## 12. 可观测性

日志至少包含：

- `request_id`、`job_id`。
- 任务状态变化。
- 每阶段耗时。
- 视频时长、采样帧数和推理参数。
- 模型键和代码版本。
- 异常类型与 traceback 日志位置。

首版使用结构化文本/JSON 日志，不引入监控平台。任务详情暴露用户需要的状态和错误摘要，不暴露内部路径或 traceback。

## 13. 演进路径

MVP 完成后按实际瓶颈升级：

1. Gallery 帧特征缓存，避免同一视频重复编码。
2. SSE 替换高频轮询。
3. 多视频批量检索与批任务。
4. LanguageBind/InternVideo2 适配器。
5. GPU Worker。
6. PostgreSQL + 专业任务队列。
7. NAS/MinIO `StorageAdapter`。
8. 用户、角色、审计和人工反馈。

模型、存储和队列都通过接口边界演进，前端的资产、任务和结果概念保持稳定。

## 14. 已确认决策

1. 首版为 Windows + CPU + 单机 + 单 Worker + 本地单用户。
2. 服务只绑定 `127.0.0.1`，不实现登录。
3. 扫描已有数据时引用原文件，不复制大视频。
4. MVP 只开放 OpenCLIP，一个任务只检索一个目标视频。
5. Web 应用只持久化 Top-1 区间、片段和最佳帧；Top-K 只保留为研究脚本能力。
6. 先完成 Swagger 后端闭环，review 通过后再开发 React 页面。
