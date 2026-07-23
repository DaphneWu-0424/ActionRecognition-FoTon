# Action Recognition Local Web App

本项目使用 OpenCLIP，在完整视频中定位与一张 Query 图片最相似的 Top-1 时间区间。当前目标是 Windows、CPU、本地单用户应用。

设计和实施状态：

- [DESIGN.md](DESIGN.md)
- [APP_TODO.md](APP_TODO.md)

仓库根目录本地保留的 `TODO.md` 是研究实验路线，不属于 Web MVP，因此继续由 `.gitignore` 排除。

## 功能

- 上传图片/视频，或扫描配置好的本地目录。
- 选择一张 Query 图片和一个完整目标视频创建异步任务。
- OpenCLIP 在 CPU 上完成抽帧、滑窗打分和 Temporal NMS。
- 只持久化 Top-1 区间、分数、最佳帧和导出片段。
- 在完整视频时间轴上定位并播放命中区间。
- 查看任务进度，取消运行，检查失败原因并重试。
- SQLite 保存任务元数据，真实媒体保存在文件系统中。

## 架构

```text
React / Vite :5173
        │ HTTP + Range
        ▼
FastAPI :8010 ─────── SQLite
        │                 ▲
        │                 │ claim / progress / result
        ▼                 │
local media storage ◀─ Inference Worker
                       OpenCLIP + OpenCV
```

API 只负责校验、持久化和媒体响应。CPU 推理由独立 Worker 执行，不会阻塞 HTTP 请求。当前服务仅绑定 `127.0.0.1`，不包含登录和远程访问能力。

## 目录

```text
backend/
├── app/
│   ├── api/          # 媒体、任务、健康检查接口
│   ├── db/           # SQLAlchemy 模型和会话
│   ├── inference/    # OpenCLIP locator
│   ├── services/     # 存储、媒体探测、任务服务
│   ├── main.py
│   └── worker.py
├── migrations/       # Alembic migrations
└── tests/
frontend/             # React + TypeScript + Vite
scripts/
├── start_app.ps1
└── smoke_web_backend.py
```

## 环境

- Python 3.10.8
- Node.js 24.12.0 / npm 11.6.2
- PyTorch 2.13.0+cpu
- OpenCLIP 3.3.0
- OpenCV 5.0.0
- FastAPI 0.139.2
- SQLAlchemy 2.0.51
- SQLite

系统当前未在 `PATH` 中提供 FFmpeg/ffprobe。应用使用 OpenCV 探测视频和导出片段；健康检查会显示 FFmpeg 不可用，但这不会阻塞本地 MVP。

## Python 依赖

所有 Python 包安装到仓库内的 `.venv`：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

不要使用全局 Python 安装项目依赖。

前端依赖：

```powershell
Set-Location frontend
npm install
Set-Location ..
```

## 配置

默认配置可直接在 `D:\ActionRecognition` 使用。需要覆盖时：

```powershell
Copy-Item .env.example .env
```

媒体扫描只接受服务端配置的根目录 ID，不接受任意绝对路径。上传媒体和应用数据库写入 `data/app_storage/`，该目录不会提交到 Git。

## 开发启动

首次启动数据库迁移：

```powershell
$env:PYTHONPATH=".;backend"
.\.venv\Scripts\python.exe -m alembic upgrade head
```

一键启动 API、Worker 和前端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_app.ps1
```

启动后访问：

- Web：<http://127.0.0.1:5173>
- Swagger：<http://127.0.0.1:8010/docs>
- 健康检查：<http://127.0.0.1:8010/api/health>

按 `Ctrl+C` 停止三个进程。

也可以在三个 PowerShell 窗口中分别运行：

```powershell
$env:PYTHONPATH=".;backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8010
```

```powershell
$env:PYTHONPATH=".;backend"
.\.venv\Scripts\python.exe -m app.worker
```

```powershell
Set-Location frontend
npm run dev
```

## 使用流程

1. 在“媒体库”点击“扫描 inputs”，或分别上传 Query 图片和目标视频。
2. 打开“新建检索”，选择一张图片和一个视频。
3. 使用默认参数创建任务。
4. Worker 完成后，结果页会显示 Top-1 区间、分数、最佳帧和片段。
5. 点击时间轴或区间卡片，在完整视频中播放对应区间。

Web 应用只持久化 Top-1。现有研究脚本的 Top-K 行为不变。

## 验证

后端测试：

```powershell
$env:PYTHONPATH=".;backend"
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

前端检查：

```powershell
Set-Location frontend
npm run lint
npm run test
npm run build
```

真实 OpenCLIP 后端闭环：

```powershell
$env:PYTHONPATH=".;backend"
.\.venv\Scripts\python.exe scripts\smoke_web_backend.py
```

该脚本使用 `inputs/query.png` 和 `inputs/test_video.mp4`，会创建一条真实任务和 Top-1 结果。

2026-07-23 的本机 CPU smoke test 记录：

- 输入视频：12 秒、320×240、25 FPS
- 采样：2 FPS，2 秒窗口，0.5 秒步长
- 进程总耗时：约 6.0 秒（包含 Python/API 测试客户端与模型加载）
- Python 进程峰值工作集：约 1.4 GB
- CLI 与 Worker 均得到 Top-1 `[6.0s, 8.0s)`，分数 `0.9566456`

## 数据、备份与清理

应用数据位于：

```text
data/app_storage/
├── app.sqlite3
├── sources/
├── derived/
└── tmp/
```

备份时先停止 API 和 Worker，再复制整个 `data/app_storage/`。扫描引用的原视频不在该目录内，需要单独备份原始数据目录。

删除媒体使用 Web/API 的软删除功能，不要手工删除仍被任务引用的文件。`tmp/` 中只保存未提交的临时产物；Worker 正常结束后会清理。

## 常见问题

### 健康检查显示 FFmpeg 为 false

当前电脑没有把 FFmpeg/ffprobe 加入 `PATH`。应用会使用 OpenCV 探测视频并导出片段，属于正常降级状态。

### 任务一直停留在 queued

确认 Worker 窗口仍在运行：

```powershell
$env:PYTHONPATH=".;backend"
.\.venv\Scripts\python.exe -m app.worker
```

### 端口无法启动

本项目使用前端 `5173` 和 API `8010`。检查是否有其他进程占用这些端口。

### 模型无法加载

确认 `models/openclip_vit_b32_openai/` 存在，并检查 `.env` 中的 `ACTION_APP_MODEL_CACHE`。模型已经缓存时，正常使用不需要联网。
