# Action Recognition Web App TODO

> 文档状态：Draft for review
> 更新时间：2026-07-23
> 配套设计：[DESIGN.md](DESIGN.md)
> 研究与模型实验路线仍见：[TODO.md](TODO.md)

## 1. MVP 目标

在当前 Windows、CPU、单机环境中打通以下闭环：

```text
登记/上传 Query 图片与目标视频
        ↓
创建异步检索任务
        ↓
Worker 调用现有 OpenCLIP 时序检索
        ↓
只持久化 Top-1 时间区间、分数、片段和最佳帧
        ↓
前端查看进度、播放原视频、点击区间跳转
```

MVP 验收必须同时满足：

- [x] 用户不需要运行 Python 命令即可完成一次检索。
- [x] API 请求不会阻塞等待整段视频推理完成。
- [x] 服务或页面刷新后，任务与结果仍可查询。
- [x] 结果页能展示 Query 图片、完整视频、Top-1 区间、分数和最佳帧。
- [x] 点击结果可跳转到原视频的 `start_sec`，并在 `end_sec` 附近停止。
- [x] 推理失败时能看到可理解的错误信息，Worker 可继续处理下一任务。
- [x] 现有 `scripts/search_image_in_video.py` 命令行用法保持可用。

已确认的产品约束：

- 本地单用户，仅绑定 `127.0.0.1`，不实现登录。
- Windows + CPU + 单机 + 单 Worker。
- MVP 只开放 OpenCLIP，单张 Query 图片检索单个目标视频。
- 扫描已有数据时引用原文件，不复制大视频。
- 只长期保存 Top-1；其他候选不进入数据库、API 或前端。
- 先完成 Swagger 后端闭环，再开发 React 页面。

## 2. 已有基础

- [x] 单张图片编码。
- [x] 视频按 FPS 采样并批量编码。
- [x] 滑动窗口打分。
- [x] Temporal NMS 与 Top-K。
- [x] 导出命中片段和最佳帧。
- [x] 输出 `predictions.json`、窗口分数和帧分数。
- [x] THUMOS14 单例与批量评测脚本。
- [x] HMDB51 检索实验脚本和特征缓存。
- [x] 可复用的推理服务接口。
- [x] 数据库、后端 API、独立 Worker。
- [x] React 前端。
- [x] 自动化测试与应用级启动说明。

## 3. P0：工程基线

- [x] 新增根目录 `README.md`，说明应用目标、依赖和本地启动方式。
- [x] 固定 Python 依赖并记录当前可工作的 Python、PyTorch、OpenCLIP、OpenCV 版本。
- [x] 新增前端依赖锁文件。
- [x] 将运行时配置集中到环境变量和配置类：
  - [x] 数据库 URL。
  - [x] 媒体存储根目录。
  - [x] 模型缓存目录。
  - [x] 允许扫描的目录白名单。
  - [x] Worker 轮询间隔。
  - [x] 上传大小上限。
- [x] 在 `.gitignore` 中确认忽略数据库、上传媒体、推理产物、日志和前端构建产物。
- [x] 新增应用目录，但不移动当前研究数据与输出：

```text
backend/
frontend/
data/app_storage/
```

验收：

- [x] 新环境能按 README 启动 API。
- [x] 配置中不写死 `D:\ActionRecognition`。
- [x] 应用写入只发生在配置的存储目录内。

## 4. P1：推理核心模块化

目标是将 `scripts/search_image_in_video.py` 从“只能执行的脚本”拆成“可复用模块 + 薄 CLI”。

- [x] 定义 `TemporalActionLocator` 接口。
- [x] 定义强类型的 `LocalizationParams`、`VideoMetadata` 和 `DetectionResult`。
- [x] 将现有核心逻辑接入 `backend/app/inference/` 的可复用 locator：
  - [x] OpenCLIP 模型加载与 Query 编码。
  - [x] 视频采样与帧编码。
  - [x] 滑窗打分。
  - [x] Temporal IoU/NMS。
  - [x] 片段与最佳帧导出。
- [x] `localize(...)` 支持进度回调和取消检查。
- [x] Worker 进程启动时只加载一次模型。
- [x] 保留原 CLI 参数和输出结构；CLI 与 Worker 复用同一组推理核心函数。
- [x] 为关键纯函数补单元测试：
  - [x] 窗口起点生成。
  - [x] 短视频与尾部窗口。
  - [x] Temporal IoU。
  - [x] NMS 排序与抑制。
  - [x] 参数边界校验。
- [x] 使用一个现有 Query/视频做重构前后回归对比。

验收：

- [x] 相同输入和参数下，重构前后的 Top-1 区间与分数在容差内一致。
- [x] API/Worker 不通过 `subprocess` 调用检索脚本。
- [x] CLI 仍能产出 `predictions.json`、片段、最佳帧和 CSV。

## 5. P2：数据库与媒体存储

- [x] 接入 FastAPI、SQLAlchemy 2.x、Alembic 和 SQLite。
- [x] 建立首个 migration。
- [x] 实现 `media_assets`：
  - [x] 图片、视频、片段、最佳帧四种类型。
  - [x] 原文件名、存储键、MIME、大小、SHA-256。
  - [x] 视频时长、宽高、FPS。
  - [x] 来源资产关系。
  - [x] 软删除时间。
- [x] 实现 `inference_jobs`：
  - [x] Query、目标视频、模型标识。
  - [x] 状态、阶段、进度、参数快照。
  - [x] 错误摘要和错误日志路径。
  - [x] 创建、开始、心跳、结束时间。
  - [x] 取消请求时间。
- [x] 实现与任务一对一的 `localization_results`：
  - [x] 起止时间、综合/平均/最大分数。
  - [x] 最佳帧时间。
  - [x] 片段和最佳帧资产关系。
- [x] 实现本地 `StorageAdapter`，数据库只保存相对存储键。
- [x] 实现 SHA-256 去重策略。
- [x] 使用 `ffprobe` 获取视频元数据，OpenCV 作为明确的降级方案。
- [x] 生成图片/视频缩略图。
- [x] 写入文件采用临时文件后原子重命名，失败时清理临时产物。

验收：

- [x] 数据库中不保存图片或视频二进制。
- [x] 数据库迁移可在空目录重复执行。
- [x] 删除源资产前会检查是否被任务引用。
- [x] 任意存储键都不能逃逸应用存储根目录。

## 6. P3：媒体 API

- [x] `GET /api/health`：API、数据库、FFmpeg 与模型配置健康状态。
- [x] `POST /api/assets/upload`：上传图片或视频。
- [x] `POST /api/assets/scan`：扫描白名单目录中的已有文件。
- [x] `GET /api/assets`：按类型、关键字、分页查询。
- [x] `GET /api/assets/{id}`：媒体详情。
- [x] `GET /api/assets/{id}/thumbnail`：缩略图。
- [x] `GET /api/assets/{id}/content`：
  - [x] 图片响应。
  - [x] 视频 HTTP Range 请求和正确的 `Content-Type`。
- [x] `DELETE /api/assets/{id}`：MVP 采用软删除。
- [x] 校验扩展名、探测后的真实媒体类型、文件大小和损坏文件。
- [x] API 错误统一为稳定的错误码和用户可读消息。
- [x] 生成 OpenAPI schema 并加入契约检查。

验收：

- [x] Swagger 中可上传/扫描并查询图片和视频。
- [x] 浏览器可拖动播放较大的本地视频。
- [x] 重复扫描不会产生重复资产。
- [x] 非白名单路径和路径穿越请求被拒绝。

## 7. P4：任务 API 与 Worker

- [x] `POST /api/jobs`：校验资产与参数后创建 `queued` 任务。
- [x] `GET /api/jobs`：按状态分页查询。
- [x] `GET /api/jobs/{id}`：返回任务、进度和结果摘要。
- [x] `GET /api/jobs/{id}/result`：返回唯一的 Top-1 结果。
- [x] `POST /api/jobs/{id}/cancel`：请求取消排队中或运行中的任务。
- [x] `POST /api/jobs/{id}/retry`：失败任务复制参数后重试。
- [x] 独立 `python -m app.worker` 进程：
  - [x] 原子领取一个 `queued` 任务。
  - [x] 更新阶段、进度与心跳。
  - [x] 调用模块化推理接口。
  - [x] 在同一事务中保存结果并标记成功。
  - [x] 捕获异常、保存错误并继续轮询。
  - [x] 在安全检查点响应取消。
- [x] Worker 启动时恢复策略：
  - [x] 超过心跳阈值的 `running` 任务标记为 `failed`。
  - [x] 不自动重复可能已有部分产物的任务。
- [x] 暂时限制为单 Worker；代码和文档明确此约束。

验收：

- [x] 创建任务接口在短时间内返回 `202 Accepted`。
- [x] 状态只按设计状态机迁移。
- [x] API 重启不影响正在运行的独立 Worker。
- [x] Worker 异常退出后不会留下永久 `running` 的任务。
- [x] 失败任务可查看原因并可显式重试。

## 8. P5：前端 MVP

- [x] 使用 React + TypeScript + Vite。
- [x] 建立统一 API client 和由 OpenAPI 对齐的类型。
- [x] 应用框架：
  - [x] 顶部/侧边导航。
  - [x] 全局错误提示。
  - [x] 空状态、加载状态。
- [x] 媒体库页：
  - [x] 图片/视频筛选。
  - [x] 搜索、分页。
  - [x] 上传。
  - [x] 扫描已有目录。
  - [x] 缩略图与元数据预览。
- [x] 创建任务页：
  - [x] 选择 Query 图片。
  - [x] 选择目标视频。
  - [x] 视频和图片预览。
  - [x] 参数使用推荐默认值，高级参数可折叠。
  - [x] 表单边界校验与二次提交防护。
- [x] 任务列表页：
  - [x] 状态、阶段、进度、创建时间。
  - [x] 取消、重试、查看结果。
- [x] 任务详情/结果页：
  - [x] Query 图片。
  - [x] 完整视频播放器。
  - [x] 轮询任务状态。
  - [x] Top-1 时间轴标记。
  - [x] Top-1 区间、分数和最佳帧。
  - [x] 点击结果跳转并只播放对应区间。
  - [x] 直接播放或下载 Top-1 导出片段。
- [x] 组件测试覆盖时间格式化、任务状态和播放器区间控制。

验收：

- [x] Chrome/Edge 当前稳定版中完成完整用户流程。
- [x] 页面刷新后可根据 URL 重新加载任务详情。
- [x] 视频播放、跳转和区间停止误差不超过播放器可接受范围。
- [x] 后端错误不会只显示“请求失败”，而会显示可操作的信息。

## 9. P6：集成、测试与交付

- [x] 后端测试：
  - [x] SQLite 临时数据库。
  - [x] 资产 API。
  - [x] 任务状态机。
  - [x] Worker 成功、失败、取消。
  - [x] Range 请求。
- [x] 推理测试使用 fake locator，避免每次加载真实模型。
- [x] 至少保留一个真实模型的手工/慢速 smoke test。
- [x] 前端构建和静态检查。
- [x] 端到端 happy path：
  - [x] 登记资产。
  - [x] 创建任务。
  - [x] Worker 完成。
  - [x] 结果页跳转视频。
- [x] 记录 CPU 环境下一条代表性视频的耗时和内存。
- [x] 增加一键开发启动脚本，分别管理 API、Worker、前端。
- [x] README 补充备份、清理产物、常见错误与恢复方式。

MVP 发布门槛：

- [x] P0—P6 的验收项全部通过。
- [x] 没有会导致源媒体丢失的已知问题。
- [x] 没有路径穿越、任意目录扫描或任意文件读取问题。
- [x] 核心流程有自动化测试，真实模型流程有 smoke test 记录。

## 10. P1 之后再做

- [ ] SSE 实时进度；轮询在 MVP 中足够。
- [ ] Gallery 特征缓存与缓存失效策略。
- [ ] 一张 Query 检索多个视频的批量任务。
- [ ] LanguageBind/InternVideo2 模型适配器。
- [ ] GPU Worker 和显存监控。
- [ ] PostgreSQL。
- [ ] Redis/专业任务队列。
- [ ] NAS/MinIO 存储适配器。
- [ ] 局域网/公网部署时再增加用户登录、角色和审计日志；本地 MVP 不实现。
- [ ] 人工标注“正确/错误”与结果导出。
- [ ] 分数曲线可视化和实验指标页。

## 11. 明确不进入 MVP

- Docker/Kubernetes。
- 多机调度和多 Worker 并发领取。
- WebSocket。
- 将 MP4/BLOB 存入数据库。
- 模型训练或微调。
- VLM/LLM 对话。
- 实时摄像头流。
- 复杂工艺流程或动作顺序约束。

## 12. 建议执行顺序

1. [x] P0 工程基线。
2. [x] P1 推理核心模块化。
3. [x] P2 数据库与存储。
4. [x] P3 媒体 API。
5. [x] P4 任务 API 与 Worker。
6. [x] 用 Swagger 完成后端闭环 review。
7. [x] P5 前端 MVP。
8. [x] P6 集成、测试与交付。

第一个 review 节点：Swagger 中完成“选择现有 Query/视频 → 创建任务 → Worker 完成 → 返回 Top-1 结果和媒体 URL”，确认后再投入完整前端。
