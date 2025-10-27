# 高并发检索服务优化指南

## 🎯 核心问题

您遇到的内存溢出问题（OOM）是由于使用了 **10 workers**，每个 worker 独立加载模型和索引：

```
10 workers × 85GB/worker = 850GB 内存占用 ❌
```

这超过了您节点的可用内存（896GB），导致 Ray 杀死进程。

## ✅ 推荐解决方案：单 Worker + 异步并发

### 架构对比

| 方案 | 内存占用 | 并发能力 | 吞吐量 | 推荐 |
|------|---------|---------|--------|------|
| **10 workers (旧)** | ~850GB | 高 | 高 | ❌ 内存溢出 |
| **1 worker + async (新)** | ~85GB | 高 | 高 | ✅ 推荐 |
| 模型服务分离 | ~85GB | 极高 | 极高 | 🔄 复杂场景 |

### 为什么单 Worker 也能高并发？

现在的服务器已经优化为**异步架构**：

```python
# 关键优化点
1. async/await 异步处理
2. CPU 密集操作在线程池执行（不阻塞事件循环）
3. FAISS 搜索异步化
4. 模型编码异步化
```

### 性能预期

使用 **1 worker + async 模式**：

- **内存占用**: ~85GB（节省 90%）
- **并发处理**: 100+ 并发请求
- **吞吐量**: 10-50 req/s（取决于查询复杂度）
- **延迟**: 100-500ms/请求

## 🚀 快速启动

### 方案 1: 使用启动脚本（推荐）

```bash
# 单 worker 模式（默认，推荐）
bash retrieval/launch_server.sh

# 指定端口
bash retrieval/launch_server.sh /path/to/data 8080

# 开发模式（自动重载）
bash retrieval/launch_server.sh /path/to/data 8080 1 reload
```

### 方案 2: 直接运行 Python

```bash
# 单 worker（推荐）
python retrieval/server.py --workers 1 --host 0.0.0.0 --port 2727

# 如果真的需要多 worker（需确保有足够内存）
python retrieval/server.py --workers 2 --host 0.0.0.0 --port 2727
# 需要: 2 × 85GB = 170GB 可用内存
```

## 📊 性能测试

### 运行并发测试

```bash
# 安装依赖
pip install aiohttp

# 测试 100 个请求，50 并发
python retrieval/benchmark_concurrency.py --server http://localhost:2727 --requests 100 --concurrent 50

# 测试 1000 个请求，100 并发
python retrieval/benchmark_concurrency.py --requests 1000 --concurrent 100
```

### 预期结果（单 worker）

```
总请求: 1000
并发数: 100
总耗时: ~50-100秒
吞吐量: 10-20 req/s
成功率: >99%
平均延迟: 200-500ms
```

## 🔧 高级优化选项

### 1. 调整 uvicorn 参数

```bash
# 增加每个 worker 的连接数限制
python retrieval/server.py \
    --workers 1 \
    --host 0.0.0.0 \
    --port 2727 \
    --limit-concurrency 1000 \
    --limit-max-requests 10000
```

### 2. 系统级优化

```bash
# 增加文件描述符限制
ulimit -n 65536

# 调整 TCP 参数（/etc/sysctl.conf）
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
```

### 3. FAISS 优化

如果需要更高性能，考虑：

```python
# 使用 GPU FAISS（如果有 GPU）
import faiss
gpu_res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

# 使用更快的索引类型
# IVF 索引：快速但略微损失精度
# HNSW 索引：平衡速度和精度
```

## 🏗️ 方案 3: 模型服务分离（超高并发场景）

如果单 worker 仍不满足需求，可以将模型服务独立部署：

```
┌─────────────────┐
│  API Gateway    │  (多实例，轻量级)
└────────┬────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌────────┐ ┌────────┐
│ Model  │ │ Model  │  (重量级服务，1-2实例)
│Service │ │Service │
└────────┘ └────────┘
```

### 实现方案

```python
# 1. 独立的模型服务（使用 Ray Serve 或 Triton）
# 2. API 网关调用模型服务
# 3. 负载均衡

# 优点：
# - 模型实例数量可独立控制
# - 更好的资源利用
# - 支持更复杂的部署策略

# 缺点：
# - 架构更复杂
# - 网络延迟增加
```

## 📈 监控和调优

### 监控关键指标

```bash
# 1. 内存使用
watch -n 1 'ps aux | grep server.py | grep -v grep'

# 2. 连接数
netstat -an | grep :2727 | wc -l

# 3. 系统负载
htop
```

### 使用 FastAPI 内置监控

访问以下端点：
- `http://localhost:2727/docs` - API 文档
- `http://localhost:2727/health` - 健康检查
- `http://localhost:2727/openapi.json` - OpenAPI 规范

### 添加 Prometheus 监控（可选）

```python
# 安装
pip install prometheus-fastapi-instrumentator

# 在 server.py 添加
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

## 🎯 最佳实践总结

### ✅ 推荐配置

```bash
# 生产环境
python retrieval/server.py \
    --workers 1 \
    --host 0.0.0.0 \
    --port 2727 \
    --log-level info

# 内存占用: ~85GB
# 可处理: 50-100 并发请求
# 适用于: 大多数生产场景
```

### ❌ 避免的配置

```bash
# 不推荐：多 worker 模式（除非有充足内存）
python retrieval/server.py --workers 10  # 需要 850GB 内存！

# 不推荐：同步模式（已废弃）
# 旧版本使用同步搜索，性能差
```

### 🔄 何时需要多 Worker

仅在以下情况考虑多 worker：

1. **充足内存**: 确保有 `workers × 85GB` 可用内存
2. **CPU 密集**: 任务主要是 CPU 计算（不是 I/O）
3. **已验证瓶颈**: 通过测试确认单 worker 是瓶颈
4. **监控到位**: 有完善的内存监控和告警

```bash
# 2 workers 配置（需要 170GB 可用内存）
python retrieval/server.py --workers 2 --host 0.0.0.0

# 使用前先测试内存
free -h
# 确保 available memory > 170GB
```

## 🐛 故障排查

### 问题 1: 仍然 OOM

**检查**:
```bash
# 查看实际 worker 数量
ps aux | grep "server.py" | grep -v grep

# 查看内存占用
ps aux | grep python | awk '{sum+=$6} END {print sum/1024/1024 " GB"}'
```

**解决**:
- 确认只启动了 1 个 server 实例
- 检查其他 Python 进程是否占用内存
- 考虑重启服务器清理内存

### 问题 2: 性能不佳

**检查**:
```bash
# 运行性能测试
python retrieval/benchmark_concurrency.py --concurrent 50

# 检查 CPU 使用率
htop
```

**优化**:
- 如果 CPU 满载：考虑增加 worker（需要更多内存）
- 如果 CPU 不满：检查网络/磁盘瓶颈
- 优化查询：减少 top_k 参数

### 问题 3: 连接被拒绝

**检查**:
```bash
# 查看最大连接数
ulimit -n

# 查看当前连接
netstat -an | grep :2727 | grep ESTABLISHED | wc -l
```

**解决**:
```bash
# 增加文件描述符限制
ulimit -n 65536

# 或在 /etc/security/limits.conf 永久设置
* soft nofile 65536
* hard nofile 65536
```

## 📞 获取帮助

如果遇到问题：

1. 查看日志: 服务器会输出详细的启动和运行日志
2. 运行测试: 使用 `benchmark_concurrency.py` 验证性能
3. 检查健康: 访问 `/health` 端点确认服务状态

## 🎓 技术细节

### 异步架构原理

```python
# 传统同步模式（阻塞）
def search(query):
    vector = model.encode(query)      # 阻塞 100ms
    results = index.search(vector)     # 阻塞 50ms
    return results
# 总延迟: 150ms，其间无法处理其他请求

# 新异步模式（非阻塞）
async def search_async(query):
    # 在线程池执行，不阻塞事件循环
    vector = await run_in_executor(model.encode, query)
    results = await run_in_executor(index.search, vector)
    return results
# 总延迟: 150ms，但期间可以处理其他请求！
```

### 并发处理示例

```
时间轴:
0ms:    请求1开始 → 提交到线程池
10ms:   请求2开始 → 提交到线程池
20ms:   请求3开始 → 提交到线程池
...
150ms:  请求1完成
160ms:  请求2完成
170ms:  请求3完成

单 worker 在 170ms 内处理了 3 个请求！
如果是同步模式，需要 450ms（150ms × 3）
```

这就是为什么单 worker 也能高并发的原因。

