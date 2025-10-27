# 快速启动指南 - 高并发检索服务

## 🚀 30 秒快速启动

```bash
# 单 worker 高并发模式（推荐）
bash retrieval/launch_server.sh

# 服务地址: http://0.0.0.0:2727
# API 文档: http://0.0.0.0:2727/docs
# 健康检查: http://0.0.0.0:2727/health
```

**内存占用**: ~85GB  
**并发能力**: 100+ 并发请求  
**吞吐量**: 10-50 req/s

## ⚠️ 重要：避免内存溢出

### ❌ 不要这样做

```bash
# 这会导致 OOM！
bash retrieval/launch_server.sh /path/to/data 2727 10
# 10 workers × 85GB = 850GB 内存！
```

### ✅ 正确做法

```bash
# 单 worker 异步模式（默认）
bash retrieval/launch_server.sh

# 或明确指定
bash retrieval/launch_server.sh /path/to/data 2727 1
```

## 📊 性能测试

```bash
# 安装测试依赖
pip install aiohttp

# 运行性能测试
python retrieval/benchmark_concurrency.py \
    --server http://localhost:2727 \
    --requests 100 \
    --concurrent 50
```

## 🔧 常用配置

### 生产环境

```bash
python retrieval/server.py \
    --workers 1 \
    --host 0.0.0.0 \
    --port 2727 \
    --log-level info
```

### 开发环境（自动重载）

```bash
python retrieval/server.py \
    --workers 1 \
    --host 127.0.0.1 \
    --port 2727 \
    --reload \
    --log-level debug
```

### 高并发场景

```bash
# 确保系统限制足够
ulimit -n 65536

# 启动服务
python retrieval/server.py \
    --workers 1 \
    --host 0.0.0.0 \
    --port 2727
```

## 🎯 API 使用示例

### Python 客户端

```python
import requests

# 同步请求
response = requests.post(
    "http://localhost:2727/retrieve",
    json={"query": "What is machine learning?", "top_k": 10}
)
results = response.json()

# 异步请求（推荐）
import aiohttp
import asyncio

async def search(query: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:2727/retrieve",
            json={"query": query, "top_k": 10}
        ) as response:
            return await response.json()

# 批量并发请求
queries = ["query1", "query2", "query3"]
results = await asyncio.gather(*[search(q) for q in queries])
```

### cURL

```bash
# 健康检查
curl http://localhost:2727/health

# 检索请求
curl -X POST http://localhost:2727/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 10}'
```

## 📈 监控

### 检查服务状态

```bash
# 健康检查
curl http://localhost:2727/health | jq

# 查看进程
ps aux | grep "server.py"

# 查看内存占用
ps aux | grep "server.py" | awk '{print $6/1024/1024 " GB"}'
```

### 实时监控

```bash
# 监控内存
watch -n 1 'ps aux | grep server.py | grep -v grep'

# 监控连接数
watch -n 1 'netstat -an | grep :2727 | grep ESTABLISHED | wc -l'
```

## 🐛 问题排查

### 服务无法启动

```bash
# 检查端口占用
netstat -tlnp | grep 2727

# 查看错误日志
# 日志会输出到终端

# 检查数据文件
ls -lh /mnt/public/sunjinfeng/data/search_data/prebuilt_indices/
```

### 内存溢出

```bash
# 确认只有 1 个 worker
ps aux | grep "server.py" | wc -l

# 应该只有 2 行（1 个主进程 + 1 个 worker）

# 如果有多个进程，杀掉重启
pkill -f "server.py"
bash retrieval/launch_server.sh
```

### 性能不佳

```bash
# 运行性能测试
python retrieval/benchmark_concurrency.py --concurrent 50

# 检查 CPU 使用率
htop

# 检查网络延迟
ping your_server_ip
```

## 📚 更多信息

- **完整文档**: `HIGH_CONCURRENCY_GUIDE.md`
- **API 文档**: http://localhost:2727/docs
- **性能测试**: `benchmark_concurrency.py`

## 💡 关键要点

1. ✅ **默认使用 1 worker** - 已优化异步高并发
2. ✅ **内存占用 ~85GB** - 不是 850GB
3. ✅ **可处理 100+ 并发** - 单 worker 足够
4. ❌ **避免多 worker** - 除非有充足内存和明确需求
5. ⚡ **使用异步客户端** - 获得最佳性能

## 🎓 技术原理

```
传统多 worker 模式:
Worker 1: [Model 85GB] [Index 85GB] 
Worker 2: [Model 85GB] [Index 85GB]
...
总计: N × 85GB ❌

优化后异步模式:
Worker 1: [Model 85GB] [Index 85GB] 
          ↓
    [Async Event Loop]
          ↓
    [Thread Pool for CPU tasks]
          ↓
    处理 100+ 并发请求 ✅
总计: 85GB
```

**关键**: 通过异步 I/O 和线程池，单 worker 可以在等待 CPU 任务时处理其他请求，实现高并发。

