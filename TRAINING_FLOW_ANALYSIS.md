
## 概述
本文档详细解析当执行 `train_search_agent.sh` 脚本时，RLLM框架如何处理一条训练数据的完整流程，包括数据传递、模型推理、工具调用、奖励计算和参数更新的每一个细节。

## 示例训练数据
```
问题ID: 0
问题: Which magazine was started first Arthur's Magazine or First for Women?
答案: Arthur's Magazine
数据源: hotpotqa
```

---

## 一、初始化阶段

### 1.1 脚本配置解析 (`train_search_agent.sh`)

```bash
# 关键环境变量
export RETRIEVAL_SERVER_URL="http://127.0.0.1:2727"  # 检索服务器地址
export CUDA_VISIBLE_DEVICES=3,4                       # 使用GPU 3和4
export WANDB_API_KEY=...                              # W&B日志记录

# 核心训练参数
- data.train_batch_size=64                            # 训练批次大小
- data.max_prompt_length=2048                         # 最大提示词长度
- data.max_response_length=2048                       # 最大响应长度
- actor_rollout_ref.rollout.n=8                       # 每个问题采样8个回复
- rllm.agent.max_steps=20                             # 最大推理步数
- trainer.total_epochs=15                             # 训练15个epoch
```

### 1.2 训练器初始化 (`train_search_agent.py`)

```python
# 1. 加载数据集
train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
# 从 /root/github_project/rllm/rllm/data/datasets/hotpotqa/train.parquet 加载

# 2. 配置工具映射
tool_map = {"local_search": LocalRetrievalTool}
# LocalRetrievalTool 连接到 RETRIEVAL_SERVER_URL 的检索服务

# 3. 配置环境参数
env_args = {
    "max_steps": 20,                    # 最多20步
    "tool_map": tool_map,               # 工具映射
    "reward_fn": search_reward_fn,      # 奖励函数
}

# 4. 配置Agent参数
agent_args = {
    "system_prompt": SEARCH_SYSTEM_PROMPT,  # 系统提示词
    "tool_map": tool_map,                   # 工具映射
    "parser_name": "qwen"                   # 使用Qwen格式解析器
}
```

### 1.3 系统提示词 (`SEARCH_SYSTEM_PROMPT`)

```
You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \boxed{} format

For example:
- If the answer is "American", write: \boxed{American}
- If the answer is "yes", write: \boxed{yes}
- If the answer is a year like "1985", write: \boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \boxed{} format.
```

### 1.4 工具定义 (`LocalRetrievalTool`)

工具会被转换成以下JSON格式供模型调用：

```json
{
  "type": "function",
  "function": {
    "name": "local_search",
    "description": "Search for information using a dense retrieval server with Wikipedia corpus",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search query to retrieve relevant documents"
        },
        "top_k": {
          "type": "integer",
          "description": "Number of results to return (default: 10)",
          "minimum": 1,
          "maximum": 50
        }
      },
      "required": ["query"]
    }
  }
}
```

---

## 二、数据加载与预处理

### 2.1 数据集格式

原始数据存储在 Parquet 文件中：
```python
{
    "id": 0,
    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
    "answer": "Arthur's Magazine",
    "data_source": "hotpotqa"
}
```

### 2.2 Verl 格式转换

数据会被转换为 Verl 训练框架所需的格式：

```python
{
    "prompt": [{"role": "user", "content": "placeholder"}],
    "reward_model": {
        "style": "rule",
        "ground_truth": None,
    },
    "extra_info": {
        "id": 0,
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "data_source": "hotpotqa"
    }
}
```

### 2.3 批次组织

```python
# 训练批次大小 = 64
# 每个问题采样 n=8 个回复
# 实际处理的序列数 = 64 × 8 = 512 条轨迹

batch = DataProto.from_single_dict(batch_dict)
# 添加唯一ID
batch.non_tensor_batch["uid"] = [uuid.uuid4() for _ in range(64)]
# 重复以支持多次采样
batch = batch.repeat(repeat_times=8, interleave=True)
```

---

## 三、Agent-Environment 交互循环

这是最核心的部分！让我们以示例数据为例，详细追踪整个交互过程。

### 3.1 环境初始化

```python
# 为每个批次项创建环境
env = ToolEnvironment(
    task={
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "data_source": "hotpotqa"
    },
    tool_map={"local_search": LocalRetrievalTool},
    reward_fn=search_reward_fn,
    max_steps=20
)

# 为每个批次项创建Agent
agent = ToolAgent(
    system_prompt=SEARCH_SYSTEM_PROMPT,
    tool_map={"local_search": LocalRetrievalTool},
    parser_name="qwen"
)
```

### 3.2 并发与批次规模

在详细讲解交互循环前，先明确并发和批次的规模：

```python
# 批次配置
train_batch_size = 64              # 每个批次64个不同的问题
rollout.n = 8                      # 每个问题采样8次
total_trajectories = 64 × 8 = 512  # 总共512条轨迹需要生成

# 并发配置
# 1. Agent执行层面：512条轨迹并发执行
#    使用 AsyncAgentExecutionEngine 和 asyncio.gather
n_parallel_agents = 512  # 同时运行512个Agent-Environment对

# 2. GPU配置
n_gpus = 2  # 使用2个GPU (CUDA_VISIBLE_DEVICES=3,4)

# 3. 环境执行层面
max_env_workers = 64  # 最多64个线程并发执行环境操作（工具调用）

# 实际运行时：
# - 512个Agent实例被创建
# - 512个Environment实例被创建
# - 所有轨迹异步并发执行
# - 工具调用通过线程池并发（最多64线程）
# - 模型推理通过vLLM批处理（动态批次大小）
```

### 3.3 初始状态

现在让我们追踪**单条轨迹**的完整过程（从512条中选一条）：

```python
# ========== 环境初始化 ==========
env = ToolEnvironment(
    task={
        "id": 0,
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "data_source": "hotpotqa"
    },
    tool_map={"local_search": LocalRetrievalTool},
    reward_fn=search_reward_fn,
    max_steps=20
)
env.step_count = 0

# ========== Agent初始化 ==========
agent = ToolAgent(
    system_prompt=SEARCH_SYSTEM_PROMPT,
    tool_map={"local_search": LocalRetrievalTool},
    parser_name="qwen"
)

# agent内部结构：
# - agent.messages: list[dict]  # 对话历史
# - agent._trajectory: Trajectory  # 轨迹记录
# - agent.tools: MultiTool  # 工具管理器
# - agent.tool_parser: QwenToolParser  # 解析器

# ========== 环境重置 ==========
observation, info = env.reset()

# 返回值：
observation = {
    "question": "Which magazine was started first Arthur's Magazine or First for Women?"
}
info = {}

# ========== Agent重置 ==========
agent.reset()

# 重置后的agent.messages：
agent.messages = [
    {
        "role": "system", 
        "content": """You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \\boxed{} format

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \\boxed{} format.

Available tools:
{
  "type": "function",
  "function": {
    "name": "local_search",
    "description": "Search for information using a dense retrieval server with Wikipedia corpus",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query to retrieve relevant documents"},
        "top_k": {"type": "integer", "description": "Number of results to return (default: 10)"}
      },
      "required": ["query"]
    }
  }
}"""
    }
]

# 重置后的agent._trajectory：
agent._trajectory = Trajectory(
    uid="abc123-def456-...",  # 唯一ID
    name="agent",
    task=None,
    steps=[],  # 空的步骤列表
    reward=0.0,
    info={}
)

# ========== Agent处理初始观察 ==========
agent.update_from_env(
    observation=observation,
    reward=0.0,
    done=False,
    info=info
)

# update_from_env做了什么？
# 1. 格式化observation为消息
obs_messages = agent._format_observation_as_messages(observation)
# obs_messages = [
#     {"role": "user", "content": "Which magazine was started first Arthur's Magazine or First for Women?"}
# ]

# 2. 扩展消息历史
agent.messages.extend(obs_messages)

# 3. 保存当前观察
agent.current_observation = observation

# 4. 如果有之前的步骤，更新其reward和done
if agent._trajectory.steps:
    agent._trajectory.steps[-1].reward = 0.0
    agent._trajectory.steps[-1].done = False
    agent._trajectory.steps[-1].info = info

# 现在agent.messages包含：
agent.messages = [
    {"role": "system", "content": "...系统提示 + 工具定义..."},
    {"role": "user", "content": "Which magazine was started first Arthur's Magazine or First for Women?"}
]

# 注意：agent.messages是所有信息流转的核心！
# - 它记录了完整的对话历史
# - 每次调用模型时，都会传递这个messages
# - 模型的响应和环境的反馈都会追加到这里
# - Trajectory中的每个Step也会保存messages的快照
```

**关于 `update_from_env` 的详细说明：**

`update_from_env` 是Agent的核心方法之一，它的作用包括：

1. **更新对话历史（agent.messages）**：
   - 将环境的observation格式化为消息
   - 追加到messages列表中
   - messages是与模型交互的唯一接口

2. **更新Trajectory**：
   - 如果已有步骤，更新上一步的reward、done、info
   - Trajectory记录完整的交互历史，用于后续分析

3. **保存当前状态**：
   - current_observation：当前环境观察
   - 为下一次模型调用做准备

**messages vs Trajectory的区别：**
- **messages**：实时对话历史，用于与模型交互
- **Trajectory**：结构化记录，包含每一步的完整信息（observation, action, reward等）

### 3.4 推理循环 (最多20步)

#### **第1步：模型生成工具调用**

```python
# ========== 1. 构造提示词 ==========
prompt_messages = agent.chat_completions  # 获取当前对话历史
# prompt_messages = [
#   {"role": "system", "content": "...system prompt + tools..."},
#   {"role": "user", "content": "Which magazine was started first Arthur's Magazine or First for Women?"}
# ]

# ========== 2. 通过Verl引擎调用模型 ==========
# 这是异步调用，512条轨迹并发进行
response = await rollout_engine.get_model_response(
    prompt_messages,
    application_id="trajectory_001",  # 轨迹唯一标识
    max_tokens=2048,
    temperature=0.7
)

# 内部流程（vLLM引擎）：
# 1. 将prompt_messages转换为token序列
# 2. 添加到vLLM的请求队列
# 3. vLLM动态批处理多个请求
# 4. GPU执行推理
# 5. 返回生成的文本

# ========== 3. 模型生成的响应（完整示例）==========
response = """I need to search for information about these two magazines to determine which was started first. Let me search for Arthur's Magazine first.

<tool_call>
{"name": "local_search", "arguments": {"query": "Arthur's Magazine history founding date publication", "top_k": 5}}
</tool_call>"""

# 注意：
# - 模型会用自然语言解释其思考过程
# - 然后用<tool_call>标签包裹工具调用
# - 工具调用是JSON格式
# - Qwen解析器会提取这些信息
```

#### **第2步：解析工具调用**

```python
# ========== Agent更新状态 ==========
action = agent.update_from_model(response)

# update_from_model内部流程：

# 1. 使用Qwen解析器提取工具调用
tool_calls = agent.tool_parser.parse(response)
# 输入: """I need to search...<tool_call>{"name": "local_search", ...}</tool_call>"""
# 输出: [
#   ToolCall(
#     name="local_search",
#     arguments={
#       "query": "Arthur's Magazine history founding date publication",
#       "top_k": 5
#     }
#   )
# ]

# 2. 为每个工具调用生成唯一ID
tool_calls_dict = [
    {
        "id": "call_a1b2c3d4",  # 生成UUID
        "type": "function",
        "function": {
            "name": "local_search",
            "arguments": '{"query": "Arthur\'s Magazine history founding date publication", "top_k": 5}'
        }
    }
]

# 3. 更新agent.messages
assistant_message = {
    "role": "assistant",
    "content": """I need to search for information about these two magazines to determine which was started first. Let me search for Arthur's Magazine first.

<tool_call>
{"name": "local_search", "arguments": {"query": "Arthur's Magazine history founding date publication", "top_k": 5}}
</tool_call>"""
}
agent.messages.append(assistant_message)

# 现在agent.messages = [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "Which magazine was started first..."},
#   {"role": "assistant", "content": "I need to search...<tool_call>..."}
# ]

# 4. 创建新的Step并添加到Trajectory
new_step = Step(
    chat_completions=copy.deepcopy(agent.messages),  # 保存当前对话历史快照
    action=tool_calls_dict,  # 工具调用
    model_response=response,  # 原始模型输出
    observation=agent.current_observation,  # 当前观察
    reward=0.0,  # 初始化为0
    done=False,
    info={}
)
agent._trajectory.steps.append(new_step)

# 现在agent._trajectory.steps = [Step1]

# 5. 返回Action
action = Action(action=tool_calls_dict)
# action.action = [
#   {
#     "id": "call_a1b2c3d4",
#     "type": "function",
#     "function": {...}
#   }
# ]
```

#### **第3步：执行工具调用**

```python
# ========== 环境执行工具调用 ==========
next_observation, reward, done, info = env.step(action.action)

# env.step内部流程：

# 1. 增加步数计数
env.step_count += 1  # 现在 = 1

# 2. 检查是否是finish工具调用
done = False  # 不是finish，继续
for tool_call in action.action:
    if tool_call.get("function", {}).get("name") == "finish":
        done = True
        break

# 3. 并行执行所有工具调用（使用线程池）
tool_outputs = {}

# 对于我们的例子，只有一个工具调用
tool_call = action.action[0]
# {
#   "id": "call_a1b2c3d4",
#   "type": "function",
#   "function": {
#     "name": "local_search",
#     "arguments": '{"query": "Arthur\'s Magazine history founding date publication", "top_k": 5}'
#   }
# }

tool_name = tool_call["function"]["name"]  # "local_search"
tool_args = json.loads(tool_call["function"]["arguments"])
# {"query": "Arthur's Magazine history founding date publication", "top_k": 5}

# 4. 调用LocalRetrievalTool
tool_instance = env.tools(tool_name=tool_name, **tool_args)
# 等价于：
# tool = LocalRetrievalTool()
# output = tool.forward(query="Arthur's Magazine...", top_k=5)

# LocalRetrievalTool.forward内部流程：
# a. 构造HTTP请求
payload = {
    "query": "Arthur's Magazine history founding date publication",
    "top_k": 5
}

# b. 发送到检索服务器
# POST http://127.0.0.1:2727/retrieve
response = requests.post("http://127.0.0.1:2727/retrieve", json=payload)

# c. 检索服务器返回结果（向量检索）
response_data = {
    "results": [
        {
            "id": "wiki_arthur_1844",
            "content": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Founded by Timothy Shay Arthur, it featured fiction, essays, and poetry. The magazine was published monthly from 1844 to 1846.",
            "score": 0.892,
            "metadata": {"title": "Arthur's Magazine", "source": "wikipedia"}
        },
        {
            "id": "wiki_arthur_timothy",
            "content": "Timothy Shay Arthur (June 6, 1809 – March 6, 1885) was an American author and editor. He is best known for his temperance novel Ten Nights in a Bar-Room. In 1844, he founded Arthur's Magazine, also known as Arthur's Home Magazine, which he edited until 1846.",
            "score": 0.854,
            "metadata": {"title": "Timothy Shay Arthur", "source": "wikipedia"}
        },
        {
            "id": "wiki_periodicals_1840s",
            "content": "During the 1840s, numerous literary magazines emerged in the United States. Arthur's Magazine, founded in 1844, was one of several Philadelphia-based periodicals that sought to provide moral and educational content to middle-class readers.",
            "score": 0.821,
            "metadata": {"title": "American periodicals", "source": "wikipedia"}
        },
        {
            "id": "wiki_arthur_home_mag",
            "content": "Arthur's Home Magazine was the later name for Arthur's Magazine after it was revived. The original Arthur's Magazine ran from 1844 to 1846 under Timothy Shay Arthur's editorship.",
            "score": 0.798,
            "metadata": {"title": "Arthur's Home Magazine", "source": "wikipedia"}
        },
        {
            "id": "wiki_philly_lit_scene",
            "content": "Philadelphia had a thriving literary scene in the mid-19th century. Magazines such as Graham's Magazine, Godey's Lady's Book, and Arthur's Magazine competed for readers interested in fiction, poetry, and moral instruction.",
            "score": 0.776,
            "metadata": {"title": "Philadelphia literary scene", "source": "wikipedia"}
        }
    ]
}

# d. 格式化输出
formatted_output = """[Document 1] (ID: wiki_arthur_1844, Score: 0.892)
Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Founded by Timothy Shay Arthur, it featured fiction, essays, and poetry. The magazine was published monthly from 1844 to 1846.

[Document 2] (ID: wiki_arthur_timothy, Score: 0.854)
Timothy Shay Arthur (June 6, 1809 – March 6, 1885) was an American author and editor. He is best known for his temperance novel Ten Nights in a Bar-Room. In 1844, he founded Arthur's Magazine, also known as Arthur's Home Magazine, which he edited until 1846.

[Document 3] (ID: wiki_periodicals_1840s, Score: 0.821)
During the 1840s, numerous literary magazines emerged in the United States. Arthur's Magazine, founded in 1844, was one of several Philadelphia-based periodicals that sought to provide moral and educational content to middle-class readers.

[Document 4] (ID: wiki_arthur_home_mag, Score: 0.798)
Arthur's Home Magazine was the later name for Arthur's Magazine after it was revived. The original Arthur's Magazine ran from 1844 to 1846 under Timothy Shay Arthur's editorship.

[Document 5] (ID: wiki_philly_lit_scene, Score: 0.776)
Philadelphia had a thriving literary scene in the mid-19th century. Magazines such as Graham's Magazine, Godey's Lady's Book, and Arthur's Magazine competed for readers interested in fiction, poetry, and moral instruction."""

# e. 创建ToolOutput对象
tool_output = ToolOutput(
    name="local_search",
    output=formatted_output,
    metadata={
        "query": "Arthur's Magazine history founding date publication",
        "num_results": 5,
        "retriever_type": "dense",
        "server_url": "http://127.0.0.1:2727"
    }
)

# 5. 保存工具输出
tool_output_str = tool_output.to_string()
# to_string() 返回：
# """Tool: local_search
# Output: [Document 1] (ID: wiki_arthur_1844, Score: 0.892)
# Arthur's Magazine (1844–1846) was an American literary periodical...
# ...
# """

tool_outputs[tool_call["id"]] = tool_output_str
# tool_outputs = {
#   "call_a1b2c3d4": "Tool: local_search\nOutput: [Document 1]...\n..."
# }

# 6. 返回观察
next_observation = {"tool_outputs": tool_outputs}
reward = 0  # 中间步骤没有奖励
done = False  # 未结束
info = {"response": action.action, "metadata": {}}

# 返回
return next_observation, reward, done, info
```

#### **第4步：Agent处理工具输出**

```python
# ========== Agent更新状态 ==========
agent.update_from_env(
    observation=next_observation,
    reward=0,
    done=False,
    info=info
)

# update_from_env内部：

# 1. 格式化observation为消息
obs_messages = agent._format_observation_as_messages(next_observation)
# next_observation = {"tool_outputs": {"call_a1b2c3d4": "Tool: local_search\n..."}}

# 对于tool_outputs类型的observation：
obs_messages = [
    {
        "role": "tool",
        "content": """Tool: local_search
Output: [Document 1] (ID: wiki_arthur_1844, Score: 0.892)
Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Founded by Timothy Shay Arthur, it featured fiction, essays, and poetry. The magazine was published monthly from 1844 to 1846.

[Document 2] (ID: wiki_arthur_timothy, Score: 0.854)
Timothy Shay Arthur (June 6, 1809 – March 6, 1885) was an American author and editor. He is best known for his temperance novel Ten Nights in a Bar-Room. In 1844, he founded Arthur's Magazine, also known as Arthur's Home Magazine, which he edited until 1846.

[Document 3] (ID: wiki_periodicals_1840s, Score: 0.821)
During the 1840s, numerous literary magazines emerged in the United States. Arthur's Magazine, founded in 1844, was one of several Philadelphia-based periodicals that sought to provide moral and educational content to middle-class readers.

[Document 4] (ID: wiki_arthur_home_mag, Score: 0.798)
Arthur's Home Magazine was the later name for Arthur's Magazine after it was revived. The original Arthur's Magazine ran from 1844 to 1846 under Timothy Shay Arthur's editorship.

[Document 5] (ID: wiki_philly_lit_scene, Score: 0.776)
Philadelphia had a thriving literary scene in the mid-19th century. Magazines such as Graham's Magazine, Godey's Lady's Book, and Arthur's Magazine competed for readers interested in fiction, poetry, and moral instruction.""",
        "tool_call_id": "call_a1b2c3d4"
    }
]

# 2. 扩展消息历史
agent.messages.extend(obs_messages)

# 3. 更新上一步的reward和done
agent._trajectory.steps[-1].reward = 0
agent._trajectory.steps[-1].done = False
agent._trajectory.steps[-1].info = info

# 现在agent.messages = [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "Which magazine..."},
#   {"role": "assistant", "content": "I need to search...<tool_call>..."},
#   {"role": "tool", "content": "Tool: local_search\nOutput: [Document 1]...", "tool_call_id": "call_a1b2c3d4"}
# ]
```

#### **第5步：模型继续推理（第2次调用）**

```python
# ========== 模型基于检索结果继续推理 ==========
prompt_messages = agent.chat_completions
# 现在包含4条消息：system, user, assistant(tool_call), tool(results)

response = await rollout_engine.get_model_response(
    prompt_messages,
    application_id="trajectory_001",
    max_tokens=2048,
    temperature=0.7
)

# 模型生成：
response = """Great! I found that Arthur's Magazine was founded in 1844. Now let me search for information about First for Women magazine.

<tool_call>
{"name": "local_search", "arguments": {"query": "First for Women magazine founding date history", "top_k": 5}}
</tool_call>"""
```

#### **第6步-第8步：重复工具调用流程**

```python
# ========== 第6步：解析第二个工具调用 ==========
action = agent.update_from_model(response)
# 添加assistant消息，创建Step2，生成新的tool_call

# ========== 第7步：执行第二个工具调用 ==========
next_observation, reward, done, info = env.step(action.action)
env.step_count = 2  # 现在是第2步

# 检索服务器返回（完整示例）：
retrieval_results = {
    "results": [
        {
            "id": "wiki_ffw_magazine",
            "content": "First for Women is a woman's magazine published by Bauer Media Group in the United States. The magazine was first published in 1989 and has a circulation of 1.3 million readers. It features articles on fashion, health, beauty, food, and lifestyle topics targeted at women.",
            "score": 0.913,
            "metadata": {"title": "First for Women", "source": "wikipedia"}
        },
        {
            "id": "wiki_bauer_media",
            "content": "Bauer Media Group is a German media company headquartered in Hamburg. The company publishes over 600 magazines worldwide, including First for Women, which was launched in the United States in 1989.",
            "score": 0.867,
            "metadata": {"title": "Bauer Media Group", "source": "wikipedia"}
        },
        {
            "id": "wiki_womens_magazines_1980s",
            "content": "The 1980s saw the launch of several successful women's magazines. First for Women debuted in 1989, focusing on practical advice and real-life stories for American women. The magazine quickly gained popularity for its relatable content.",
            "score": 0.834,
            "metadata": {"title": "Women's magazines 1980s", "source": "wikipedia"}
        },
        {
            "id": "wiki_magazine_circulation",
            "content": "First for Women maintains a strong circulation in the US market. Since its founding in 1989, it has become one of the leading women's lifestyle magazines with weekly issues covering health, food, and entertainment.",
            "score": 0.801,
            "metadata": {"title": "Magazine circulation", "source": "wikipedia"}
        },
        {
            "id": "wiki_lifestyle_magazines",
            "content": "Lifestyle magazines targeting women have evolved significantly. First for Women, launched in 1989, represents a shift toward accessible, practical content compared to earlier women's magazines from the 19th century like Arthur's Magazine or Godey's Lady's Book.",
            "score": 0.778,
            "metadata": {"title": "Lifestyle magazines history", "source": "wikipedia"}
        }
    ]
}

# 格式化后的工具输出类似第一次

# ========== 第8步：Agent处理第二次工具输出 ==========
agent.update_from_env(next_observation, reward=0, done=False, info=info)

# agent.messages现在有6条：
# [system, user, assistant1, tool1, assistant2, tool2]
```

#### **第9步：模型生成最终答案**

```python
# ========== 模型第3次调用 ==========
prompt_messages = agent.chat_completions  # 6条消息
response = await rollout_engine.get_model_response(prompt_messages, ...)

# 模型生成最终答案：
response = """Perfect! Now I have all the information I need:

From my search results:
- Arthur's Magazine was founded in 1844 by Timothy Shay Arthur in Philadelphia
- First for Women was first published in 1989 by Bauer Media Group

Comparing these dates: 1844 comes before 1989, so Arthur's Magazine was started first, approximately 145 years earlier.

<tool_call>
{"name": "finish", "arguments": {"response": "\\boxed{Arthur's Magazine}"}}
</tool_call>"""

# ========== 解析finish调用 ==========
action = agent.update_from_model(response)

# action.action = [
#   {
#     "id": "call_xyz789",
#     "type": "function",
#     "function": {
#       "name": "finish",
#       "arguments": '{"response": "\\\\boxed{Arthur\'s Magazine}"}'
#     }
#   }
# ]

# 添加到messages，创建Step3
# agent.messages = [system, user, assistant1, tool1, assistant2, tool2, assistant3]
# agent._trajectory.steps = [Step1, Step2, Step3]
```

#### **第10步：环境终止并计算奖励**

```python
# ========== 环境检测finish ==========
next_observation, reward, done, info = env.step(action.action)

# env.step内部：
env.step_count = 3

# 检测到finish工具调用
done = True
for tool_call in action.action:
    if tool_call.get("function", {}).get("name") == "finish":
        done = True
        # 提取finish的arguments
        arguments = json.loads(tool_call["function"]["arguments"])
        llm_response = arguments.get("response", "")
        break

# llm_response = "\\boxed{Arthur's Magazine}"

# ========== 调用奖励函数 ==========
task_info = env.task
# {
#   "id": 0,
#   "question": "Which magazine was started first Arthur's Magazine or First for Women?",
#   "answer": "Arthur's Magazine",
#   "data_source": "hotpotqa"
# }

reward_output = reward_fn(task_info=task_info, action=llm_response)

# reward_fn内部（详见第四章）：
# 1. 提取答案：unbox("\\boxed{Arthur's Magazine}") -> "Arthur's Magazine"
# 2. 标准化：normalize("Arthur's Magazine") -> "arthurs magazine"
# 3. 计算EM：normalize(pred) == normalize(gt) -> True
# 4. 计算F1：1.0
# 5. 判断正确：is_correct = True
# 6. 赋予奖励：reward = 1.0

reward_output = RewardOutput(
    reward=1.0,
    is_correct=True,
    metadata={
        "extracted_answer": "Arthur's Magazine",
        "ground_truths": ["Arthur's Magazine"],
        "evaluation_method": "exact_match",
        "best_match": "Arthur's Magazine",
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "exact_match": True,
        "f1_threshold": 0.3
    }
)

# 返回
reward = reward_output.reward  # 1.0
info = {
    "response": action.action,
    "metadata": reward_output.metadata,
    "is_correct": reward_output.is_correct
}

return {}, reward, done, info

# ========== Agent最终更新 ==========
agent.update_from_env(observation={}, reward=1.0, done=True, info=info)

# 更新Step3的reward和done
agent._trajectory.steps[-1].reward = 1.0
agent._trajectory.steps[-1].done = True
agent._trajectory.steps[-1].info = info

# 计算轨迹总奖励
agent._trajectory.reward = sum(step.reward for step in agent._trajectory.steps)
# = 0 + 0 + 1.0 = 1.0
```

### 3.5 完整Trajectory数据结构

完成推理后，轨迹包含完整的交互历史：

```python
trajectory = Trajectory(
    uid="abc123-def456-...",
    name="agent",
    task={
        "id": 0,
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "data_source": "hotpotqa"
    },
    reward=1.0,  # 总奖励
    info={},
    steps=[
        # ========== Step 1: 第一次搜索 ==========
        Step(
            chat_completions=[
                {"role": "system", "content": "...系统提示+工具定义..."},
                {"role": "user", "content": "Which magazine was started first..."},
                {"role": "assistant", "content": "I need to search...<tool_call>{Arthur's Magazine}</tool_call>"}
            ],
            observation={"question": "Which magazine was started first..."},
            action=[{
                "id": "call_a1b2c3d4",
                "type": "function",
                "function": {
                    "name": "local_search",
                    "arguments": '{"query": "Arthur\'s Magazine history founding date publication", "top_k": 5}'
                }
            }],
            model_response="I need to search...<tool_call>...</tool_call>",
            reward=0.0,
            done=False,
            info={}
        ),
        
        # ========== Step 2: 第二次搜索 ==========
        Step(
            chat_completions=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": "Which magazine was started first..."},
                {"role": "assistant", "content": "I need to search...<tool_call>...</tool_call>"},
                {"role": "tool", "content": "Tool: local_search\nOutput: [Document 1]...[Document 5]...", "tool_call_id": "call_a1b2c3d4"},
                {"role": "assistant", "content": "Great! I found...<tool_call>{First for Women}</tool_call>"}
            ],
            observation={"tool_outputs": {"call_a1b2c3d4": "Tool: local_search\n..."}},
            action=[{
                "id": "call_e5f6g7h8",
                "type": "function",
                "function": {
                    "name": "local_search",
                    "arguments": '{"query": "First for Women magazine founding date history", "top_k": 5}'
                }
            }],
            model_response="Great! I found...<tool_call>...</tool_call>",
            reward=0.0,
            done=False,
            info={}
        ),
        
        # ========== Step 3: 给出最终答案 ==========
        Step(
            chat_completions=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": "Which magazine was started first..."},
                {"role": "assistant", "content": "I need to search...<tool_call>...</tool_call>"},
                {"role": "tool", "content": "...[Document 1-5 about Arthur's]...", "tool_call_id": "call_a1b2c3d4"},
                {"role": "assistant", "content": "Great! I found...<tool_call>...</tool_call>"},
                {"role": "tool", "content": "...[Document 1-5 about First for Women]...", "tool_call_id": "call_e5f6g7h8"},
                {"role": "assistant", "content": "Perfect! Now I have all...<tool_call>{finish: \\boxed{Arthur's Magazine}}</tool_call>"}
            ],
            observation={"tool_outputs": {"call_e5f6g7h8": "Tool: local_search\n..."}},
            action=[{
                "id": "call_xyz789",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": '{"response": "\\\\boxed{Arthur\'s Magazine}"}'
                }
            }],
            model_response="Perfect! Now I have all...<tool_call>...</tool_call>",
            reward=1.0,  # 最终奖励在这里！
            done=True,
            info={
                "response": [...],
                "metadata": {
                    "extracted_answer": "Arthur's Magazine",
                    "ground_truths": ["Arthur's Magazine"],
                    "evaluation_method": "exact_match",
                    "f1_score": 1.0,
                    "exact_match": True
                },
                "is_correct": True
            }
        )
    ]
)

# 关键统计：
# - 总步数：3步
# - 模型调用次数：3次
# - 工具调用次数：2次（2次搜索）
# - 总奖励：1.0
# - 是否正确：True
```

### 3.6 并发执行总结

**单条轨迹的完整执行过程已经讲完，现在回到并发视角：**

```python
# ========== 批次级并发 ==========
batch_size = 64  # 64个不同问题
rollout_n = 8    # 每个问题8次采样
total_trajectories = 512  # 总共512条轨迹

# ========== 异步并发执行 ==========
async def run_all_trajectories():
    tasks = []
    for idx in range(512):
        # 每条轨迹都是独立的协程
        task = run_agent_trajectory_async(
            idx=idx,
            agent=agents[idx],
            env=envs[idx],
            application_id=f"trajectory_{idx}"
        )
        tasks.append(task)
    
    # 512条轨迹并发执行
    # 模型推理通过vLLM动态批处理
    # 工具调用通过线程池并发（最多64线程）
    results = await asyncio.gather(*tasks)
    return results

# 执行结果示例（512条轨迹）：
results = [
    trajectory_001,  # 我们详细追踪的这条，reward=1.0, steps=3
    trajectory_002,  # 可能 reward=0.0, steps=5（答错了）
    trajectory_003,  # 可能 reward=1.0, steps=2（答对了，只用2步）
    # ... 509 more trajectories
]

# ========== 统计信息 ==========
total_correct = sum(1 for r in results if r.reward > 0.5)  # 假设230条
total_wrong = 512 - total_correct  # 282条
success_rate = total_correct / 512  # 0.449 (44.9%)

avg_steps = sum(len(r.steps) for r in results) / 512  # 假设平均4.5步
avg_reward = sum(r.reward for r in results) / 512  # 0.449
```

---

## 四、奖励计算详解

### 4.1 答案提取 (`extract_answer_from_response`)

奖励函数首先从模型输出中提取答案，按优先级顺序：

```python
llm_response = "\\boxed{Arthur's Magazine}"

# 1. 最高优先级：提取 \boxed{} 内容
def unbox(s: str):
    i = s.find("boxed{")
    if i == -1:
        return None
    i += 6
    depth = 1
    j = i
    while depth and j < len(s):
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    return s[i:j-1]

extracted_answer = unbox(llm_response)
# extracted_answer = "Arthur's Magazine"

# 如果没有\boxed{}，还会尝试：
# 2. 粗体文本 **text**
# 3. 日期模式
# 4. 人名模式
# 5. 数字模式
# 6. 答案关键字模式
# 7. 最有信息量的句子
# 8. 前100字符
```

### 4.2 答案标准化 (`normalize_answer`)

```python
def normalize_answer(s: str) -> str:
    s = s.lower()                          # 转小写
    s = remove_punc(s)                     # 移除标点
    s = re.sub(r"\b(a|an|the)\b", " ", s) # 移除冠词
    s = white_space_fix(s)                 # 规范空白
    return s

# 示例：
# "Arthur's Magazine" -> "arthurs magazine"
# "Arthur's Magazine" (ground_truth) -> "arthurs magazine"
```

### 4.3 相似度计算

#### **Exact Match (EM)**

```python
def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# 我们的例子：
# normalize("Arthur's Magazine") == normalize("Arthur's Magazine")
# "arthurs magazine" == "arthurs magazine"
# EM = True
```

#### **F1 Score**

```python
def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    # pred_tokens = ["arthurs", "magazine"]
    # gt_tokens = ["arthurs", "magazine"]
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())  # 2
    
    if num_same == 0:
        return 0, 0, 0
    
    precision = num_same / len(pred_tokens)  # 2/2 = 1.0
    recall = num_same / len(gt_tokens)       # 2/2 = 1.0
    f1 = (2 * precision * recall) / (precision + recall)  # 1.0
    
    return f1, precision, recall

# 我们的例子：F1 = 1.0
```

### 4.4 奖励赋值

```python
# RewardConfig 默认值
config = RewardConfig(
    correct_reward=1.0,      # 正确答案奖励
    incorrect_reward=0.0,    # 错误答案奖励
    f1_threshold=0.3         # F1阈值（代码中硬编码）
)

# 判断是否正确
is_correct = EM or (F1 >= 0.3)

# 计算奖励
if is_correct:
    if EM:
        reward = 1.0  # 完全匹配
    else:
        reward = 1.0 * F1  # 部分匹配，按F1缩放
else:
    reward = 0.0

# 我们的例子：
# EM = True
# reward = 1.0

# 返回
return RewardOutput(
    reward=1.0,
    is_correct=True,
    metadata={
        "extracted_answer": "Arthur's Magazine",
        "ground_truths": ["Arthur's Magazine"],
        "evaluation_method": "exact_match",
        "best_match": "Arthur's Magazine",
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "exact_match": True,
        "f1_threshold": 0.3
    }
)
```

---

## 五、Token级轨迹转换

Agent交互生成的是对话形式，需要转换为token序列用于PPO训练。

### 5.1 对话到Token的转换

```python
# 完整对话历史
messages = [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "Which magazine was started first..."},
    {"role": "assistant", "content": "I need to search...<tool_call>..."},
    {"role": "tool", "content": "[Document 1]...", "tool_call_id": "..."},
    {"role": "assistant", "content": "Now let me search...<tool_call>..."},
    {"role": "tool", "content": "[Document 1]...", "tool_call_id": "..."},
    {"role": "assistant", "content": "Based on...<tool_call>{finish}..."}
]

# 使用Qwen的chat template转换
chat_parser = ChatTemplateParser.get_parser(tokenizer)
```

### 5.2 Prompt Tokens (输入部分)

```python
# 初始prompt（系统提示 + 用户问题）
prompt_str = chat_parser.parse(
    messages[:2],  # system + user
    add_generation_prompt=True,
    is_first_msg=True
)

# 实际格式（Qwen）：
# <|im_start|>system
# You are a helpful AI assistant...
# Available tools: ...
# <|im_end|>
# <|im_start|>user
# Which magazine was started first Arthur's Magazine or First for Women?
# <|im_end|>
# <|im_start|>assistant
# 

prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
# prompt_tokens = [151644, 8948, 198, 2610, 525, ...] (长度约300-500)
```

### 5.3 Response Tokens (生成部分)

每一步的assistant消息和tool消息都会被tokenize：

```python
response_tokens = []
response_masks = []

# 第1步：assistant的tool_call
assistant_msg_1 = {"role": "assistant", "content": "I need to search...<tool_call>..."}
tokens_1, masks_1 = convert_messages_to_tokens_and_masks([assistant_msg_1], ...)
# tokens_1 = [40, 1184, 311, 2711, ...] (长度约50-100)
# masks_1 = [1, 1, 1, ...]  # 全1，因为是模型生成的

response_tokens.extend(tokens_1)
response_masks.extend(masks_1)

# 第1步：tool的返回
tool_msg_1 = {"role": "tool", "content": "[Document 1]...", ...}
tokens_env_1, masks_env_1 = convert_messages_to_tokens_and_masks([tool_msg_1], ...)
# tokens_env_1 = [58, 24361, 352, 60, ...] (长度约100-300)
# masks_env_1 = [0, 0, 0, ...]  # 全0，因为不是模型生成的

response_tokens.extend(tokens_env_1)
response_masks.extend(masks_env_1)

# 第2步：重复上述过程
# ...

# 最后一步：finish
assistant_msg_final = {"role": "assistant", "content": "...<tool_call>{finish}..."}
tokens_final, masks_final = convert_messages_to_tokens_and_masks([assistant_msg_final], ...)
response_tokens.extend(tokens_final)
response_masks.extend(masks_final)

# 最终结果：
# response_tokens = [token1, token2, ..., token_n]  # 长度可能500-1500
# response_masks = [1, 1, ..., 0, 0, ..., 1, 1, ...]
#                  ↑模型生成  ↑工具返回  ↑模型生成
```

### 5.4 完整序列构造

```python
# 合并prompt和response
input_ids = prompt_tokens + response_tokens
# input_ids.shape = (total_length,)  # 例如 1200

attention_mask = [1] * len(input_ids)
# attention_mask.shape = (total_length,)

# response_mask指示哪些token是模型生成的（用于计算loss）
full_response_mask = [0] * len(prompt_tokens) + response_masks
# full_response_mask.shape = (total_length,)
# [0, 0, ..., 0, 1, 1, ..., 0, 0, ..., 1, 1, ...]
#  ↑ prompt部分  ↑模型生成1 ↑工具返回 ↑模型生成2
```

### 5.5 奖励分配到Token

```python
# 初始化token级奖励（全0）
token_level_scores = torch.zeros(len(input_ids))

# 只在最后一个模型生成的token上分配奖励
# 找到最后一个mask=1的位置
last_model_token_idx = -1
for i in range(len(full_response_mask) - 1, -1, -1):
    if full_response_mask[i] == 1:
        last_model_token_idx = i
        break

# 将轨迹总奖励分配到最后一个token
token_level_scores[last_model_token_idx] = 1.0  # 假设答对了

# token_level_scores = [0, 0, ..., 0, 1.0, 0, 0]
#                                     ↑最后一个模型生成的token
```

---

## 六、PPO训练过程

### 6.1 数据批次准备

经过上述处理后，一个批次的数据结构：

```python
batch = DataProto(
    batch={
        # [512, max_length] - 批次中所有序列
        "input_ids": torch.tensor([...]),
        "attention_mask": torch.tensor([...]),
        "response_mask": torch.tensor([...]),
        "responses": torch.tensor([...]),  # 只包含response部分
        
        # [512, max_length] - token级奖励
        "token_level_scores": torch.tensor([...]),
        
        # 待计算的量
        "old_log_probs": None,  # 旧策略的log概率
        "values": None,         # 价值函数估计
        "advantages": None,     # 优势函数
    },
    non_tensor_batch={
        "uid": [...],  # 每条轨迹的唯一ID
        "extra_info": [...],  # 原始任务信息
    }
)
```

### 6.2 计算 Old Log Probabilities

```python
# 使用当前（训练前）的Actor模型计算log概率
old_log_prob = actor_rollout_wg.compute_log_prob(batch)

# 模型前向传播
logits = model(input_ids, attention_mask)
# logits.shape = [512, max_length, vocab_size]

# 计算每个token的log概率
log_probs = torch.log_softmax(logits, dim=-1)
# 取实际生成的token的log概率
old_log_probs = log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
# old_log_probs.shape = [512, response_length]

# 只保留response_mask=1的部分
old_log_probs = old_log_probs * response_mask

# 添加到batch
batch.batch["old_log_probs"] = old_log_probs
```

### 6.3 计算 Values (如果使用Critic)

```python
# 使用Critic网络估计状态价值
values = critic_wg.compute_values(batch)

# Critic模型前向传播
value_logits = critic_model(input_ids, attention_mask)
# value_logits.shape = [512, max_length, 1]

values = value_logits.squeeze(-1)
# values.shape = [512, max_length]

batch.batch["values"] = values
```

### 6.4 Rejection Sampling（可选）

```python
# 根据奖励过滤样本
uids = batch.non_tensor_batch["uid"]
unique_uids = np.unique(uids)
valid_mask = torch.ones(len(uids), dtype=torch.bool)

for uid in unique_uids:
    uid_mask = uids == uid
    uid_rewards = batch.batch["token_level_scores"][uid_mask].sum(-1)
    
    # 同一个问题的8个采样
    # 如果全部答错(reward <= 0) 或 全部答对(reward >= 1)
    # 则过滤掉，因为无法提供学习信号
    if (uid_rewards <= 0).all() or (uid_rewards >= 1).all():
        valid_mask[uid_mask] = False

# 保留有效样本
batch = batch[valid_mask]
# 批次大小可能从512降到300-400
```

### 6.5 计算 Advantages

```python
# 计算优势函数 A(s,a) = Q(s,a) - V(s)
advantages = compute_advantage(
    rewards=batch.batch["token_level_scores"],
    values=batch.batch["values"],
    response_mask=batch.batch["response_mask"],
    gamma=0.99,  # 折扣因子
    lam=0.95     # GAE参数
)

# GAE (Generalized Advantage Estimation)
# A_t = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...
# 其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

# 由于我们的奖励只在最后一个token，简化为：
# A_t = R - V(s_t)  对于最后一个token
# A_t = -V(s_t)     对于其他token

batch.batch["advantages"] = advantages
# advantages.shape = [batch_size, max_length]
```

### 6.6 PPO Actor更新

```python
# PPO算法核心：多个epoch更新Actor
for ppo_epoch in range(num_ppo_epochs):  # 通常1-4个epoch
    # Mini-batch训练
    for mini_batch in split_batch(batch, mini_batch_size=32):
        # 1. 计算新的log概率
        new_log_probs = actor_model(mini_batch)
        
        # 2. 计算重要性采样比率
        ratio = torch.exp(new_log_probs - mini_batch["old_log_probs"])
        # ratio.shape = [32, max_length]
        
        # 3. 计算裁剪目标
        clip_ratio = 0.28  # 从配置中获取
        advantages = mini_batch["advantages"]
        
        # 未裁剪的目标
        surr1 = ratio * advantages
        
        # 裁剪后的目标
        ratio_clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        surr2 = ratio_clipped * advantages
        
        # PPO目标：取两者的最小值
        policy_loss = -torch.min(surr1, surr2)
        
        # 4. 应用response_mask
        response_mask = mini_batch["response_mask"]
        policy_loss = policy_loss * response_mask
        
        # 5. 聚合loss
        # loss_agg_mode = "seq-mean-token-sum"
        # 先对每个序列的token求和，再对序列求平均
        policy_loss = policy_loss.sum(dim=1).mean()
        
        # 6. 反向传播
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
```

### 6.7 Critic更新（如果使用）

```python
for ppo_epoch in range(num_ppo_epochs):
    for mini_batch in split_batch(batch, mini_batch_size=32):
        # 1. 计算新的value估计
        new_values = critic_model(mini_batch)
        
        # 2. 计算value目标（return）
        # return = reward + γV(s_{t+1})
        returns = mini_batch["token_level_scores"] + gamma * new_values[:, 1:]
        
        # 3. Value loss (MSE)
        value_loss = (new_values - returns) ** 2
        value_loss = value_loss * mini_batch["response_mask"]
        value_loss = value_loss.sum(dim=1).mean()
        
        # 4. 反向传播
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
```

---

## 七、关键指标与日志

### 7.1 训练指标

```python
metrics = {
    # 批次统计
    "batch/solve_none": 15,        # 全部答错的问题数
    "batch/solve_all": 25,         # 全部答对的问题数
    "batch/solve_partial": 24,     # 部分答对的问题数
    
    # 奖励统计
    "critic/full-score/mean": 0.45,  # 平均总奖励
    "critic/full-score/max": 1.0,    # 最大总奖励
    "critic/full-score/min": 0.0,    # 最小总奖励
    
    # Actor指标
    "actor/entropy": 2.34,           # 策略熵（探索度）
    "actor/policy_loss": 0.23,       # 策略损失
    "actor/approx_kl": 0.012,        # KL散度（策略变化）
    
    # Critic指标（如果使用）
    "critic/value_loss": 0.15,       # 价值损失
    
    # 轨迹统计
    "trajectory/avg_steps": 4.5,     # 平均步数
    "trajectory/avg_reward": 0.45,   # 平均奖励
    "trajectory/success_rate": 0.45, # 成功率
    
    # 时间统计
    "timing/generation_time": 12.3,  # 生成时间
    "timing/training_time": 5.6,     # 训练时间
}
```

### 7.2 验证流程

```python
# 每200步进行一次验证
if global_step % 200 == 0:
    val_metrics = validate_agent(val_dataset)
    
    # 验证指标
    val_metrics = {
        "val/accuracy": 0.52,        # 验证集准确率
        "val/avg_reward": 0.52,      # 平均奖励
        "val/f1_score": 0.58,        # F1分数
        "val/avg_steps": 4.2,        # 平均步数
    }
```

---

## 八、完整流程图

```
[数据加载] 
   ↓
[批次准备: 64个问题 × 8次采样 = 512条轨迹]
   ↓
┌─────────────────────────────────────────────┐
│  Agent-Environment 交互 (每条轨迹独立)      │
├─────────────────────────────────────────────┤
│  Step 1: 模型生成工具调用                   │
│    └→ 提示词 → Verl Rollout → Tool Call     │
│  Step 2: 执行工具                           │
│    └→ LocalRetrievalTool → 检索服务器       │
│  Step 3: 模型处理工具结果                   │
│    └→ 更新对话历史                          │
│  ...                                        │
│  Step N: 生成最终答案 (finish)               │
│    └→ 计算奖励 (Reward Function)            │
└─────────────────────────────────────────────┘
   ↓
[轨迹转换为Token序列]
   ├→ prompt_tokens
   ├→ response_tokens  
   ├→ response_masks (1=模型生成, 0=环境返回)
   └→ token_level_scores (奖励分配到最后一个token)
   ↓
[计算训练所需量]
   ├→ old_log_probs (旧策略的log概率)
   ├→ values (Critic估计)
   └→ advantages (优势函数)
   ↓
[Rejection Sampling]
   └→ 过滤全对/全错的样本组
   ↓
[PPO训练]
   ├→ Actor更新 (多个mini-batch, 多个epoch)
   │   └→ 最大化 min(r*A, clip(r)*A)
   └→ Critic更新 (如果使用)
       └→ 最小化 (V - Return)^2
   ↓
[保存检查点] (每200步)
   ↓
[验证] (每200步)
   ↓
[重复下一个批次]
```

---

## 九、关键技术细节

### 9.1 异步生成

```python
# 使用AsyncAgentExecutionEngine并行处理多条轨迹
async def run_all_trajectories(n_trajectories):
    tasks = []
    for idx in range(n_trajectories):
        task = run_agent_trajectory_async(idx)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# 512条轨迹可以高度并行化，大幅提升效率
```

### 9.2 动态批次大小

```python
# 根据token数量动态调整mini-batch大小
actor.ppo_max_token_len_per_gpu = 24000

# 如果某个batch的总token数超过限制，自动减小batch_size
total_tokens = sum(len(seq) for seq in batch)
if total_tokens > ppo_max_token_len_per_gpu * world_size:
    batch = reduce_batch_size(batch)
```

### 9.3 梯度检查点

```python
# 节省显存
actor_rollout_ref.model.enable_gradient_checkpointing = True

# 权衡：减少显存占用，增加计算时间
```

### 9.4 FSDP并行

```python
# 使用FSDP (Fully Sharded Data Parallel) 进行分布式训练
actor_rollout_ref.actor.strategy = "fsdp"
trainer.n_gpus_per_node = 2
trainer.nnodes = 1

# 模型参数和优化器状态分片存储在2个GPU上
```

---

## 十、性能优化

### 10.1 检索服务优化

- 使用本地检索服务器（`RETRIEVAL_SERVER_URL`）
- E5嵌入模型 + 向量数据库
- 多线程并行执行工具调用

### 10.2 推理优化

- vLLM引擎用于快速推理
- Flash Attention加速
- 异步模式减少等待时间

### 10.3 训练优化

- Rejection sampling减少无效样本
- 动态批次大小充分利用GPU
- 梯度累积支持大batch训练

---

## 十一、常见问题

### Q1: 为什么要将奖励分配到最后一个token？

因为只有在完成整个推理过程后才能判断答案是否正确。中间步骤的工具调用本身无法评判对错，只有最终答案才能与ground truth比较。

### Q2: response_mask的作用是什么？

区分哪些token是模型生成的（mask=1），哪些是环境返回的（mask=0）。只有模型生成的token才会计算loss和更新梯度，环境返回的token不应该影响模型训练。

### Q3: 为什么要采样n=8次？

增加样本多样性，同时通过rejection sampling保留最有信息量的样本（部分答对/部分答错）。全对或全错的样本组无法提供有效的学习信号。

### Q4: Stepwise advantage和Trajectory advantage的区别？

- **Trajectory-level**: 整条轨迹作为一个序列，最大长度 = max_prompt_length + max_response_length
- **Stepwise**: 每一步作为独立序列，每步最大长度独立计算，适合长轨迹任务

本任务使用trajectory-level模式。

---

## 十二、总结

这个训练流程的核心是将**多步推理问题**转化为**序列生成问题**，通过PPO算法优化模型生成能够正确使用工具并得出正确答案的轨迹。

关键创新点：
1. **Agent-Environment范式**: 将推理过程建模为Agent与环境的交互
2. **工具调用机制**: 模型学习何时调用工具、如何构造查询
3. **稀疏奖励**: 只在最终答案处给奖励，鼓励模型完成完整推理链
4. **Token级训练**: 将对话历史转换为token序列，利用标准的PPO算法

这种方法可以扩展到其他需要多步推理和工具使用的任务，如代码生成、数学问题求解等。

