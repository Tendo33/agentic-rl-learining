
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

**代码位置**: `rllm/examples/search/train_search_agent.py`

```python
# 第15-16行: 加载数据集
train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
# 从 /root/github_project/rllm/rllm/data/datasets/hotpotqa/train.parquet 加载

# 第18行: 配置工具映射
tool_map = {"local_search": LocalRetrievalTool}
# LocalRetrievalTool 连接到 RETRIEVAL_SERVER_URL 的检索服务

# 第20-24行: 配置环境参数
env_args = {
    "max_steps": 20,                    # 最多20步
    "tool_map": tool_map,               # 工具映射
    "reward_fn": search_reward_fn,      # 奖励函数 (来自 rllm/rewards/reward_fn.py:62)
}

# 第26行: 配置Agent参数
agent_args = {"system_prompt": SEARCH_SYSTEM_PROMPT, "tool_map": tool_map, "parser_name": "qwen"}
# SEARCH_SYSTEM_PROMPT 定义在 rllm/agents/system_prompts.py

# 第29-37行: 创建训练器
trainer = AgentTrainer(
    agent_class=ToolAgent,        # rllm/agents/tool_agent.py:17
    env_class=ToolEnvironment,    # rllm/environments/tools/tool_env.py:12
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    agent_args=agent_args,
    env_args=env_args,
)

# 第39行: 开始训练
trainer.train()  # 调用 rllm/trainer/agent_trainer.py:68-90
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

**代码位置**: `rllm/engine/agent_execution_engine.py:506-509`

```python
# 在 AgentExecutionEngine.execute_tasks 方法中创建环境和Agent
# 第506行: 从任务字典创建环境实例
self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
# 调用 ToolEnvironment.from_dict (rllm/environments/tools/tool_env.py:145-150)

# 第507行: 创建Agent实例
self.agents[index] = self.agent_class(**self.agent_args)
# 调用 ToolAgent.__init__ (rllm/agents/tool_agent.py:23-61)

# 具体的初始化过程：
env = ToolEnvironment(
    task={
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "data_source": "hotpotqa"
    },
    tool_map={"local_search": LocalRetrievalTool},
    reward_fn=search_reward_fn,  # rllm/rewards/reward_fn.py:62-81
    max_steps=20
)
# ToolEnvironment.__init__ 在 rllm/environments/tools/tool_env.py:17-47

agent = ToolAgent(
    system_prompt=SEARCH_SYSTEM_PROMPT,
    tool_map={"local_search": LocalRetrievalTool},
    parser_name="qwen"
)
# ToolAgent.__init__ 在 rllm/agents/tool_agent.py:23-61
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

**代码位置**: `rllm/engine/agent_execution_engine.py:168-209`

现在让我们追踪**单条轨迹**的完整过程（从512条中选一条）：

```python
# ========== 轨迹执行开始 ==========
# 第168行: 定义异步轨迹执行函数
async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
    """执行单条Agent轨迹的完整流程"""
    
    # 第170-171行: 获取当前Agent和Environment实例
    agent = self.agents[idx]  # 从512个Agent中获取第idx个
    env = self.envs[idx]      # 从512个Environment中获取第idx个
    
    # 第174-185行: 初始化轨迹追踪变量
    termination_reason = None    # 终止原因
    prompt_token_len = 0         # prompt的token长度
    prompt_tokens = []           # prompt的token列表
    response_token_len = 0       # response的token长度
    response_tokens = []         # response的token列表
    response_masks = []          # response的mask列表
    total_time = 0.0            # 总耗时
    reward_time = None          # 奖励计算耗时
    llm_time = 0.0              # LLM推理耗时
    env_time = 0.0              # 环境执行耗时
    reward = 0.0                # 累计奖励
    episode_steps = []          # 每一步的prompt-response对

# ========== 环境初始化 ==========
# 在 AgentExecutionEngine.run_agent_trajectory_async 方法中
# 环境已在 init_envs_and_agents (agent_ppo_trainer.py:170) 中创建
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
# 第191行: 异步调用环境重置
loop = asyncio.get_event_loop()
observation, info = await loop.run_in_executor(self.executor, env.reset)
# env.reset 定义在 rllm/environments/tools/tool_env.py:49-53

# 返回值：
observation = {
    "question": "Which magazine was started first Arthur's Magazine or First for Women?"
}
info = {}

# ========== Agent重置 ==========
# 第195行: 重置Agent状态
agent.reset()
# agent.reset 定义在 rllm/agents/tool_agent.py:153-156

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
# 第196-202行: Agent根据环境反馈更新状态
agent.update_from_env(
    observation=observation,
    reward=0.0,
    done=False,
    info=info
)
# agent.update_from_env 定义在 rllm/agents/tool_agent.py:86-100

# update_from_env做了什么？
# 1. 格式化observation为消息 (第93行)
obs_messages = agent._format_observation_as_messages(observation)
# _format_observation_as_messages 定义在 rllm/agents/tool_agent.py:63-84
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

**代码位置**: `rllm/engine/agent_execution_engine.py:211-243`

```python
# ========== 1. 构造提示词 ==========
# 第213行: 获取当前对话历史
prompt_messages = agent.chat_completions  # agent.chat_completions 定义在 rllm/agents/tool_agent.py:159-161
# prompt_messages = [
#   {"role": "system", "content": "...system prompt + tools..."},
#   {"role": "user", "content": "Which magazine was started first Arthur's Magazine or First for Women?"}
# ]

# ========== 2. 通过Verl引擎调用模型 ==========
# 第231行: 异步调用模型生成响应
response = await self.get_model_response(prompt_messages, application_id, **kwargs)
# self.get_model_response 定义在 rllm/engine/agent_execution_engine.py:120-151

# 对于Verl引擎，会调用：
# VerlEngine.get_model_response (rllm/engine/rollout/verl_engine.py:42-83)

# 内部流程（vLLM引擎）：
# 1. 第56行: 将prompt_messages转换为token序列
#    prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, ...)
#    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
# 
# 2. 第63行: 添加到vLLM的请求队列并异步生成
#    completion_ids = await self.server_manager.generate(
#        request_id=application_id, 
#        prompt_ids=prompt_ids, 
#        sampling_params=sampling_params
#    )
#    # server_manager 在 verl/experimental/agent_loop/agent_loop.py 中定义
# 
# 3. vLLM动态批处理多个请求
# 4. GPU执行推理
# 5. 第70行: 解码生成的token为文本
#    completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

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

**代码位置**: `rllm/agents/tool_agent.py:102-151` & `rllm/engine/agent_execution_engine.py:243-244`

```python
# ========== Agent更新状态 ==========
# 第243行 (engine): 调用Agent的update_from_model方法
action: Action = agent.update_from_model(response)
action = action.action

# update_from_model内部流程 (rllm/agents/tool_agent.py:102-151)：

# 1. 第111行: 使用Qwen解析器提取工具调用
tool_calls = agent.tool_parser.parse(response)
# tool_parser.parse 由具体的解析器实现，例如 QwenToolParser
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

# 2. 第112-119行: 为每个工具调用生成唯一ID
tool_calls_dict = [
    {
        "id": str(uuid.uuid4()),  # 第114行: 生成UUID
        "type": "function",
        "function": tool_call.to_dict(),  # 第116行: 转换为字典
    }
    for tool_call in tool_calls
]
# 实际生成的结构：
# [{
#     "id": "call_a1b2c3d4",
#     "type": "function",
#     "function": {
#         "name": "local_search",
#         "arguments": '{"query": "Arthur\'s Magazine history founding date publication", "top_k": 5}'
#     }
# }]

# 3. 第146行: 更新agent.messages
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

# 4. 第148行: 创建新的Step并添加到Trajectory
new_step = Step(
    chat_completions=copy.deepcopy(self.chat_completions),  # 保存当前对话历史快照
    action=tool_calls_dict,  # 工具调用
    model_response=response,  # 原始模型输出
    observation=self.current_observation,  # 当前观察
    # reward, done, info 将在下一次 update_from_env 时更新
)
# 第149行: 添加到轨迹
self._trajectory.steps.append(new_step)

# 现在agent._trajectory.steps = [Step1]

# 5. 第151行: 返回Action对象
return Action(action=tool_calls_dict)
# Action 定义在 rllm/agents/agent.py
# action.action = [
#   {
#     "id": "call_a1b2c3d4",
#     "type": "function",
#     "function": {...}
#   }
# ]
```

#### **第3步：执行工具调用**

**代码位置**: `rllm/environments/tools/tool_env.py:55-142` & `rllm/engine/agent_execution_engine.py:246-264`

```python
# ========== 环境执行工具调用 ==========
# 第250行 (engine): 异步调用环境的step方法
next_observation, reward, done, info = await asyncio.wait_for(
    loop.run_in_executor(self.executor, env.step, action), 
    timeout=(self.trajectory_timeout - total_time)
)

# env.step内部流程 (rllm/environments/tools/tool_env.py:55-109)：

# 1. 第70行: 增加步数计数
self.step_count += 1  # 现在 = 1

# 2. 第74-80行: 检查是否是finish工具调用
done = self.step_count >= self.max_steps or isinstance(action, str)
if isinstance(action, list) and action:
    for tool_call in action:
        if tool_call.get("function", {}).get("name") == "finish":
            done = True
            break

# 如果是finish，跳转到第82-101行计算奖励
# 如果不是finish，继续执行工具

# 3. 第105行: 并行执行所有工具调用
tool_outputs = self._execute_tool_calls(tool_calls)
# _execute_tool_calls 定义在第111-142行，使用线程池并发执行

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

# 4. 第119-122行 (_execute_tool_calls内部): 调用工具
def execute_tool(tool_call):
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])
    tool_output = self.tools(tool_name=tool_name, **tool_args)
    # self.tools 是 MultiTool 实例 (定义在 rllm/tools/multi_tool.py)
    # MultiTool.__call__ 会根据 tool_name 查找并实例化对应的工具类
    # 然后调用 tool.forward(**tool_args)
    
    tool_output_str = tool_output.to_string()
    output_queue.put((tool_call["id"], tool_output_str))

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

**代码位置**: `rllm/engine/agent_execution_engine.py:268-274` & `rllm/agents/tool_agent.py:86-100`

```python
# ========== Agent更新状态 ==========
# 第268-274行 (engine): 更新Agent内部状态
agent.update_from_env(
    observation=next_observation,
    reward=reward,
    done=done,
    info=info,
)

# update_from_env内部 (rllm/agents/tool_agent.py:86-100)：

# 1. 第93行: 格式化observation为消息
obs_messages = self._format_observation_as_messages(observation)
# _format_observation_as_messages 会检测 observation 类型
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

# ========== 中间状态快照 (第2步完成时) ==========
# Agent状态：
# - agent.messages: 6条消息 (system, user, assistant1, tool1, assistant2, tool2)
# - agent._trajectory.steps: [Step1, Step2]
# - agent.current_observation: {"tool_outputs": {...}}
# 
# Environment状态：
# - env.step_count: 2
# - env.task: 原始任务信息
# 
# Trajectory状态：
# - trajectory.uid: "abc123..."
# - trajectory.steps: [Step1, Step2]
# - trajectory.reward: 0.0 (尚未计算)
# 
# Token状态：
# - prompt_tokens: [151644, 8948, ...] (长度约400)
# - response_tokens: [40, 1184, ..., 58, 24361, ...] (长度约200)
# - response_masks: [1, 1, ..., 0, 0, ...] (长度约200)
```

#### **第9步：模型生成最终答案**

```python
# ========== 模型第3次调用 ==========
# 第231行: 再次调用模型生成
prompt_messages = agent.chat_completions  # 6条消息
response = await self.get_model_response(prompt_messages, application_id, **kwargs)

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

**代码位置**: `rllm/environments/tools/tool_env.py:74-101` & `rllm/rewards/search_reward.py:233-254`

```python
# ========== 环境检测finish ==========
# 再次调用 env.step，这次检测到finish
next_observation, reward, done, info = env.step(action.action)

# env.step内部 (rllm/environments/tools/tool_env.py:74-101)：
self.step_count += 1  # 第70行: = 3

# 第74-80行: 检测到finish工具调用
done = self.step_count >= self.max_steps or isinstance(action, str)
if isinstance(action, list) and action:
    for tool_call in action:
        if tool_call.get("function", {}).get("name") == "finish":
            done = True
            break

# 第82-97行: 处理终止情况
if done:
    # 第86-97行: 提取finish的arguments
    finish_action = None
    for tool_call in action:
        if tool_call.get("function", {}).get("name") == "finish":
            finish_action = tool_call
            break
    if finish_action:
        arguments = finish_action.get("function", {}).get("arguments", {})
        llm_response = arguments.get("response", "")

# llm_response = "\\boxed{Arthur's Magazine}"

# ========== 调用奖励函数 ==========
# 第99-100行: 调用奖励函数
task_info = self.task
# {
#   "id": 0,
#   "question": "Which magazine was started first Arthur's Magazine or First for Women?",
#   "answer": "Arthur's Magazine",
#   "data_source": "hotpotqa"
# }

reward_output = self.reward_fn(task_info=task_info, action=llm_response)
# reward_fn 是 search_reward_fn (rllm/rewards/reward_fn.py:62-81)
# 实际调用 RewardSearchFn.__call__ (rllm/rewards/search_reward.py:233-254)

# reward_fn内部流程（详见第四章）：
# 1. 第186行: 提取答案
#    extracted_answer = self.extract_answer_from_response(model_response)
#    # extract_answer_from_response 定义在第58-183行
#    # 使用 unbox 函数提取 \boxed{} 内容 (第69-85行)
#    unbox("\\boxed{Arthur's Magazine}") -> "Arthur's Magazine"
# 
# 2. 第205-213行: 计算 Exact Match
#    em = self.exact_match_score(extracted_answer, gt_str)
#    # normalize("Arthur's Magazine") == normalize("Arthur's Magazine")
#    # "arthurs magazine" == "arthurs magazine"  -> True
# 
# 3. 第216行: 计算 F1 Score
#    f1, precision, recall = self.f1_score(extracted_answer, gt_str)
#    # f1_score 定义在第31-52行
#    # F1 = 1.0
# 
# 4. 第227行: 判断是否正确
#    is_correct = max_em or max_f1 >= f1_threshold  # True
# 
# 5. 第241-250行: 赋予奖励
#    if is_correct:
#        if metadata.get("exact_match", False):
#            reward = self.config.correct_reward  # 1.0
#    else:
#        reward = self.config.incorrect_reward  # 0.0

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

**代码位置**: `rllm/rewards/search_reward.py:58-183`

奖励函数首先从模型输出中提取答案，按优先级顺序：

```python
llm_response = "\\boxed{Arthur's Magazine}"

# extract_answer_from_response 方法 (第58-183行)
# 第59行: 清理响应文本
response = response.strip()

# 第62-63行: 移除thinking标签
response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

# 1. 最高优先级：提取 \boxed{} 内容 (第68-90行)
def unbox(s: str):  # 第69行
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

**代码位置**: `rllm/rewards/search_reward.py:13-29`

```python
# normalize_answer 方法 (第13-29行)
def normalize_answer(self, s: str) -> str:
    # 第16-17行: 定义移除冠词的辅助函数
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    # 第19-20行: 规范空白
    def white_space_fix(text):
        return " ".join(text.split())
    
    # 第22-24行: 移除标点
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    # 第26-27行: 转小写
    def lower(text):
        return text.lower()
    
    # 第29行: 应用所有转换
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# 示例：
# "Arthur's Magazine" -> "arthurs magazine"
# "Arthur's Magazine" (ground_truth) -> "arthurs magazine"
```

### 4.3 相似度计算

#### **Exact Match (EM)**

**代码位置**: `rllm/rewards/search_reward.py:54-56`

```python
# exact_match_score 方法 (第54-56行)
def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
    """Calculate exact match score"""
    return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

# 我们的例子：
# normalize("Arthur's Magazine") == normalize("Arthur's Magazine")
# "arthurs magazine" == "arthurs magazine"
# EM = True
```

#### **F1 Score**

**代码位置**: `rllm/rewards/search_reward.py:31-52`

```python
# f1_score 方法 (第31-52行)
def f1_score(self, prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Calculate F1 score between prediction and ground truth"""
    # 第33-34行: 标准化预测和真实答案
    normalized_prediction = self.normalize_answer(prediction)
    normalized_ground_truth = self.normalize_answer(ground_truth)
    
    # 第36行: 定义零分指标
    ZERO_METRIC = (0, 0, 0)
    
    # 第38-41行: 处理yes/no/noanswer的特殊情况
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    
    # 第43-44行: 分词
    pred_tokens = normalized_prediction.split()
    gt_tokens = normalized_ground_truth.split()
    
    # pred_tokens = ["arthurs", "magazine"]
    # gt_tokens = ["arthurs", "magazine"]
    
    # 第45-46行: 计算共同token数量
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())  # 2
    
    # 第47-48行: 没有共同token则返回零分
    if num_same == 0:
        return ZERO_METRIC
    
    # 第49-51行: 计算精确率、召回率和F1
    precision = 1.0 * num_same / len(pred_tokens)  # 2/2 = 1.0
    recall = 1.0 * num_same / len(gt_tokens)       # 2/2 = 1.0
    f1 = (2 * precision * recall) / (precision + recall)  # 1.0
    
    # 第52行: 返回三个指标
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

**代码位置**: `rllm/engine/agent_execution_engine.py:282-338` & `rllm/agents/utils.py:38-75`

Agent交互生成的是对话形式，需要转换为token序列用于PPO训练。

### 5.1 对话到Token的转换

**代码位置**: `rllm/agents/utils.py:38-75`

```python
# 在 agent_execution_engine.py 中的 run_agent_trajectory_async 方法

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
# chat_parser 在 AgentExecutionEngine.__init__ 中初始化 (第91行)
```

### 5.2 Prompt Tokens (输入部分)

**代码位置**: `rllm/engine/agent_execution_engine.py:203-209`

```python
# 第203-204行: 初始prompt的token化
messages = agent.chat_completions
prompt_tokens, _ = convert_messages_to_tokens_and_masks(
    messages,
    tokenizer=self.tokenizer, 
    parser=self.chat_parser, 
    contains_first_msg=True,  # 包含系统消息
    contains_generation_msg=True  # 包含生成提示
)
# convert_messages_to_tokens_and_masks 定义在 rllm/agents/utils.py:38-75

# 内部流程：
# 1. 第57行: 使用chat_parser.parse转换消息
#    prompt_str = parser.parse([msg], add_generation_prompt=..., is_first_msg=...)
# 
# 2. 第64行: tokenize
#    msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
# 
# 3. 第65-66行: 生成mask (assistant=1, 其他=0)
#    mask_value = 1 if msg["role"] == "assistant" else 0
#    msg_mask = [mask_value] * len(msg_tokens)

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

# 第205行: 获取prompt长度
prompt_token_len = len(prompt_tokens)
# prompt_tokens = [151644, 8948, 198, 2610, 525, ...] (长度约300-500)
```

### 5.3 Response Tokens (生成部分)

**代码位置**: `rllm/engine/agent_execution_engine.py:282-338`

每一步的assistant消息和tool消息都会被tokenize：

```python
# 在每一步的循环中 (第211-342行)

response_tokens = []
response_masks = []

# 第282-292行: 获取最近的assistant和environment消息
chat_completions_messages = agent.chat_completions
assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)
# get_recent_assistant_user_messages 定义在 rllm/agents/utils.py:6-35

# 第288-292行: 转换为tokens
assistant_msg_tokens, assistant_msg_masks = [], []
env_msg_tokens, env_msg_masks = [], []
if assistant_message:
    # 第290行: 转换assistant消息
    assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
        [assistant_message], 
        tokenizer=self.tokenizer, 
        parser=self.chat_parser, 
        contains_first_msg=False, 
        contains_generation_msg=False
    )
if env_messages:
    # 第292行: 转换environment消息（tool输出）
    env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
        env_messages, 
        tokenizer=self.tokenizer, 
        parser=self.chat_parser, 
        contains_first_msg=False, 
        contains_generation_msg=True
    )

# 第1步：assistant的tool_call
# assistant_msg_1 = {"role": "assistant", "content": "I need to search...<tool_call>..."}
# tokens_1, masks_1 会是：
# tokens_1 = [40, 1184, 311, 2711, ...] (长度约50-100)
# masks_1 = [1, 1, 1, ...]  # 全1，因为是模型生成的

# 第321行: 添加assistant的tokens
response_tokens.extend(assistant_msg_tokens)
response_masks.extend(assistant_msg_masks)

# 第1步：tool的返回
# tool_msg_1 = {"role": "tool", "content": "[Document 1]...", ...}
# tokens_env_1, masks_env_1 会是：
# tokens_env_1 = [58, 24361, 352, 60, ...] (长度约100-300)
# masks_env_1 = [0, 0, 0, ...]  # 全0，因为不是模型生成的（第65行的逻辑）

# 第337行: 添加environment的tokens (在done=False时)
response_tokens.extend(env_msg_tokens)
response_masks.extend(env_msg_masks)

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

**主要代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:122-300`

### 6.1 数据批次准备

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:154-184`

经过上述处理后，一个批次的数据结构：

```python
# 第156-162行: 从dataloader加载批次
for batch_dict in self.train_dataloader:
    # 第157行: 转换为DataProto
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    
    # 第158行: 为每个样本添加唯一ID
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
    
    # 第159-162行: 重复样本以支持多次采样
    batch = batch.repeat(
        repeat_times=self.config.actor_rollout_ref.rollout.n,  # n=8
        interleave=True,  # 交错排列 [s1, s1, s1, ..., s2, s2, s2, ...]
    )
    # 现在 batch_size = 64 × 8 = 512

# 第170行: 初始化环境和Agent
self.init_envs_and_agents(batch)
# init_envs_and_agents 定义在第87-120行

# 第182行: 生成Agent轨迹
final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
# generate_agent_trajectory 调用 agent_execution_engine 生成轨迹

# 第183行: 合并批次数据
batch = batch.union(final_gen_batch_output)

# 合并后的batch结构：
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

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:215-217`

```python
# 第215-217行: 使用当前（训练前）的Actor模型计算log概率
if self.use_reference_policy:
    old_log_prob = self.ref_policy_wg.compute_log_prob(batch)
else:
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
# compute_log_prob 在 verl/workers/fsdp_workers.py 或 verl/workers/megatron_workers.py 中实现

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

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:187-190`

```python
# 第187-190行: 使用Critic网络估计状态价值
if self.use_critic:
    with marked_timer("values", timing_raw):
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
# compute_values 在 verl/workers/fsdp_workers.py 或 verl/workers/roles.py 中实现

# Critic模型前向传播
value_logits = critic_model(input_ids, attention_mask)
# value_logits.shape = [512, max_length, 1]

values = value_logits.squeeze(-1)
# values.shape = [512, max_length]

batch.batch["values"] = values
```

### 6.4 Rejection Sampling（可选）

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:226-264`

```python
# 第226-264行: 根据奖励过滤样本（如果启用）
if self.config.algorithm.get("rejection_sampling", {}).get("enable", False):
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

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:192-211` & `verl/trainer/ppo/ray_trainer.py:compute_advantage`

```python
# 第192-211行: 计算优势函数
with marked_timer("adv", timing_raw):
    # 如果使用奖励模型
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
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

### 6.6 PPO Actor更新（权重更新的核心）

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:417-427` & `verl/workers/fsdp_workers.py:update_actor`

这是最关键的部分！让我们详细看看权重是如何更新的：

```python
# ========== 第417-427行: PPO训练的主循环 ==========

# 第417-420行: 更新Critic（如果使用）
if self.use_critic:
    with marked_timer("update_critic", timing_raw):
        critic_output = self.critic_wg.update_critic(batch)
        # critic_wg 是 RayWorkerGroup 实例
        # update_critic 会在多个GPU上并行更新Critic网络
        critic_metrics = reduce_metrics(critic_output)
        metrics.update(critic_metrics)

# 第425-427行: 更新Actor（核心）
with marked_timer("update_actor", timing_raw):
    actor_output = self.actor_rollout_wg.update_actor(batch)
    # actor_rollout_wg 是 RayWorkerGroup 实例
    # update_actor 是权重更新的关键方法
    actor_metrics = reduce_metrics(actor_output)
    metrics.update(actor_metrics)

# ========== update_actor 内部流程 (verl/workers/fsdp_workers.py) ==========
# 这个方法在每个GPU worker上执行

def update_actor(self, data: DataProto):
    """在Actor worker上执行PPO更新"""
    
    # 1. 提取批次数据
    input_ids = data.batch["input_ids"]          # [batch_size, seq_len]
    attention_mask = data.batch["attention_mask"] # [batch_size, seq_len]
    responses = data.batch["responses"]           # [batch_size, response_len]
    old_log_probs = data.batch["old_log_probs"]  # [batch_size, response_len]
    advantages = data.batch["advantages"]         # [batch_size, response_len]
    response_mask = data.batch["response_mask"]  # [batch_size, response_len]
    
    # 2. 分割成mini-batches（如果批次太大）
    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
    ppo_micro_batch_size = self.config.actor_rollout_ref.actor.ppo_micro_batch_size
    num_minibatches = max(1, batch_size // ppo_mini_batch_size)
    
    # 3. 执行多个PPO epochs
    all_metrics = []
    for epoch_idx in range(self.config.actor_rollout_ref.actor.ppo_epochs):  # 通常1-4个epoch
        
        # 打乱数据（可选）
        if self.config.actor_rollout_ref.actor.shuffle_minibatch:
            indices = torch.randperm(batch_size)
        else:
            indices = torch.arange(batch_size)
        
        # 遍历每个mini-batch
        for mb_idx in range(num_minibatches):
            # 4. 获取mini-batch索引
            start_idx = mb_idx * ppo_mini_batch_size
            end_idx = min((mb_idx + 1) * ppo_mini_batch_size, batch_size)
            mb_indices = indices[start_idx:end_idx]
            
            # 5. 提取mini-batch数据
            mb_input_ids = input_ids[mb_indices]
            mb_attention_mask = attention_mask[mb_indices]
            mb_responses = responses[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_response_mask = response_mask[mb_indices]
            
            # ========== 6. 前向传播：计算新的log概率 ==========
            # 这是关键！使用当前（可能已更新的）模型参数
            with torch.enable_grad():  # 确保梯度计算开启
                # 调用Actor模型
                logits = self.actor_model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask
                )  # [mb_size, seq_len, vocab_size]
                
                # 计算log概率
                log_probs_all = F.log_softmax(logits, dim=-1)
                # 提取response部分的log概率
                response_len = mb_responses.shape[1]
                response_logits = logits[:, -response_len:, :]  # [mb_size, response_len, vocab_size]
                
                # 收集实际生成的token的log概率
                new_log_probs = torch.gather(
                    log_probs_all[:, -response_len:, :],
                    dim=2,
                    index=mb_responses.unsqueeze(-1)
                ).squeeze(-1)  # [mb_size, response_len]
            
            # ========== 7. 计算PPO损失 ==========
            # 7.1 计算重要性采样比率
            ratio = torch.exp(new_log_probs - mb_old_log_probs)  # [mb_size, response_len]
            
            # 7.2 计算未裁剪的目标
            surr1 = ratio * mb_advantages
            
            # 7.3 计算裁剪后的目标
            clip_range = self.config.actor_rollout_ref.actor.clip_range  # 通常0.2或0.28
            ratio_clipped = torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range
            )
            surr2 = ratio_clipped * mb_advantages
            
            # 7.4 取两者最小值（PPO的核心）
            policy_loss = -torch.min(surr1, surr2)  # [mb_size, response_len]
            
            # 7.5 应用response_mask（只对模型生成的token计算loss）
            policy_loss = policy_loss * mb_response_mask
            
            # 7.6 聚合损失
            # loss_agg_mode 通常是 "seq-mean-token-sum"
            # 先对每个序列的token求和，再对序列求平均
            policy_loss = policy_loss.sum(dim=1).mean()  # scalar
            
            # 7.7 添加熵正则化（可选，鼓励探索）
            if self.config.actor_rollout_ref.actor.entropy_coeff > 0:
                # 计算熵
                probs = F.softmax(response_logits, dim=-1)
                log_probs = F.log_softmax(response_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)  # [mb_size, response_len]
                entropy = (entropy * mb_response_mask).sum(dim=1).mean()
                
                # 添加到损失（负号因为我们要最大化熵）
                policy_loss = policy_loss - self.config.actor_rollout_ref.actor.entropy_coeff * entropy
            
            # ========== 8. 反向传播和权重更新 ==========
            # 8.1 清空梯度
            self.optimizer.zero_grad()
            
            # 8.2 计算梯度
            policy_loss.backward()
            # 此时所有参数的.grad属性被填充
            
            # 8.3 梯度裁剪（防止梯度爆炸）
            if self.config.actor_rollout_ref.actor.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor_model.parameters(),
                    self.config.actor_rollout_ref.actor.max_grad_norm
                )
            
            # 8.4 更新权重（这是真正改变模型参数的地方！）
            self.optimizer.step()
            # optimizer.step() 内部执行：
            # for param in model.parameters():
            #     param.data = param.data - learning_rate * param.grad
            
            # 8.5 更新学习率（如果使用scheduler）
            if self.scheduler is not None:
                self.scheduler.step()
            
            # ========== 9. 记录指标 ==========
            with torch.no_grad():
                # 计算KL散度（衡量策略变化）
                kl = (mb_old_log_probs - new_log_probs).mean()
                
                # 计算裁剪比例（衡量有多少比率被裁剪了）
                clip_frac = ((ratio < 1 - clip_range) | (ratio > 1 + clip_range)).float().mean()
                
                # 记录指标
                mb_metrics = {
                    "actor/policy_loss": policy_loss.item(),
                    "actor/approx_kl": kl.item(),
                    "actor/clip_frac": clip_frac.item(),
                    "actor/ratio_mean": ratio.mean().item(),
                    "actor/ratio_std": ratio.std().item(),
                    "actor/advantages_mean": mb_advantages.mean().item(),
                    "actor/advantages_std": mb_advantages.std().item(),
                }
                all_metrics.append(mb_metrics)
    
    # ========== 10. 聚合所有mini-batch和epoch的指标 ==========
    final_metrics = {}
    for key in all_metrics[0].keys():
        final_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return DataProto.from_dict({"metrics": final_metrics})

# ========== 权重更新后的状态变化 ==========
# 
# 更新前模型状态（以某一层为例）:
# - actor_model.transformer.h[0].attn.q_proj.weight[0, :5]:
#   tensor([0.0234, -0.0156, 0.0089, -0.0123, 0.0201])
# 
# 执行 optimizer.step() 后：
# - actor_model.transformer.h[0].attn.q_proj.weight[0, :5]:
#   tensor([0.0235, -0.0155, 0.0088, -0.0124, 0.0202])
#   # 注意：参数发生了微小变化（通常在1e-4到1e-6量级）
# 
# 这些微小的变化累积起来，使得模型逐渐学会：
# 1. 更准确地判断何时需要调用工具
# 2. 更好地构造搜索查询
# 3. 更有效地综合搜索结果
# 4. 更准确地生成最终答案
```

### 6.7 完整批次状态跟踪（从生成到更新）

让我们跟踪一个完整批次的状态变化：

```python
# ========== 时刻T0: 批次开始 ==========
# 配置：
# - train_batch_size = 64（64个不同问题）
# - rollout.n = 8（每个问题采样8次）
# - total_trajectories = 512

# ========== 时刻T1: 数据加载 ==========
# agent_ppo_trainer.py:156-162
batch_dict = next(train_dataloader)
# batch_dict = {
#     "prompt": [[{"role": "user", "content": "placeholder"}], ...],  # 64个
#     "extra_info": [{"question": "...", "answer": "...", ...}, ...]  # 64个
# }
batch = DataProto.from_single_dict(batch_dict)
batch.non_tensor_batch["uid"] = [uuid1, uuid2, ..., uuid64]
batch = batch.repeat(repeat_times=8, interleave=True)
# 现在 batch_size = 512
# batch.non_tensor_batch["uid"] = [uuid1, uuid1, uuid1, ..., uuid64, uuid64, uuid64]

# ========== 时刻T2: 环境和Agent创建 ==========
# agent_ppo_trainer.py:170
self.init_envs_and_agents(batch)
# 创建了 512 个 (Agent, Environment) 对

# ========== 时刻T3: 轨迹生成开始 ==========
# agent_ppo_trainer.py:182
final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(...)
# 内部调用 agent_execution_engine.trajectory_generator()
# 512个轨迹并发执行，每个轨迹最多20步

# ========== 时刻T4: 第一条轨迹完成 ==========
# 耗时：约5-10秒（取决于步数和检索速度）
trajectory_001 = {
    "prompt_tokens": [151644, 8948, ...],     # 长度: 400
    "response_tokens": [40, 1184, ...],        # 长度: 600
    "response_masks": [1, 1, ..., 0, 0, ...],  # 长度: 600
    "trajectory_reward": 1.0,                   # 答对了
    "idx": 0,
    "metrics": {
        "steps": 3,
        "total_time": 7.2,
        "llm_time": 4.5,
        "env_time": 2.7
    }
}

# ========== 时刻T5: 所有512条轨迹完成 ==========
# 耗时：约10-20秒（因为是并发的）
# 生成的数据：
all_trajectories = [trajectory_001, trajectory_002, ..., trajectory_512]

# 转换为batch格式：
batch = DataProto(
    batch={
        "input_ids": torch.tensor([...]),      # [512, 1200] (400+600+padding)
        "attention_mask": torch.tensor([...]), # [512, 1200]
        "response_mask": torch.tensor([...]),  # [512, 1200]
        "responses": torch.tensor([...]),      # [512, 600]
        "token_level_scores": torch.tensor([...]),  # [512, 1200]
    },
    non_tensor_batch={
        "uid": [uuid1, uuid1, ..., uuid64, uuid64],  # 512个
        "extra_info": [...],  # 512个
    }
)

# 统计：
# - 总轨迹数: 512
# - 答对的轨迹数: 230 (45%)
# - 答错的轨迹数: 282 (55%)
# - 平均步数: 4.5
# - 平均奖励: 0.45

# ========== 时刻T6: 计算Values ==========
# agent_ppo_trainer.py:187-190 (耗时: 约2-3秒)
values = self.critic_wg.compute_values(batch)
batch.batch["values"] = values  # [512, 1200]

# ========== 时刻T7: Rejection Sampling ==========
# agent_ppo_trainer.py:226-264
# 分析每个uid的奖励分布：
uid_analysis = {
    uuid1: {
        "trajectories": 8,
        "rewards": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "status": "solve_all",  # 全部答对，过滤掉
        "keep": False
    },
    uuid2: {
        "trajectories": 8,
        "rewards": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "status": "solve_partial",  # 部分答对，保留
        "keep": True
    },
    uuid3: {
        "trajectories": 8,
        "rewards": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "status": "solve_none",  # 全部答错，过滤掉
        "keep": False
    },
    # ...
}

# Rejection Sampling结果：
# - solve_all: 15个uid (15×8=120条轨迹) -> 过滤
# - solve_none: 25个uid (25×8=200条轨迹) -> 过滤
# - solve_partial: 24个uid (24×8=192条轨迹) -> 保留

# 过滤后batch_size: 192
batch = batch[valid_mask]

# ========== 时刻T8: 计算Old Log Probs ==========
# agent_ppo_trainer.py:305-319 (耗时: 约2-3秒)
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch.batch["old_log_probs"] = old_log_prob.batch["old_log_probs"]  # [192, 1200]

# ========== 时刻T9: 计算Advantages ==========
# agent_ppo_trainer.py:385-393 (耗时: <1秒)
batch = compute_advantage(
    batch,
    gamma=0.99,
    lam=0.95,
    num_repeat=8
)
batch.batch["advantages"] = ...  # [192, 1200]

# ========== 时刻T10: Actor更新开始 ==========
# agent_ppo_trainer.py:425-427
# 批次分割：
# - batch_size = 192
# - ppo_mini_batch_size = 32
# - num_minibatches = 192 // 32 = 6
# - ppo_epochs = 2
# 
# 总共会执行: 6 minibatches × 2 epochs = 12 次梯度更新

# 更新过程（每个minibatch）：
for epoch in range(2):
    for mb_idx in range(6):
        # 提取minibatch (32条轨迹)
        mb_batch = batch[mb_idx*32:(mb_idx+1)*32]
        
        # 前向传播
        new_log_probs = actor_model(mb_batch)  # 耗时: 约0.5秒
        
        # 计算loss
        policy_loss = compute_ppo_loss(new_log_probs, mb_batch)
        
        # 反向传播
        optimizer.zero_grad()
        policy_loss.backward()  # 耗时: 约0.5秒
        optimizer.step()        # 耗时: 约0.1秒
        
        # 模型参数已更新！

# Actor更新总耗时: 约12-15秒

# ========== 时刻T11: 批次完成 ==========
# 记录指标
metrics = {
    "batch/solve_none": 25,
    "batch/solve_all": 15,
    "batch/solve_partial": 24,
    "critic/full-score/mean": 0.45,
    "actor/policy_loss": 0.23,
    "actor/approx_kl": 0.012,
    "actor/entropy": 2.34,
    "trajectory/avg_steps": 4.5,
    "timing/generation_time": 15.2,
    "timing/training_time": 18.5,
}

# 保存checkpoint（每200步）
if global_steps % 200 == 0:
    save_checkpoint(model, optimizer, global_steps)

# ========== 时刻T12: 下一个批次开始 ==========
# global_steps += 1
# 重复上述过程...
```

### 6.8 Critic更新（如果使用）

**代码位置**: `rllm/trainer/verl/agent_ppo_trainer.py:417-420` & `verl/workers/fsdp_workers.py:update_critic`

```python
# 第417-420行: Critic更新（在Actor更新之前）
if self.use_critic:
    with marked_timer("update_critic", timing_raw):
        critic_output = self.critic_wg.update_critic(batch)
        critic_metrics = reduce_metrics(critic_output)
        metrics.update(critic_metrics)

# update_critic 内部流程（类似update_actor）
for ppo_epoch in range(num_ppo_epochs):
    for mini_batch in split_batch(batch, mini_batch_size=32):
        # 1. 计算新的value估计
        new_values = critic_model(mini_batch)
        
        # 2. 计算value目标（return）
        # 使用TD(λ)或MC return
        returns = compute_returns(
            rewards=mini_batch["token_level_scores"],
            values=new_values,
            gamma=0.99
        )
        
        # 3. Value loss (MSE)
        value_loss = (new_values - returns) ** 2
        value_loss = value_loss * mini_batch["response_mask"]
        value_loss = value_loss.sum(dim=1).mean()
        
        # 4. 反向传播
        critic_optimizer.zero_grad()
        value_loss.backward()
        
        # 5. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
        
        # 6. 更新权重
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

## 十二、关键代码位置快速参考

### 12.1 训练入口和配置
- **训练脚本**: `rllm/examples/search/train_search_agent.py`
- **训练器**: `rllm/trainer/agent_trainer.py:68-90` (`AgentTrainer.train`)
- **Ray任务运行器**: `rllm/trainer/verl/train_agent_ppo.py:54-213` (`TaskRunner.run`)
- **PPO训练器**: `rllm/trainer/verl/agent_ppo_trainer.py:122-300` (`AgentPPOTrainer.fit_agent`)

### 12.2 Agent和Environment核心组件
- **ToolAgent**: `rllm/agents/tool_agent.py:17-167`
  - `__init__`: 第23-61行
  - `update_from_model`: 第102-151行（解析工具调用）
  - `update_from_env`: 第86-100行（处理环境反馈）
  - `reset`: 第153-156行
- **ToolEnvironment**: `rllm/environments/tools/tool_env.py:12-151`
  - `__init__`: 第17-47行
  - `step`: 第55-109行（执行工具调用）
  - `_execute_tool_calls`: 第111-142行（并发执行工具）

### 12.3 Agent-Environment交互引擎
- **AgentExecutionEngine**: `rllm/engine/agent_execution_engine.py:26-538`
  - `run_agent_trajectory_async`: 第168-408行（单条轨迹异步执行）
  - `get_model_response`: 第120-151行（调用模型生成）
  - `trajectory_generator`: 第420-467行（批量并发生成）
- **Token转换工具**: `rllm/agents/utils.py`
  - `convert_messages_to_tokens_and_masks`: 第38-75行
  - `get_recent_assistant_user_messages`: 第6-35行

### 12.4 推理引擎
- **VerlEngine**: `rllm/engine/rollout/verl_engine.py:9-90`
  - `get_model_response`: 第42-83行（vLLM异步推理）
  - Token编码: 第56-58行
  - 异步生成: 第63行
  - Token解码: 第70行

### 12.5 奖励计算
- **search_reward_fn**: `rllm/rewards/reward_fn.py:62-81`
- **RewardSearchFn**: `rllm/rewards/search_reward.py:9-254`
  - `__call__`: 第233-254行（主入口）
  - `extract_answer_from_response`: 第58-183行（答案提取）
  - `unbox`: 第69-85行（提取\\boxed{}内容）
  - `normalize_answer`: 第13-29行（答案标准化）
  - `exact_match_score`: 第54-56行（精确匹配）
  - `f1_score`: 第31-52行（F1计算）
  - `evaluate_answer`: 第185-231行（综合评估）

### 12.6 PPO训练流程
- **批次准备**: `agent_ppo_trainer.py:154-184`
  - 加载批次: 第156-162行
  - 初始化环境: 第170行
  - 生成轨迹: 第182行
- **计算训练量**: `agent_ppo_trainer.py:187-224`
  - Values计算: 第187-190行
  - Advantages计算: 第192-211行
  - Old log probs: 第215-217行
- **Rejection Sampling**: `agent_ppo_trainer.py:226-264`
- **PPO更新**: `agent_ppo_trainer.py:270-295`

### 12.7 工具相关
- **MultiTool**: `rllm/tools/multi_tool.py`（工具管理器）
- **ToolParser**: `rllm/parser/`（工具调用解析器）
- **ChatTemplateParser**: `rllm/parser/`（对话模板解析）

### 12.8 数据流动路径
```
train_search_agent.py (入口)
    ↓
AgentTrainer.train() (trainer/agent_trainer.py:68)
    ↓
TaskRunner.run() (trainer/verl/train_agent_ppo.py:62)
    ↓
AgentPPOTrainer.fit_agent() (trainer/verl/agent_ppo_trainer.py:122)
    ↓
AgentExecutionEngine.trajectory_generator() (engine/agent_execution_engine.py:420)
    ↓ (并发512条轨迹)
AgentExecutionEngine.run_agent_trajectory_async() (engine/agent_execution_engine.py:168)
    ↓ (Agent-Env交互循环)
    ├─ Agent.reset() (agents/tool_agent.py:153)
    ├─ Environment.reset() (environments/tools/tool_env.py:49)
    ├─ [循环 max_steps=20 次]
    │   ├─ VerlEngine.get_model_response() (engine/rollout/verl_engine.py:42)
    │   ├─ Agent.update_from_model() (agents/tool_agent.py:102)
    │   ├─ Environment.step() (environments/tools/tool_env.py:55)
    │   └─ Agent.update_from_env() (agents/tool_agent.py:86)
    └─ RewardSearchFn.__call__() (rewards/search_reward.py:233)
    ↓
convert_messages_to_tokens_and_masks() (agents/utils.py:38)
    ↓
[返回 DataProto with tokens]
    ↓
compute_advantages() (verl/trainer/ppo/ray_trainer.py)
    ↓
Actor.update() & Critic.update() (verl/workers/)
```

## 十三、总结与关键洞察

### 13.1 训练流程核心

这个训练流程的核心是将**多步推理问题**转化为**序列生成问题**，通过PPO算法优化模型生成能够正确使用工具并得出正确答案的轨迹。

### 13.2 关键创新点

1. **Agent-Environment范式**: 将推理过程建模为Agent与环境的交互
   - Agent: 负责生成思考和工具调用
   - Environment: 负责执行工具并返回结果
   - 清晰的职责分离，易于扩展

2. **工具调用机制**: 模型学习何时调用工具、如何构造查询
   - 使用结构化的工具调用格式（JSON）
   - 支持多种工具（检索、计算、API调用等）
   - 并发执行工具调用，提升效率

3. **稀疏奖励**: 只在最终答案处给奖励，鼓励模型完成完整推理链
   - 中间步骤reward=0，只在finish时计算奖励
   - 使用EM和F1评估答案质量
   - 奖励分配到最后一个模型生成的token

4. **Token级训练**: 将对话历史转换为token序列，利用标准的PPO算法
   - response_mask区分模型生成和环境返回的token
   - 只对模型生成的token计算loss
   - 支持任意长度的多步推理

5. **异步并发**: 512条轨迹异步并发生成，大幅提升训练效率
   - 使用asyncio实现高并发
   - vLLM动态批处理多个请求
   - 工具调用通过线程池并发

6. **Rejection Sampling**: 智能过滤训练样本
   - 过滤全对或全错的样本组（信息量低）
   - 保留部分对部分错的样本（信息量高）
   - 提升训练效率和模型性能

### 13.3 权重更新机制详解

PPO算法通过以下步骤更新模型权重：

1. **采样**: 使用当前策略生成512条轨迹
2. **评估**: 计算每条轨迹的奖励
3. **优势计算**: 使用GAE计算每个token的优势函数
4. **策略更新**: 
   - 计算重要性采样比率 `ratio = exp(new_log_prob - old_log_prob)`
   - 使用裁剪防止过大的策略更新
   - 最小化 `-min(ratio * A, clip(ratio) * A)`
5. **梯度下降**: 
   - `optimizer.zero_grad()` 清空梯度
   - `loss.backward()` 计算梯度
   - `optimizer.step()` 更新参数
6. **迭代**: 重复多个epoch和mini-batch

### 13.4 时间和资源消耗

一个完整批次（64问题×8采样=512轨迹）的时间分配：

- **轨迹生成**: 10-20秒（并发）
  - LLM推理: 4-8秒
  - 工具执行: 3-7秒
  - 其他: 3-5秒
- **Values计算**: 2-3秒
- **Advantages计算**: <1秒
- **Old log probs计算**: 2-3秒
- **Actor更新**: 12-15秒（12次梯度更新）
- **Critic更新**: 5-8秒（如果使用）

**总耗时**: 约35-50秒/批次

对于15个epoch，假设每个epoch有100个批次：
- 总批次数: 1,500
- 总耗时: 约15-21小时
- 使用2个GPU（CUDA_VISIBLE_DEVICES=3,4）

### 13.5 关键参数调优建议

1. **批次大小（train_batch_size）**: 
   - 较大批次（64-128）提供更稳定的梯度
   - 较小批次（16-32）训练更快但可能不稳定

2. **采样次数（rollout.n）**:
   - 8-16次采样提供足够的多样性
   - 过多采样增加计算成本

3. **裁剪范围（clip_range）**:
   - 0.2-0.3是常用值
   - 较小值（0.1-0.2）更保守，训练更稳定
   - 较大值（0.3-0.5）更激进，可能更快收敛

4. **学习率**:
   - Actor: 1e-6 到 1e-5
   - Critic: 5e-6 到 5e-5
   - 使用cosine或linear decay

5. **最大步数（max_steps）**:
   - 根据任务复杂度调整（10-30步）
   - 过多步数增加计算成本
   - 过少步数可能无法完成推理

### 13.6 扩展性和应用

这种方法可以扩展到其他需要多步推理和工具使用的任务：

1. **代码生成**: 
   - 工具: 代码执行、文档查询、测试运行
   - 奖励: 测试通过率、代码质量

2. **数学问题求解**:
   - 工具: 符号计算、数值计算、公式查询
   - 奖励: 答案正确性

3. **科学研究辅助**:
   - 工具: 文献检索、数据分析、实验设计
   - 奖励: 研究质量评估

4. **复杂决策任务**:
   - 工具: 信息查询、模拟、专家咨询
   - 奖励: 决策质量评估

### 13.7 未来改进方向

1. **自适应步数**: 根据任务难度动态调整max_steps
2. **层次化工具**: 支持工具的嵌套调用
3. **多模态输入**: 支持图像、表格等输入
4. **在线学习**: 在部署过程中持续学习
5. **元学习**: 快速适应新任务和新工具

---

**文档完成！** 现在您应该对RLLM的训练流程有了深入而全面的理解，包括每一步的代码位置、中间状态变化和权重更新机制。

