# 开发与状态流转规范

本文档定义了 DocFoundry 项目在代码编写、数据流转、状态持久化及文档维护等方面的硬性标准。所有开发者（含 AI 编程助手）必须严格遵守。

## 1. Agent 流转数据格式规范 (State Schema)

系统采用 LangGraph 构建，Agent 之间的所有信息传递必须且只能通过全局状态对象（State）进行。严禁 Agent 之间进行点对点的直接变量传递。

### 状态对象定义
必须使用 Python 的 TypedDict 或 Pydantic BaseModel 进行强类型校验。

### 大文件"传址不传值"原则

**禁止**将大量原始数据（如 DataFrame、CSV 文本、图片二进制流）直接塞入 State。

**必须**将数据和图片落盘到本地，在 State 中只流转文件的相对路径（String）。

### 核心状态字段标准

```python
class SystemState(TypedDict):
    # 1. 元数据
    thread_id: str           # 任务唯一标识 (如: "user_center_thesis_v1")
    topic: str               # 当前调研/论文主题

    # 2. 数据与分析载体 (传址)
    raw_data_path: str       # 生成数据的 CSV 本地相对路径
    plot_image_paths: list   # 生成的图表路径列表 (支持多图)

    # 3. 洞察与内容 (传值)
    analysis_insights: dict  # JSON 格式的深度分析结论 (必须包含 key: metric, conclusion, anomaly)
    thesis_draft: str        # Markdown 格式的论文/报告正文

    # 4. 异常与控制
    error_logs: list         # Agent 运行时的 Traceback 报错记录栈
    current_step: str        # 记录当前执行到的节点名称
```

## 2. Checkpoint 保存与恢复格式 (Persistence)

长链路的学术撰写或数据分析（尤其是涉及机器学习聚类或大量 API 调用时）极易中断。系统必须实现 Checkpoint 机制，支持断点续传。

### 底层机制
强制使用 LangGraph 的 MemorySaver（开发期）或 SQLite/Postgres Checkpointer（生产期）作为状态持久化工具。

### Thread ID 命名规范
触发任务时，必须分配明确的 thread_id，格式为 `<模块名称>_<任务类型>_<时间戳>`（例如：`user_center_module_auth_analysis_1710748800`）。

### 人工介入点 (Human-in-the-loop)

在进入 writer_agent 之前，必须设置一个中断点（Interrupt）。

格式要求：系统在此处暂停，将当前的 analysis_insights（JSON 格式）和图表路径输出到控制台，等待人类确认或修改 JSON 结论后，再继续生成最终文档。

## 3. 项目结构与路径规范 (Project Structure)

为了保证代码在不同环境下的可执行性，对目录结构和路径操作做以下限制。

### 路径寻址规范

**严禁**在代码中使用绝对路径（如 `C:/Users/...` 或 `/Users/mac/...`）。

**所有**文件读写操作，必须基于项目根目录，使用 `os.path.join` 或 `pathlib.Path` 构建相对路径。

### 结构划分不可逾越

- **src/agents/**：只允许存放 Agent 的定义、Prompt 组装和节点函数逻辑。严禁在此写具体的业务执行代码（如直接写 pandas 操作）。
- **src/tools/**：存放所有原子化工具（如 Python 沙箱、数据库查询）。工具必须是纯函数，且必须包含详细的异常捕获。
- **data/**：
  - **data/raw_data/**：所有生成的模拟数据必须存入此目录
  - **data/processed/**：清洗或处理后的中间表存入此目录
  - **data/output/**：最终输出的论文/报告存入此目录

## 4. 文档更新与编写规范 (Documentation)

项目文档是系统的最高指导原则，必须与代码保持绝对同步。

### Prompt 即代码 (Prompt as Code)

任何对 Agent 角色能力、输出格式的修改，必须首先更新 `docs/ROLES.md`。

代码中的 System Prompt 必须通过读取配置文件或清晰的常量定义，其内容必须与 `ROLES.md` 保持 100% 一致。

### 函数注释标准

所有放置在 `src/tools/` 下的函数，必须包含标准的 Google Style Docstring。因为大模型会读取这些注释来决定如何调用工具。

Docstring 必须包含：
- 功能描述
- 参数类型说明
- 返回值说明
- 可能抛出的异常类型

示例：
```python
def expand_data(persona_json_path: str, output_csv_path: str, num_samples: int = 5000) -> Dict[str, Any]:
    """
    读取用户画像比例，注入噪声并批量扩增数据。
    
    Args:
        persona_json_path: 用户画像 JSON 文件路径
        output_csv_path: 输出 CSV 文件路径
        num_samples: 生成的样本数量，默认为 5000
        
    Returns:
        包含生成结果的字典，格式为 {
            "output_path": str,  # 生成的 CSV 文件路径
            "num_samples": int,  # 实际生成的样本数
            "success": bool      # 是否成功
        }
        
    Raises:
        FileNotFoundError: 如果画像文件不存在
        ValueError: 如果 JSON 格式不正确
    """
```

### 输出物规范

最终由 writer_agent 编写的论文/报告，必须以标准的 Markdown 格式输出到 `data/output/` 目录下。

文件名必须包含时间戳和 Thread ID，例如：`thesis_user_center_v1_20260318.md`。

## 5. 错误处理与日志规范

### 异常捕获
- 所有 Agent 节点必须包含 try-except 块
- 捕获的异常信息必须记录到 `state["error_logs"]`
- 遇到致命错误时，将 `state["current_step"]` 设置为 "error"

### 日志记录
- 使用 Python 的 logging 模块
- 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
- 日志格式：`[时间戳] [级别] [模块名] 消息内容`

## 6. Git 提交规范

### Commit Message 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具链相关

### 示例
```
feat(agents): 添加 survey_agent 节点

实现了调研问卷设计专家 Agent，支持生成结构化问卷 JSON。

- 添加了 survey_agent.py 文件
- 更新了 ROLES.md 文档
- 在 graph.py 中注册了新节点

Closes #1
```

## 7. 性能优化规范

### API 调用优化
- 批量处理请求，减少 API 调用次数
- 使用缓存机制，避免重复调用
- 设置合理的超时时间

### 数据处理优化
- 使用 pandas 的向量化操作
- 避免在循环中操作 DataFrame
- 大数据集使用分块处理

### 内存管理
- 及时释放不再使用的大对象
- 使用生成器处理流式数据
- 监控内存使用情况
