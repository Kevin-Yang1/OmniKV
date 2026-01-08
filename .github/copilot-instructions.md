# AI 代理 OmniKV 代码库指南

## 项目概述
OmniKV 是一种针对大型语言模型 (LLM) 的推理加速方法，它使用**动态上下文选择 (Dynamic Context Selection)**。通过仅将“重要”Token 保留在 GPU 内存中，同时保留其他 Token（可能卸载到 CPU），从而优化 KV 缓存使用，避免永久性信息丢失。

代码库分为：
1.  **Core Modeling**: `modeling/` 目录下的 PyTorch/HuggingFace 实现。
2.  **OmniKV-LightLLM**: `omnikv-lightllm/` 目录下修改后的 LightLLM 服务框架分支。
3.  **Benchmarks**: `benchmark/` 目录下的 LongBench 和 InfiniteBench 测试脚本。
4.  **Baselines**: `baselines/` 目录下的基准实现 (H2O, InfLLM)。

## 架构与数据流

### 核心建模 (`modeling/`)
*   **入口点**: `modeling/omnikv.py` 包含主要逻辑。
*   **Token 选择**: 核心算法运行在 `select_tokens_by_attn_universal` (位于 `omnikv.py`) 中，该函数根据“过滤层 (filter layers)”的注意力分数选择 Token。
*   **配置**:
    *   `modeling/omnikv_config.py`: 扩展 `LlamaConfig` 以添加 OmniKV 特定参数。
    *   `LlamaCompressorConfig`: 用于将配置注入到模型中。
*   **缓存管理**: `modeling/spec_cache.py` 和 `modeling/offload_select_once.py` 管理 KV 缓存，可能将部分卸载到 CPU (`real_offload` 参数)。

### 服务边界
*   **HuggingFace 集成**: 代码对 Llama 模型进行了 Monkey Patch 或包装。它依赖于标准的 HF `transformers` 结构。
*   **LightLLM**: 独立的服务引擎。OmniKV 的更改直接集成在 `omnikv-lightllm/lightllm/` 中。

## 开发者工作流

### 配置
*   **JSON 驱动**: 所有实验都使用位于 `configs/` 中的 JSON 配置文件。
*   **关键参数** (参见 `configs/example.json`):
    *   `do_select_layers`: 用于筛选重要 Token 的层 (例如 "2,8,18")。
    *   `num_of_selected_tokens`: 要保留的 Token 分数或数量。
    *   `real_offload`: 是否启用 CPU 卸载的布尔值。

### 运行评估
不要直接运行 Python 脚本进行基准测试；请使用设置了 `PYTHONPATH` 的 Shell 包装脚本。

*   **LongBench**:
    ```bash
    PYTHONPATH=./ bash shells/eval/eval_any_long.sh configs/example.json <world_size>
    ```
*   **InfiniteBench**:
    ```bash
    PYTHONPATH=./ bash shells/eval/eval_any_inf.sh configs/example.json
    ```

### LightLLM 开发
*   **安装**: 使用 `omnikv-lightllm/` 内部的 `setup.py`。
    ```bash
    cd omnikv-lightllm
    pip install -r requirements.txt
    python setup.py install
    ```

## 约定与模式

### 编码风格
*   **日志**: 使用 `tiny_tools.log.logger` 进行调试，而不是 print 语句。
    ```python
    from tiny_tools.log import logger
    logger.debug("Debug message")
    ```
*   **路径**: 脚本假设 `PYTHONPATH=./` (项目根目录)。尽可能在内部使用相对路径。
*   **配置访问**: 在建模文件中工作时，通过 `self.config` (已 Patch 到 HF config 中) 访问 OmniKV 配置。

### 关键依赖
*   `transformers==4.41.2`: 严格的版本要求，以避免 Patch 兼容性问题。
*   `flash-attn`: 性能所需。

## 常见任务

### 添加新模型支持
检查 `modeling/patch_of_llama3_1.py` 以获取如何 Patch RoPE 或其他模型细节的示例。通常需要确保标准 HF 模型 `forward` 方法与 `OmniKV` 缓存包装器兼容。

### 调试 Token 选择
Token 选择逻辑是核心。在 `select_tokens_by_attn_universal` (位于 `omnikv.py`) 中跟踪 `idx` 和 `v` 变量，以查看保留了哪些 Token。
