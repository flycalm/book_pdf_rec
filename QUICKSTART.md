# PDF 图书转 Markdown 工具 - 快速开始

## ⚡ 三步上手

### 1️⃣ 安装依赖

```bash
# 安装系统依赖
sudo apt-get install poppler-utils

# 激活 uv 环境
source .venv/bin/activate

# 安装 PaddlePaddle（关闭 VPN！）
uv pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 安装其他依赖
uv pip install "paddleocr[doc-parser]" pdf2image Pillow PyYAML openai anthropic
```

### 2️⃣ 配置 config.yaml

```yaml
# 复制示例配置
cp config.example.yaml config.yaml

# 编辑配置文件
nano config.yaml
```

**最简配置（不用 LLM）：**
```yaml
pdf:
  page_range: [1, 20]  # 只转换前 20 页（测试用）

llm:
  enabled: false  # 不使用 LLM
```

**完整配置（使用 DeepSeek）：**
```yaml
pdf:
  dpi: 300
  page_range: null  # 转换全部页面

llm:
  enabled: true
  provider: openai
  api_key: "sk-xxx"  # 替换为你的 API Key
  api_base: "https://api.deepseek.com"
  model: "deepseek-chat"
  max_workers: 5  # 并行处理 5 个块
```

### 3️⃣ 运行转换

```bash
# 基本使用（根据 config.yaml 配置运行）
python pdf_to_markdown.py books/your_book.pdf

# 就这么简单！
```

## 📋 配置说明

所有配置都在 `config.yaml` 中设置，无需命令行参数！

### PDF 处理配置

```yaml
pdf:
  dpi: 300  # 图像质量
            # 150 = 快速
            # 300 = 推荐
            # 600 = 高质量
  
  page_range: [1, 50]  # 页面范围
              # null = 全部页面
              # [1, 20] = 第 1-20 页
```

### LLM 配置

```yaml
llm:
  enabled: true  # 是否启用
  
  # DeepSeek 配置
  provider: openai
  api_key: "sk-xxx"
  api_base: "https://api.deepseek.com"
  model: "deepseek-chat"
  
  # 性能调优
  chunk_size: 2000  # 每块大小
  chunk_overlap: 200  # 重叠大小
  max_workers: 5  # 并行数（1-10）
```

## 🎯 常用场景

### 场景 1：快速测试（5 页，不用 LLM）

```yaml
pdf:
  page_range: [1, 5]
llm:
  enabled: false
```

```bash
python pdf_to_markdown.py test.pdf
```

### 场景 2：完整转换 + LLM 优化

```yaml
pdf:
  page_range: null  # 全部
llm:
  enabled: true
  api_key: "sk-xxx"
  model: "deepseek-chat"
  max_workers: 5
```

```bash
python pdf_to_markdown.py book.pdf
```

### 场景 3：高质量扫描

```yaml
pdf:
  dpi: 600  # 高分辨率
  page_range: [1, 100]
llm:
  enabled: true
  chunk_size: 3000
  max_workers: 3  # 降低并发
```

## 📁 输出文件

```
output/
├── your_book.md              # 最终输出
└── temp_20241230_123456/     # 中间文件（可选）
    ├── temp_page_1.png
    ├── page_1_raw.md
    └── ...
```

## ⚙️ 高级选项

### 命令行临时覆盖

```bash
# 临时指定页面范围（覆盖配置文件）
python pdf_to_markdown.py book.pdf --pages 10-20

# 指定输出文件
python pdf_to_markdown.py book.pdf -o my_output.md

# 使用不同的配置文件
python pdf_to_markdown.py book.pdf -c test_config.yaml
```

### 批量转换

```bash
# 使用批量脚本
./batch_convert.sh
```

## 🔧 性能优化

### 提升速度

1. **降低 DPI**：`dpi: 200`（牺牲质量）
2. **增加并发**：`max_workers: 10`（注意 API 限流）
3. **关闭中间文件**：`save_intermediate: false`

### 提升质量

1. **提高 DPI**：`dpi: 600`
2. **启用 LLM**：`enabled: true`
3. **使用更好的模型**：`model: gpt-4`

### 节省费用

1. **使用 DeepSeek**：比 GPT-4 便宜 10 倍
2. **减小块大小**：`chunk_size: 1500`
3. **降低并发**：`max_workers: 2`

## ❓ 常见问题

**Q: 首次运行很慢？**  
A: 正在下载 PaddleOCR 模型（约 2GB），之后会快很多。

**Q: 如何只转换部分页面？**  
A: 在 `config.yaml` 中设置 `page_range: [起始, 结束]`

**Q: LLM 并行处理出错？**  
A: 降低 `max_workers` 值，可能是 API 限流。

**Q: 跨页文本断开了？**  
A: 程序会自动处理跨页文本，将断句智能连接。

## 📚 完整文档

详细说明请查看 [README.md](README.md)

## 🎉 就是这么简单！

配置好 `config.yaml`，然后：

```bash
python pdf_to_markdown.py your_book.pdf
```

搞定！✨
