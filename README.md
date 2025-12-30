# PDF 图书转 Markdown 工具
## books文件夹里面是转换前和转换后的对比效果
将扫描版 PDF 图书转换为格式良好的 Markdown 文档，支持 OCR 识别和可选的 LLM 文本优化。

## 功能特性

✅ **PDF 转图像**：高质量 DPI 转换  
✅ **OCR 识别**：基于 PaddleOCR-VL 的强大文档解析能力  
✅ **智能分块**：自动分割长文本，保持上下文连贯  
✅ **LLM 优化**（可选）：修正 OCR 错误，优化格式  
✅ **多 API 支持**：OpenAI、Claude、本地 LLM（Ollama 等）  
✅ **页面范围**：支持只转换指定页面  
✅ **中间文件**：可选保存处理过程中的临时文件  

## 快速开始

### 1. 安装依赖

#### 1.1 安装系统依赖（用于 PDF 转图像）

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# 下载 poppler: https://github.com/oschwartz10612/poppler-windows/releases
# 解压并添加 bin 目录到系统 PATH
```

#### 1.2 安装 PaddlePaddle（重要！）

**注意：必须从 PaddlePaddle 官方源安装，不要使用 pip 默认源！**

```bash
# 使用 uv 环境（推荐）
source .venv/bin/activate

# GPU 版本 (CUDA 12.6) - 推荐有 GPU 的用户
uv pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 或 GPU 版本 (CUDA 11.8)
uv pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 或 CPU 版本
uv pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# ⚠️ 安装时请关闭 VPN/梯子，否则可能连接失败
```

#### 1.3 安装其他 Python 依赖

```bash
# 安装 PaddleOCR 和其他依赖
uv pip install "paddleocr[doc-parser]" pdf2image Pillow PyYAML openai anthropic
```

### 2.
