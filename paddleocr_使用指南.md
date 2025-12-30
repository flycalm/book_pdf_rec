# PaddleOCR-VL 模型调用指南

## 简介

PaddleOCR-VL 是一个强大的文档解析视觉语言模型，支持：
- **OCR 文字识别**：识别 109 种语言的文本
- **表格识别**：提取表格结构和内容
- **图表解析**：理解各类图表信息
- **公式识别**：转换数学公式
- **文档解析**：完整的页面级文档理解

## 环境安装

```bash
# 安装 PaddlePaddle (GPU 版本)
python -m pip install paddlepaddle-gpu==3.2.0

# 安装 PaddleOCR 及文档解析依赖
python -m pip install -U "paddleocr[doc-parser]"
```

## 调用方式

### 方式一：命令行调用（最简单）

```bash
# 处理单张图片
paddleocr doc_parser -i /path/to/your/image.jpg

# 处理网络图片
paddleocr doc_parser -i https://example.com/image.png

# 指定输出目录
paddleocr doc_parser -i image.jpg -o output/
```

### 方式二：Python API 调用（推荐）

```python
from paddleocr import PaddleOCRVL

# 初始化模型（首次运行会自动下载）
pipeline = PaddleOCRVL()

# 处理本地图片
output = pipeline.predict("/path/to/image.jpg")

# 处理网络图片
output = pipeline.predict("https://example.com/image.jpg")

# 输出结果
for res in output:
    # 打印到控制台
    res.print()

    # 保存为 JSON 文件
    res.save_to_json(save_path="output")

    # 保存为 Markdown 文件
    res.save_to_markdown(save_path="output")
```

### 方式三：批量处理

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()

# 批量处理多张图片
image_list = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

for img_path in image_list:
    output = pipeline.predict(img_path)
    for res in output:
        res.save_to_markdown(save_path="output")
```

## 输出结果格式

模型返回三种格式的结果：

1. **控制台输出**：直接打印识别结果
2. **JSON 格式**：结构化的数据，便于程序处理
3. **Markdown 格式**：可读性强的文档格式

## 常见问题

**Q: 第一次运行很慢？**
A: 首次运行会自动下载模型文件（约几GB），之后会使用缓存，速度会快很多。

**Q: 支持哪些图片格式？**
A: 支持 JPG、PNG、PDF 等常见格式。

**Q: 如何提高处理速度？**
A: 可以使用 GPU 加速，确保安装了 GPU 版本的 PaddlePaddle。

**Q: 内存不足怎么办？**
A: 可以尝试降低图片分辨率或分批处理大量图片。

## 更多信息

- 项目主页：PaddlePaddle/PaddleOCR-VL
- 完整文档：参见项目 README.md
