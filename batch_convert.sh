#!/bin/bash
# 批量转换脚本示例

# 设置输入输出目录
INPUT_DIR="books"
OUTPUT_DIR="output"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 遍历所有 PDF 文件
for pdf in "$INPUT_DIR"/*.pdf; do
    if [ -f "$pdf" ]; then
        echo "========================================"
        echo "正在处理: $(basename "$pdf")"
        echo "========================================"
        
        # 提取文件名（不含扩展名）
        filename=$(basename "$pdf" .pdf)
        
        # 转换 PDF
        python pdf_to_markdown.py "$pdf" \
            -o "$OUTPUT_DIR/${filename}.md" \
            -c config.yaml
        
        echo ""
        echo "完成: ${filename}.md"
        echo ""
    fi
done

echo "========================================"
echo "批量转换完成！"
echo "========================================"
