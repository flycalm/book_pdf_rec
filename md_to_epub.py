#!/usr/bin/env python3
"""
Markdown to Kindle-optimized EPUB converter
将 Markdown 文件转换为适配 Kindle 的 EPUB 格式
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple
import argparse

try:
    import markdown
    from ebooklib import epub
except ImportError:
    print("错误: 缺少必要的库")
    print("请运行: pip install ebooklib markdown")
    exit(1)


class MarkdownToEpubConverter:
    """Markdown 到 Kindle 优化 EPUB 转换器"""

    def __init__(
        self,
        md_file: str,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        language: str = "zh-CN",
        cover_image: Optional[str] = None
    ):
        """
        初始化转换器

        Args:
            md_file: Markdown 文件路径
            output_file: 输出 EPUB 文件路径
            title: 书籍标题
            author: 作者名称
            language: 语言代码
            cover_image: 封面图片路径
        """
        self.md_file = Path(md_file)
        self.output_file = Path(output_file) if output_file else self._default_output_path()
        self.title = title or self.md_file.stem
        self.author = author or "Unknown"
        self.language = language
        self.cover_image = cover_image

        # 创建 EPUB 书籍
        self.book = epub.EpubBook()

        # 存储章节
        self.chapters = []
        self.toc = []

    def _default_output_path(self) -> Path:
        """生成默认输出路径"""
        return self.md_file.parent / f"{self.md_file.stem}.epub"

    def _set_metadata(self):
        """设置书籍元数据"""
        self.book.set_identifier(str(self.md_file))
        self.book.set_title(self.title)
        self.book.set_language(self.language)
        self.book.add_author(self.author)

        # 添加 Kindle 优化的 CSS 样式
        css_content = self._get_kindle_css()
        nav_css = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=css_content
        )
        self.book.add_item(nav_css)

    def _get_kindle_css(self) -> str:
        """生成 Kindle 优化的 CSS 样式"""
        return """
        /* Kindle 优化样式 */
        @charset "UTF-8";

        body {
            margin: 1em;
            padding: 0;
            font-family: "PMingLiU", "MingLiU", "STSong", "SimSun", serif;
            font-size: 1.1em;
            line-height: 1.8;
            text-align: justify;
            color: #000000;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "PMingLiU", "MingLiU", "STSong", "SimSun", serif;
            font-weight: bold;
            margin: 1.5em 0 0.8em 0;
            page-break-before: always;
        }

        h1 {
            font-size: 1.8em;
            text-align: center;
            margin-top: 2em;
        }

        h2 {
            font-size: 1.5em;
            page-break-before: always;
        }

        h3 {
            font-size: 1.3em;
            page-break-before: avoid;
        }

        h4 {
            font-size: 1.2em;
            page-break-before: avoid;
        }

        p {
            margin: 0.8em 0;
            text-indent: 2em;
            line-height: 1.8;
        }

        /* 列表 */
        ul, ol {
            margin: 1em 0;
            padding-left: 2em;
        }

        li {
            margin: 0.5em 0;
            line-height: 1.6;
        }

        /* 代码块 */
        pre {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 1em;
            margin: 1em 0;
            overflow-x: auto;
            font-family: "Courier New", monospace;
            font-size: 0.9em;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            page-break-inside: avoid;
        }

        code {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 0.2em 0.4em;
            font-family: "Courier New", monospace;
            font-size: 0.9em;
        }

        pre code {
            background-color: transparent;
            border: none;
            padding: 0;
        }

        /* 引用 */
        blockquote {
            margin: 1em 2em;
            padding: 0.5em 1em;
            border-left: 4px solid #ccc;
            background-color: #f9f9f9;
            font-style: italic;
            page-break-inside: avoid;
        }

        /* 表格 */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            page-break-inside: avoid;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 0.5em 0.8em;
            text-align: left;
        }

        th {
            background-color: #f0f0f0;
            font-weight: bold;
        }

        /* 图片 */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
            page-break-inside: avoid;
        }

        /* 链接 */
        a {
            color: #0000FF;
            text-decoration: underline;
        }

        /* 分隔线 */
        hr {
            border: none;
            border-top: 1px solid #ccc;
            margin: 2em 0;
            page-break-after: avoid;
        }

        /* 避免孤行和寡行 */
        p, li, td, th {
            orphans: 3;
            widows: 3;
        }

        /* 标题后的段落不要首行缩进 */
        h1 + p, h2 + p, h3 + p, h4 + p, h5 + p, h6 + p {
            text-indent: 0;
        }

        /* 列表项不要首行缩进 */
        li p {
            text-indent: 0;
        }
        """

    def _parse_markdown(self) -> str:
        """解析 Markdown 文件"""
        with open(self.md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 转换 Markdown 为 HTML
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'extra',
                'codehilite',
                'toc',
                'tables',
                'fenced_code'
            ]
        )

        return html_content

    def _split_into_chapters(self, html_content: str) -> List[Tuple[str, str]]:
        """
        将 HTML 内容分割为章节
        基于 h1 和 h2 标题分割
        """
        # 查找所有 h1 和 h2 标签
        pattern = r'<h[12][^>]*>(.*?)</h[12]>'
        matches = list(re.finditer(pattern, html_content, re.IGNORECASE))

        if not matches:
            # 如果没有找到标题，整个内容作为一章
            return [("封面", html_content)]

        chapters = []
        for i, match in enumerate(matches):
            title = re.sub(r'<[^>]+>', '', match.group(1))
            start = match.start()

            # 确定结束位置
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(html_content)

            content = html_content[start:end]
            chapters.append((title, content))

        return chapters

    def _add_cover_image(self):
        """添加封面图片"""
        if not self.cover_image or not os.path.exists(self.cover_image):
            return

        with open(self.cover_image, 'rb') as f:
            cover_data = f.read()

        cover_filename = Path(self.cover_image).name
        cover_item = epub.EpubItem(
            uid="cover_img",
            file_name=f"images/{cover_filename}",
            media_type="image/jpeg",
            content=cover_data
        )
        self.book.add_item(cover_item)

        # 设置封面
        self.book.set_cover(f"images/{cover_filename}", cover_data)

    def _create_chapters(self, html_content: str):
        """创建章节"""
        chapters_data = self._split_into_chapters(html_content)

        for i, (title, content) in enumerate(chapters_data):
            # 创建章节文件名
            file_name = f"chap_{i+1:03d}.xhtml"

            # 创建章节
            chapter = epub.EpubHtml(
                title=title,
                file_name=file_name,
                lang=self.language
            )

            # 添加内容（包含 CSS）
            chapter.content = f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE html>
            <html xmlns="http://www.w3.org/1999/xhtml">
            <head>
                <title>{title}</title>
                <link rel="stylesheet" type="text/css" href="style/nav.css"/>
            </head>
            <body>
                {content}
            </body>
            </html>
            """

            chapter.add_item(epub.EpubItem(
                uid="style_nav",
                file_name="style/nav.css",
                media_type="text/css"
            ))

            self.book.add_item(chapter)
            self.chapters.append(chapter)

        # 设置目录
        self.book.toc = tuple(self.chapters)

        # 添加导航文件
        self.book.add_item(epub.EpubNcx())
        self.book.add_item(epub.EpubNav())

    def _create_spine(self):
        """创建书籍阅读顺序"""
        self.book.spine = ['nav'] + self.chapters

    def convert(self) -> str:
        """
        执行转换

        Returns:
            输出文件路径
        """
        print(f"正在转换: {self.md_file}")

        # 设置元数据
        self._set_metadata()

        # 添加封面
        self._add_cover_image()

        # 解析 Markdown
        print("正在解析 Markdown...")
        html_content = self._parse_markdown()

        # 创建章节
        print("正在创建章节...")
        self._create_chapters(html_content)

        # 创建阅读顺序
        self._create_spine()

        # 写入 EPUB 文件
        print(f"正在生成 EPUB 文件: {self.output_file}")
        epub.write_epub(str(self.output_file), self.book, {})

        print(f"转换完成！")
        print(f"输出文件: {self.output_file}")

        return str(self.output_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将 Markdown 文件转换为适配 Kindle 的 EPUB 格式'
    )
    parser.add_argument(
        'md_file',
        help='Markdown 文件路径'
    )
    parser.add_argument(
        '-o', '--output',
        help='输出 EPUB 文件路径（默认与输入文件同名）',
        default=None
    )
    parser.add_argument(
        '-t', '--title',
        help='书籍标题（默认使用文件名）',
        default=None
    )
    parser.add_argument(
        '-a', '--author',
        help='作者名称',
        default='Unknown'
    )
    parser.add_argument(
        '-l', '--language',
        help='语言代码（默认: zh-CN）',
        default='zh-CN'
    )
    parser.add_argument(
        '-c', '--cover',
        help='封面图片路径',
        default=None
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.md_file):
        print(f"错误: 文件不存在: {args.md_file}")
        return 1

    # 检查封面图片
    if args.cover and not os.path.exists(args.cover):
        print(f"警告: 封面图片不存在: {args.cover}")

    # 创建转换器并执行转换
    converter = MarkdownToEpubConverter(
        md_file=args.md_file,
        output_file=args.output,
        title=args.title,
        author=args.author,
        language=args.language,
        cover_image=args.cover
    )

    try:
        output_path = converter.convert()
        return 0
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
