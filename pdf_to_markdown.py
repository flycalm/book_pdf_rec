#!/usr/bin/env python3
"""
PDF 图书转 Markdown 工具
支持使用 PaddleOCR-VL 进行 OCR 识别，并可选使用 LLM 进行文本优化
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from pdf2image import convert_from_path
    from PIL import Image
    from paddleocr import PaddleOCRVL
    from openai import OpenAI
    import anthropic
except ImportError as e:
    print(f"缺少依赖库: {e}")
    print("请运行: pip install -r requirements.txt")
    exit(1)


class PDFToMarkdownConverter:
    """PDF 转 Markdown 转换器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化转换器"""
        self.config = self._load_config(config_path)
        self.ocr_pipeline = None
        self.llm_client = None
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """默认配置"""
        return {
            'pdf': {
                'dpi': 300,  # PDF 转图像的分辨率
                'page_range': None,  # 页面范围 [start, end] 或 None 表示全部
            },
            'ocr': {
                'use_gpu': True,
            },
            'llm': {
                'enabled': False,
                'provider': 'openai',  # openai, claude, local
                'api_key': '',
                'api_base': '',
                'model': 'gpt-4',
                'chunk_size': 2000,
                'chunk_overlap': 200,
                'max_workers': 5,  # 并行处理的最大线程数
            },
            'output': {
                'save_intermediate': True,
                'output_dir': 'output',
            }
        }
    
    def initialize_ocr(self):
        """初始化 OCR 模型"""
        if self.ocr_pipeline is None:
            print("正在初始化 PaddleOCR-VL 模型...")
            self.ocr_pipeline = PaddleOCRVL()
            print("OCR 模型初始化完成")
    
    def initialize_llm(self):
        """初始化 LLM 客户端"""
        if not self.config['llm']['enabled']:
            return
        
        provider = self.config['llm']['provider']
        api_key = self.config['llm']['api_key']
        
        if provider == 'openai':
            # OpenAI 1.0+ 新版 API
            if self.config['llm'].get('api_base'):
                self.llm_client = OpenAI(
                    api_key=api_key,
                    base_url=self.config['llm']['api_base']
                )
            else:
                self.llm_client = OpenAI(api_key=api_key)
            print(f"已配置 OpenAI API")
            
        elif provider == 'claude':
            self.llm_client = anthropic.Anthropic(api_key=api_key)
            print(f"已配置 Claude API")
            
        elif provider == 'local':
            # 本地 LLM 支持（如 Ollama）
            if self.config['llm'].get('api_base'):
                self.llm_client = OpenAI(
                    api_key="dummy",  # 本地模型不需要真实 API key
                    base_url=self.config['llm']['api_base']
                )
                print(f"已配置本地 LLM: {self.config['llm']['api_base']}")
    
    def pdf_to_images(self, pdf_path: str, page_range: Optional[Tuple[int, int]] = None) -> List[Image.Image]:
        """将 PDF 转换为图像列表"""
        print(f"正在转换 PDF: {pdf_path}")
        
        # 从配置获取 DPI
        dpi = self.config.get('pdf', {}).get('dpi', 300)
        
        # 如果没有传入 page_range，从配置中读取
        if page_range is None:
            config_range = self.config.get('pdf', {}).get('page_range')
            if config_range and len(config_range) == 2:
                page_range = tuple(config_range)
        
        if page_range:
            first_page, last_page = page_range
            print(f"转换页面范围: {first_page} - {last_page}")
            images = convert_from_path(
                pdf_path,
                first_page=first_page,
                last_page=last_page,
                dpi=dpi
            )
        else:
            print("转换所有页面")
            images = convert_from_path(pdf_path, dpi=dpi)
        
        print(f"成功转换 {len(images)} 页")
        return images
    
    def ocr_images(self, images: List[Image.Image], output_dir: str) -> List[str]:
        """对图像列表执行 OCR 识别"""
        self.initialize_ocr()
        
        markdown_pages = []
        total = len(images)
        
        for idx, image in enumerate(images, 1):
            print(f"正在处理第 {idx}/{total} 页...")
            
            # 保存临时图像
            temp_img_path = os.path.join(output_dir, f"temp_page_{idx}.png")
            image.save(temp_img_path)
            
            # OCR 识别
            try:
                output = self.ocr_pipeline.predict(temp_img_path)
                
                # 提取 Markdown 内容
                page_markdown = ""
                for res in output:
                    # 保存为临时 Markdown
                    temp_md_path = os.path.join(output_dir, f"page_{idx}_raw.md")
                    res.save_to_markdown(save_path=output_dir)
                    
                    # 读取生成的 Markdown
                    # PaddleOCR 会生成带时间戳的文件，需要找到对应文件
                    md_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.md')])
                    if md_files:
                        latest_md = os.path.join(output_dir, md_files[-1])
                        with open(latest_md, 'r', encoding='utf-8') as f:
                            page_markdown = f.read()
                        # 重命名为标准名称
                        os.rename(latest_md, temp_md_path)
                
                markdown_pages.append(page_markdown)
                
                # 清理临时图像
                if not self.config['output']['save_intermediate']:
                    os.remove(temp_img_path)
                    
            except Exception as e:
                print(f"处理第 {idx} 页时出错: {e}")
                markdown_pages.append(f"# 页面 {idx} 处理失败\n\n{str(e)}\n\n")
        
        return markdown_pages
    
    def ocr_images_streaming(self, images: List[Image.Image], output_dir: str, page_offset: int = 0):
        """流式 OCR 识别:逐页处理并立即写入临时文件,释放内存"""
        self.initialize_ocr()
        
        total = len(images)
        
        for idx, image in enumerate(images, 1):
            actual_page_num = page_offset + idx
            print(f"正在处理第 {actual_page_num}/{page_offset + total} 页...")
            
            # 保存临时图像
            temp_img_path = os.path.join(output_dir, f"temp_page_{actual_page_num}.png")
            image.save(temp_img_path)
            
            # OCR 识别
            try:
                output = self.ocr_pipeline.predict(temp_img_path)
                
                # 提取 Markdown 内容
                page_markdown = ""
                for res in output:
                    # 保存为临时 Markdown
                    temp_md_path = os.path.join(output_dir, f"page_{actual_page_num}_raw.md")
                    res.save_to_markdown(save_path=output_dir)
                    
                    # 读取生成的 Markdown
                    md_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.md')])
                    if md_files:
                        latest_md = os.path.join(output_dir, md_files[-1])
                        with open(latest_md, 'r', encoding='utf-8') as f:
                            page_markdown = f.read()
                        # 重命名为标准名称
                        os.rename(latest_md, temp_md_path)
                
                # 立即写入临时文件,不保存在内存中
                yield actual_page_num, page_markdown
                
                # 清理临时图像
                if not self.config['output']['save_intermediate']:
                    os.remove(temp_img_path)
                    
            except Exception as e:
                print(f"处理第 {actual_page_num} 页时出错: {e}")
                yield actual_page_num, f"# 页面 {actual_page_num} 处理失败\n\n{str(e)}\n\n"
            
            # 释放图像内存
            del image
    
    def _smart_merge_pages(self, pages: List[str]) -> str:
        """
        智能合并页面，处理跨页文本
        - 如果上一页末尾是不完整的句子（没有结束标点），直接连接下一页
        - 如果是完整的段落，保持段落分隔
        """
        if not pages:
            return ""
        
        if len(pages) == 1:
            return pages[0]
        
        result = []
        
        for i, page in enumerate(pages):
            page = page.strip()
            if not page:
                continue
            
            if i == 0:
                result.append(page)
                continue
            
            # 获取上一页的最后几个字符
            prev_page = result[-1].rstrip() if result else ""
            if not prev_page:
                result.append(page)
                continue
            
            # 检查上一页是否以句子结束符结尾
            sentence_endings = ('。', '！', '？', '.', '!', '?', '\n', ')', '）', '"', '"', '」', '》')
            
            # 如果上一页末尾没有结束标点，说明可能是跨页断句
            if prev_page[-1] not in sentence_endings:
                # 直接连接，不添加分隔符
                result[-1] = prev_page + page
            else:
                # 正常段落分隔
                result.append(page)
        
        return '\n\n'.join(result)
    
    def split_into_chunks(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """将文本分割成带重叠的块"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界分割
            if end < len(text):
                # 查找最近的句子结束符
                for separator in ['\n\n', '。\n', '.\n', '。', '.', '\n']:
                    last_sep = text.rfind(separator, start, end)
                    if last_sep != -1:
                        end = last_sep + len(separator)
                        break
            
            # 防止 end 没有推进导致死循环
            if end <= start:
                end = min(start + chunk_size, len(text))
                if end <= start:
                    break
            
            chunks.append(text[start:end])
            if end < len(text):
                next_start = end - overlap
                # 若重叠导致不前进，禁用本次重叠
                if next_start <= start:
                    next_start = end
                start = max(0, next_start)
            else:
                start = end
        
        return chunks
    
    def process_with_llm(self, text: str) -> str:
        """使用 LLM 处理文本（支持并行）"""
        if not self.config['llm']['enabled'] or not self.llm_client:
            return text
        
        chunk_size = self.config['llm']['chunk_size']
        overlap = self.config['llm']['chunk_overlap']
        chunks = self.split_into_chunks(text, chunk_size, overlap)
        
        total = len(chunks)
        max_workers = self.config['llm'].get('max_workers', 5)
        
        # 如果块数较少或设置为单线程，使用串行处理
        if total <= 1 or max_workers <= 1:
            return self._process_chunks_serial(chunks)
        
        # 使用并行处理
        return self._process_chunks_parallel(chunks, max_workers)
    
    def _process_chunks_serial(self, chunks: List[str]) -> str:
        """串行处理文本块"""
        processed_chunks = []
        total = len(chunks)
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"  使用 LLM 处理文本块 {idx}/{total}...")
            
            try:
                processed = self._call_llm(chunk)
                processed_chunks.append(processed)
            except Exception as e:
                print(f"  LLM 处理失败: {e}")
                processed_chunks.append(chunk)  # 失败时使用原文
        
        return '\n\n'.join(processed_chunks)
    
    def _process_chunks_parallel(self, chunks: List[str], max_workers: int) -> str:
        """并行处理文本块"""
        total = len(chunks)
        print(f"  使用 {max_workers} 个线程并行处理 {total} 个文本块...")
        
        # 存储结果，保持顺序
        results = [None] * total
        
        # 获取请求间隔配置（用于免费API限流）
        request_interval = self.config['llm'].get('request_interval', 0.5)  # 默认每个请求间隔0.5秒
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，添加延迟避免同时请求
            future_to_index = {}
            for idx, chunk in enumerate(chunks):
                if idx > 0 and request_interval > 0:
                    time.sleep(request_interval)  # 提交任务之间的延迟
                future = executor.submit(self._call_llm_safe, chunk, idx)
                future_to_index[future] = idx
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                completed += 1
                print(f"  进度: {completed}/{total}")
                
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  块 {idx+1} 处理失败: {e}")
                    results[idx] = chunks[idx]  # 失败时使用原文
        
        return '\n\n'.join(results)
    
    def _call_llm_safe(self, chunk: str, idx: int) -> str:
        """安全地调用 LLM（带异常处理和重试）"""
        retry_times = self.config['llm'].get('retry_times', 3)
        retry_delay = self.config['llm'].get('retry_delay', 2)
        
        for attempt in range(retry_times):
            try:
                return self._call_llm(chunk)
            except Exception as e:
                error_msg = str(e)
                if attempt < retry_times - 1:
                    # 检查是否是速率限制错误
                    if '429' in error_msg or 'rate' in error_msg.lower():
                        wait_time = retry_delay * (attempt + 1)  # 指数退避
                        print(f"  块 {idx+1} 遇到速率限制，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"  块 {idx+1} LLM 调用失败（尝试 {attempt+1}/{retry_times}），重试中...")
                        time.sleep(retry_delay)
                else:
                    print(f"  块 {idx+1} LLM 调用异常（已重试{retry_times}次）: {e}")
                    return chunk  # 返回原文
    
    def _call_llm(self, text: str) -> str:
        """调用 LLM API"""
        prompt = f"""你是一个专业的文档编辑助手。请优化以下 OCR 识别的文本。

要求：
1. 修正 OCR 识别错误（如拼写错误、连字符断词等）
2. 改进 Markdown 格式（标题层级、列表格式、强调标记）
3. 保持原文意思和段落结构，不要添加或删除实质内容
4. 【重要】直接输出优化后的文本内容，不要输出任何解释、说明或前缀文字
5. 【重要】不要输出类似"好的"、"优化后的文本"、"以下是"、"修正了"等说明性文字
6. 【重要】直接从文章正文开始输出

原始文本：
{text}

请直接输出优化后的文本："""

        try:
            if isinstance(self.llm_client, OpenAI):
                # OpenAI 1.0+ 新版 API
                response = self.llm_client.chat.completions.create(
                    model=self.config['llm']['model'],
                    messages=[
                        {"role": "system", "content": "你是一个专业的文档编辑助手。你只输出优化后的文本内容，不输出任何解释、说明或额外文字。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )
                result = response.choices[0].message.content.strip()
                # 清理可能的前缀文字
                return self._clean_llm_output(result)
            
            elif isinstance(self.llm_client, anthropic.Anthropic):
                response = self.llm_client.messages.create(
                    model=self.config['llm']['model'],
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )
                return response.content[0].text.strip()
        
        except Exception as e:
            print(f"LLM API 调用失败: {e}")
            raise
    
    def _clean_llm_output(self, text: str) -> str:
        """清理 LLM 输出中的多余说明文字（只移除明确的 AI 生成说明）"""
        original_text = text
        
        # 1. 移除开头的明确说明性前缀（只匹配完整短语）
        prefixes_to_remove = [
            "好的，这是根据您的要求优化后的文本。\n\n",
            "好的，这是根据您的要求优化后的文本。\n",
            "好的，以下是优化后的文本：\n\n",
            "好的，以下是优化后的文本：\n",
            "优化后的文本：\n\n",
            "优化后的文本：\n",
            "优化后的内容：\n\n",
            "优化后的内容：\n",
            "以下是优化后的文本：\n\n",
            "以下是优化后的文本：\n",
            "根据要求，优化后的文本如下：\n\n",
            "根据要求，优化后的文本如下：\n",
        ]
        
        # 只在文本开头匹配，且必须包含换行符，避免误删正文
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break  # 只移除一次
        
        # 2. 移除结尾的说明段落（必须是独立段落）
        # 这些标记通常出现在文档末尾，前面有两个换行符
        explanation_patterns = [
            "\n\n优化内容包括：",
            "\n\n优化说明：", 
            "\n\n修改说明：",
            "\n\n改进点：",
            "\n\n主要改进：",
            "\n\n* **修正OCR错误",  # 匹配项目符号列表形式的说明
            "\n\n**修正OCR错误",
        ]
        
        for pattern in explanation_patterns:
            if pattern in text:
                # 只截取第一次出现之前的内容
                parts = text.split(pattern, 1)
                if len(parts) > 1:
                    # 检查截取后的内容是否合理（不能太短）
                    if len(parts[0].strip()) > 100:  # 至少保留100字符
                        text = parts[0].strip()
                        break
        
        # 3. 移除开头的单行说明（更保守的检查）
        lines = text.split('\n')
        if len(lines) > 2:  # 至少有3行才处理
            first_line = lines[0].strip()
            # 只移除明确的说明性单行，且下一行不是空行
            if (len(first_line) < 50 and  # 说明文字通常较短
                not first_line.startswith('#') and  # 不是标题
                any(word in first_line for word in ['优化后', '以下是', '根据要求']) and
                lines[1].strip() != ''):  # 下一行不是空行
                text = '\n'.join(lines[1:]).strip()
        
        # 4. 安全检查：如果清理后文本变化太大（减少超过20%），保留原文
        if len(text) < len(original_text) * 0.8:
            print("  警告: LLM 输出清理可能过度，保留原始输出")
            return original_text
        
        return text
    
    def process_pages_with_llm_streaming(self, work_dir: str, total_pages: int, output_file: str):
        """流式处理:分批读取页面,LLM 处理后立即写入输出文件"""
        if not self.config['llm']['enabled']:
            # 如果不使用 LLM,直接合并所有页面
            self._merge_pages_to_file(work_dir, total_pages, output_file)
            return
        
        print("正在使用 LLM 流式优化文本...")
        self.initialize_llm()
        
        if not self.llm_client:
            print("警告: LLM 客户端初始化失败,将直接合并页面")
            self._merge_pages_to_file(work_dir, total_pages, output_file)
            return
        
        batch_size = self.config['llm'].get('batch_size', 10)  # 每批处理的页数
        
        with open(output_file, 'a', encoding='utf-8') as out_f:
            for batch_start in range(1, total_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, total_pages)
                print(f"处理页面 {batch_start}-{batch_end}/{total_pages}...")
                
                # 读取一批页面
                batch_pages = []
                for page_num in range(batch_start, batch_end + 1):
                    page_file = os.path.join(work_dir, f"page_{page_num}_raw.md")
                    if os.path.exists(page_file):
                        with open(page_file, 'r', encoding='utf-8') as f:
                            batch_pages.append(f.read())
                    else:
                        batch_pages.append(f"# 页面 {page_num} 缺失\n\n")
                
                # 合并这批页面
                batch_text = self._smart_merge_pages(batch_pages)
                
                # LLM 处理
                processed_text = self.process_with_llm(batch_text)
                
                # 立即写入输出文件
                out_f.write(processed_text)
                out_f.write('\n\n')  # 批次之间的分隔
                out_f.flush()  # 强制写入磁盘
                
                # 释放内存
                del batch_pages
                del batch_text
                del processed_text
    
    def _merge_pages_to_file(self, work_dir: str, total_pages: int, output_file: str):
        """直接合并所有页面到输出文件(不使用 LLM)"""
        print("正在合并所有页面...")
        
        with open(output_file, 'a', encoding='utf-8') as out_f:
            prev_page = ""
            for page_num in range(1, total_pages + 1):
                page_file = os.path.join(work_dir, f"page_{page_num}_raw.md")
                if os.path.exists(page_file):
                    with open(page_file, 'r', encoding='utf-8') as f:
                        current_page = f.read().strip()
                    
                    if not current_page:
                        continue
                    
                    # 智能连接页面
                    if prev_page:
                        sentence_endings = ('。', '!', '?', '.', '!', '?', '\n', ')', ')', '"', '"', '」', '》')
                        if prev_page[-1] not in sentence_endings:
                            out_f.write(current_page)
                        else:
                            out_f.write('\n\n' + current_page)
                    else:
                        out_f.write(current_page)
                    
                    prev_page = current_page
                    out_f.flush()
    
    def convert(self, pdf_path: str, output_path: Optional[str] = None, 
                page_range: Optional[Tuple[int, int]] = None) -> str:
        """
        转换 PDF 为 Markdown
        
        Args:
            pdf_path: PDF 文件路径
            output_path: 输出 Markdown 文件路径
            page_range: 页面范围 (起始页, 结束页)，从 1 开始
        
        Returns:
            输出文件路径
        """
        # 创建输出目录
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建临时工作目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(output_dir, f"temp_{timestamp}")
        os.makedirs(work_dir, exist_ok=True)
        
        # 确定输出文件名
        if output_path is None:
            pdf_name = Path(pdf_path).stem
            output_path = os.path.join(output_dir, f"{pdf_name}.md")
        
        try:
            # 步骤 1: 确定总页数
            print(f"正在分析 PDF: {pdf_path}")
            dpi = self.config.get('pdf', {}).get('dpi', 300)
            config_range = self.config.get('pdf', {}).get('page_range')
            if page_range is None and config_range and len(config_range) == 2:
                page_range = tuple(config_range)
            
            # 先获取总页数(不加载图像内容)
            from pdf2image import pdfinfo_from_path
            pdf_info = pdfinfo_from_path(pdf_path)
            total_pages = pdf_info.get('Pages', 0)
            
            if page_range:
                first_page, last_page = page_range
                total_pages = last_page - first_page + 1
                page_offset = first_page - 1
            else:
                first_page = 1
                last_page = total_pages
                page_offset = 0
            
            print(f"总页数: {total_pages},处理范围: {first_page}-{last_page}")
            
            # 写入文件头部
            with open(output_path, 'w', encoding='utf-8') as f:
                header = f"""# {Path(pdf_path).stem}

> 转换时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 原文件: {pdf_path}
> 页数: {total_pages}

---

"""
                f.write(header)
            
            # 步骤 2: 流式处理 - 分批转换和 OCR
            # 每次只处理少量页面,避免内存占用过大
            batch_size = self.config.get('pdf', {}).get('batch_size', 20)  # 每批处理的页数
            
            for batch_start in range(first_page, last_page + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, last_page)
                print(f"\n=== 处理批次: 页面 {batch_start}-{batch_end} ===")
                
                # 只转换当前批次的页面
                print(f"正在转换 PDF 页面 {batch_start}-{batch_end}...")
                batch_images = convert_from_path(
                    pdf_path,
                    first_page=batch_start,
                    last_page=batch_end,
                    dpi=dpi
                )
                
                # OCR 识别(流式,逐页处理)
                for page_num, page_markdown in self.ocr_images_streaming(batch_images, work_dir, batch_start - 1):
                    # 写入临时文件
                    temp_md_path = os.path.join(work_dir, f"page_{page_num}_raw.md")
                    with open(temp_md_path, 'w', encoding='utf-8') as f:
                        f.write(page_markdown)
                
                # 释放图像内存
                del batch_images
                print(f"批次 {batch_start}-{batch_end} 处理完成,内存已释放")
            
            # 步骤 3: 流式 LLM 处理和合并
            # 分批读取页面,处理后立即写入输出文件
            print("\n=== 开始文本优化和合并 ===")
            self.process_pages_with_llm_streaming(work_dir, last_page, output_path)
            
            print(f"\n✓ 转换完成! 输出文件: {output_path}")
            
            # 清理临时文件
            if not self.config['output']['save_intermediate']:
                import shutil
                shutil.rmtree(work_dir)
                print("临时文件已清理")
            else:
                print(f"中间文件保存在: {work_dir}")
            
            return output_path
            
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PDF 图书转 Markdown 工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（所有配置在 config.yaml 中设置）
  python pdf_to_markdown.py book.pdf
  
  # 指定输出文件
  python pdf_to_markdown.py book.pdf -o my_book.md
  
  # 使用自定义配置文件
  python pdf_to_markdown.py book.pdf -c my_config.yaml
  
  # 命令行覆盖页面范围（优先级高于配置文件）
  python pdf_to_markdown.py book.pdf --pages 1-10

配置说明:
  推荐在 config.yaml 中设置所有选项，包括:
  - PDF 处理参数（DPI、页面范围）
  - LLM 配置（API Key、模型、并行数）
  - 输出选项
  
  命令行参数仅用于临时覆盖配置文件的设置。
        """
    )
    
    parser.add_argument('pdf_path', help='PDF 文件路径')
    parser.add_argument('-o', '--output', help='输出 Markdown 文件路径（覆盖配置文件）')
    parser.add_argument('-p', '--pages', help='页面范围，格式: 起始-结束 (如: 1-10，覆盖配置文件)', default=None)
    parser.add_argument('-c', '--config', help='配置文件路径（默认: config.yaml）', default='config.yaml')
    
    args = parser.parse_args()
    
    # 解析页面范围（命令行参数优先于配置文件）
    page_range = None
    if args.pages:
        try:
            start, end = map(int, args.pages.split('-'))
            page_range = (start, end)
        except ValueError:
            print(f"错误: 页面范围格式不正确，应为 '起始-结束'，如 '1-10'")
            return 1
    
    # 创建转换器
    converter = PDFToMarkdownConverter(args.config)
    
    # 执行转换（命令行参数 page_range 会覆盖配置文件）
    try:
        converter.convert(args.pdf_path, args.output, page_range)
        return 0
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
