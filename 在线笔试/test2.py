import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import json
import os
import cv2
import numpy as np
from collections import defaultdict


class OptimizedComponentExtractor:
    def __init__(self, tesseract_path=None):
        """
        优化版元器件提取器
        """
        # 设置Tesseract路径
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✓ Tesseract路径: {tesseract_path}")
        else:
            # 自动查找
            auto_path = self._find_tesseract()
            if auto_path:
                pytesseract.pytesseract.tesseract_cmd = auto_path
                print(f"✓ 自动找到Tesseract: {auto_path}")
            else:
                print("❌ 未找到Tesseract")
                return

        # 验证Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract版本: {version}")
        except:
            print("❌ Tesseract不可用")
            return

        # 元器件知识库（扩展版）
        self.component_knowledge = {
            # 显示设备
            'display': {
                'keywords': ['液晶屏', '显示屏', '显示器', '屏幕', 'LCD', 'LED屏', '触摸屏', '显示面板'],
                'patterns': [r'液晶屏', r'显示屏', r'显示.*屏'],
                'description': '显示设备'
            },
            # 传感器
            'sensor': {
                'keywords': ['传感器', '探头', '感应器', '检测器', '测温', '测压', '测位'],
                'patterns': [r'传感器', r'[温压位速转]传感器', r'.*探头'],
                'description': '传感器类'
            },
            # 开关
            'switch': {
                'keywords': ['开关', '按钮', '按键', '旋钮', '拨杆', '翘板'],
                'patterns': [r'开关', r'按钮', r'按键', r'旋钮'],
                'description': '开关类'
            },
            # 连接器
            'connector': {
                'keywords': ['连接器', '插头', '插座', '端子', '接插件', '接头'],
                'patterns': [r'连接器', r'[插接]头', r'[插接]座', r'端子'],
                'description': '连接器'
            },
            # 控制模块
            'controller': {
                'keywords': ['模块', 'ECU', '电脑', '控制器', '控制单元', '控制模块'],
                'patterns': [r'[模电]控', r'ECU', r'控制.*模块', r'电脑板'],
                'description': '控制模块'
            },
            # 线束
            'harness': {
                'keywords': ['线束', '电缆', '电线', '导线', '线缆', '线束总成'],
                'patterns': [r'线束', r'[电线]缆', r'导线'],
                'description': '线束类'
            },
            # 特殊系统
            'adblue': {
                'keywords': ['AdBlue', '尿素', 'SCR', '排放', '尾气'],
                'patterns': [r'AdBlue', r'尿素', r'SCR'],
                'description': '尿素系统'
            },
            # FA10发动机
            'fa10': {
                'keywords': ['FA10', '锡柴', '发动机', '引擎'],
                'patterns': [r'FA10', r'锡柴', r'发动机'],
                'description': 'FA10发动机'
            }
        }

        # 尺寸/规格模式
        self.spec_patterns = [
            r'(\d+\.?\d*)\s*[×xX*]\s*(\d+\.?\d*)',  # 100×50
            r'(\d+\.?\d*)\s*[±±]\s*(\d+\.?\d*)',  # 330.8±0.5
            r'[ΦϕØ]\s*(\d+\.?\d*)',  # Φ10
            r'(\d+\.?\d*)\s*[Mm][Mm]',  # 100mm
            r'(\d+\.?\d*)\s*[Cc][Mm]',  # 10cm
            r'(\d+\.?\d*)\s*°[Cc]',  # 25°C
        ]

        # 连接器编码模式
        self.connector_patterns = [
            r'([A-Z][0-9]+)P([0-9]+)',  # C2P1
            r'([A-Z][0-9]+)[-_\.]([0-9]+)',  # C2-1, C2_1, C2.1
            r'(J[0-9]+)',  # J100
            r'(X[0-9]+)',  # X1
        ]

        # 存储结果
        self.components = []
        self.stats = defaultdict(int)

    def _find_tesseract(self):
        """查找Tesseract"""
        paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'D:\Tesseract-OCR\tesseract.exe',
            r'E:\Tesseract-OCR\tesseract.exe',
            r'R:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def _preprocess_image(self, image):
        """
        图像预处理 - 提高OCR准确率
        """
        # 转换为灰度
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        # 转换为numpy数组
        img_array = np.array(gray)

        # 1. 自适应阈值（提高对比度）
        binary = cv2.adaptiveThreshold(img_array, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 2. 降噪
        denoised = cv2.medianBlur(binary, 3)

        # 3. 锐化（增强边缘）
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 4. 形态学操作（连接断开的笔画）
        kernel2 = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel2)

        # 转回PIL图像
        processed = Image.fromarray(morphed)

        return processed

    def _enhance_ocr_accuracy(self, image):
        """
        额外的图像增强
        """
        # 调整对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # 调整亮度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)

        # 锐化
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)

        return image

    def _ocr_with_retry(self, image):
        """
        带重试机制的OCR
        """
        results = []

        # 尝试不同的OCR配置
        configs = [
            {'lang': 'chi_sim', 'config': '--oem 3 --psm 6'},
            {'lang': 'chi_sim+eng', 'config': '--oem 3 --psm 6'},
            {'lang': 'eng', 'config': '--oem 3 --psm 6'},
            {'lang': 'chi_sim', 'config': '--oem 3 --psm 3'},
            {'lang': 'chi_sim', 'config': '--oem 1 --psm 6'},
        ]

        for config in configs:
            try:
                text = pytesseract.image_to_string(
                    image,
                    lang=config['lang'],
                    config=config['config']
                )

                if text and text.strip():
                    results.append({
                        'text': text,
                        'config': config,
                        'length': len(text.strip())
                    })

                    # 如果识别结果足够长，提前返回
                    if len(text.strip()) > 20:
                        break

            except Exception as e:
                continue

        # 选择最好的结果
        if results:
            # 按文本长度排序
            results.sort(key=lambda x: x['length'], reverse=True)
            return results[0]['text']

        return ""

    def extract_from_pdf(self, pdf_path, max_pages=None, save_images=False):
        """
        从PDF提取元器件
        """
        print(f"🔍 开始分析: {os.path.basename(pdf_path)}")

        if not os.path.exists(pdf_path):
            print(f"❌ 文件不存在: {pdf_path}")
            return []

        # 创建输出目录
        if save_images:
            output_dir = "processed_images"
            os.makedirs(output_dir, exist_ok=True)

        try:
            # 打开PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            if max_pages:
                total_pages = min(total_pages, max_pages)

            print(f"📊 PDF总页数: {len(doc)} (处理前 {total_pages} 页)")

            for page_num in range(total_pages):
                print(f"\n📖 第 {page_num + 1}/{total_pages} 页")
                page = doc[page_num]

                # 获取页面图像
                image_list = page.get_images()

                if image_list:
                    print(f"  发现 {len(image_list)} 个图像")
                    self.stats['total_images'] += len(image_list)

                    for img_idx, img_info in enumerate(image_list):
                        self.stats['processed_images'] += 1

                        try:
                            # 提取图像
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]

                            # 转换为PIL图像
                            original = Image.open(io.BytesIO(image_bytes))

                            # 预处理图像
                            processed = self._preprocess_image(original)
                            enhanced = self._enhance_ocr_accuracy(processed)

                            # 保存处理后的图像（调试用）
                            if save_images:
                                img_name = f"page_{page_num + 1}_img_{img_idx + 1}.png"
                                enhanced.save(os.path.join(output_dir, img_name))

                            # OCR识别
                            text = self._ocr_with_retry(enhanced)

                            if text and text.strip():
                                # 清理文本
                                cleaned = self._clean_ocr_text(text)

                                print(f"    图像 {img_idx + 1}: ✓ 识别成功 ({len(cleaned)} 字符)")

                                # 提取元器件
                                found = self._analyze_text(cleaned, page_num + 1, img_idx + 1)

                                if found:
                                    self.components.extend(found)
                                    print(f"      ✅ 找到 {len(found)} 个元器件")

                                    # 显示前几个
                                    for comp in found[:3]:
                                        print(f"        • {comp['name']}")

                                # 保存识别的文本
                                if save_images and cleaned:
                                    txt_name = f"page_{page_num + 1}_img_{img_idx + 1}.txt"
                                    with open(os.path.join(output_dir, txt_name),
                                              'w', encoding='utf-8') as f:
                                        f.write(cleaned)

                            else:
                                print(f"    图像 {img_idx + 1}: ⚠️ 未识别到文字")

                        except Exception as e:
                            print(f"    图像 {img_idx + 1}: ❌ 处理失败 - {str(e)[:50]}")

                else:
                    print(f"  ⚠️ 本页无图像")

            doc.close()

        except Exception as e:
            print(f"❌ 处理PDF失败: {e}")

        return self.components

    def _clean_ocr_text(self, text):
        """
        清理OCR识别的文本
        """
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)

        # 修复常见OCR错误
        corrections = {
            '土': '±',
            '一': '-',
            '三': '=',
            '巳': '已',
            '曰': '日',
            '冫': '冰',
            '十': '+',
            '口': '口',
        }

        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)

        # 修复尺寸符号
        text = re.sub(r'(\d+\.?\d*)\s*[十+]', r'\1±', text)

        return text.strip()

    def _analyze_text(self, text, page_num, img_num):
        """
        分析文本，提取元器件信息
        """
        components = []

        # 按行分割
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or len(line) < 2:
                continue

            # 1. 查找元器件关键词
            for category, info in self.component_knowledge.items():
                for keyword in info['keywords']:
                    if keyword in line:
                        component = self._create_component(
                            line, keyword, info['description'],
                            page_num, img_num, line_num
                        )
                        if component:
                            components.append(component)
                        break  # 每个关键词只匹配一次

            # 2. 查找连接器编码
            for pattern in self.connector_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    if len(match.groups()) >= 1:
                        component = self._create_connector_component(
                            match, line, page_num, img_num, line_num
                        )
                        if component:
                            components.append(component)

            # 3. 查找尺寸规格
            for pattern in self.spec_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    component = self._create_spec_component(
                        match, line, page_num, img_num, line_num
                    )
                    if component:
                        components.append(component)

        return components

    def _create_component(self, line, keyword, category, page_num, img_num, line_num):
        """
        创建元器件信息
        """
        # 提取包含关键词的上下文
        name = self._extract_context(line, keyword)

        # 提取规格
        specs = self._extract_specifications(line)

        component = {
            'name': name,
            'category': category,
            'keyword': keyword,
            'page': page_num,
            'image': img_num,
            'line': line_num,
            'text': line[:100],
            'specifications': specs,
            'confidence': self._estimate_confidence(line, keyword),
            'type': 'keyword'
        }

        return component

    def _create_connector_component(self, match, line, page_num, img_num, line_num):
        """
        创建连接器组件
        """
        groups = match.groups()

        if len(groups) >= 2:
            # 格式如 C2P1
            connector = groups[0]
            pin = groups[1]
            code = f"{connector}P{pin}"
            name = f"{connector}连接器 (针脚{pin})"
        else:
            # 格式如 J100
            connector = groups[0]
            pin = None
            code = connector
            name = f"{connector}连接器"

        component = {
            'name': name,
            'category': '连接器',
            'connector': connector,
            'pin': pin,
            'code': code,
            'page': page_num,
            'image': img_num,
            'line': line_num,
            'text': line[:100],
            'confidence': '高',
            'type': 'connector'
        }

        return component

    def _create_spec_component(self, match, line, page_num, img_num, line_num):
        """
        创建规格组件
        """
        spec_text = match.group()

        component = {
            'name': f"规格: {spec_text}",
            'category': '规格参数',
            'specification': spec_text,
            'page': page_num,
            'image': img_num,
            'line': line_num,
            'text': line[:100],
            'confidence': '高',
            'type': 'specification'
        }

        return component

    def _extract_context(self, line, keyword):
        """
        提取包含关键词的上下文
        """
        words = line.split()
        for i, word in enumerate(words):
            if keyword in word:
                # 获取前后词汇
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                return ' '.join(words[start:end])

        return line[:50]

    def _extract_specifications(self, line):
        """
        提取规格参数
        """
        specs = {}

        for pattern in self.spec_patterns:
            matches = re.findall(pattern, line)
            if matches:
                if '×' in pattern or 'x' in pattern or 'X' in pattern:
                    specs['dimensions'] = matches[0]
                elif '±' in pattern:
                    specs['tolerance'] = matches[0]
                elif 'Φ' in pattern or 'ϕ' in pattern or 'Ø' in pattern:
                    specs['diameter'] = matches[0]
                elif 'mm' in pattern.lower():
                    specs['length_mm'] = matches
                elif 'cm' in pattern.lower():
                    specs['length_cm'] = matches
                elif '°C' in pattern:
                    specs['temperature'] = matches

                # 找到第一个就停止
                break

        return specs

    def _estimate_confidence(self, line, keyword):
        """
        估计识别置信度
        """
        if len(line) < 30 and keyword in line:
            return '高'
        elif len(line) < 100 and keyword in line:
            return '中'
        else:
            return '低'

    def generate_report(self, output_file="ocr_components_report.txt"):
        """
        生成详细报告
        """
        print(f"\n📊 生成分析报告...")

        report = []
        report.append("=" * 80)
        report.append("📄 PDF图像元器件识别报告 (使用OCR)")
        report.append("=" * 80)

        # 统计信息
        report.append(f"\n📈 处理统计:")
        report.append(f"  处理的图像总数: {self.stats.get('processed_images', 0)}")
        report.append(f"  找到的元器件总数: {len(self.components)}")

        if self.components:
            # 按类别统计
            category_stats = defaultdict(int)
            type_stats = defaultdict(int)
            confidence_stats = defaultdict(int)

            for comp in self.components:
                category_stats[comp['category']] += 1
                type_stats[comp['type']] += 1
                confidence_stats[comp['confidence']] += 1

            report.append(f"\n🔧 元器件分类:")
            for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {category}: {count}个")

            report.append(f"\n📋 识别类型:")
            for type_name, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {type_name}: {count}个")

            report.append(f"\n🎯 置信度分布:")
            for conf, count in sorted(confidence_stats.items()):
                icon = '✅' if conf == '高' else '⚠️' if conf == '中' else '❓'
                report.append(f"  {icon} {conf}: {count}个")

            # 详细列表（按页码）
            report.append(f"\n📖 详细元器件列表:")

            by_page = defaultdict(list)
            for comp in self.components:
                by_page[comp['page']].append(comp)

            for page in sorted(by_page.keys()):
                report.append(f"\n  第 {page} 页 ({len(by_page[page])}个):")

                for i, comp in enumerate(by_page[page][:15], 1):
                    conf_icon = '✅' if comp['confidence'] == '高' else '⚠️' if comp['confidence'] == '中' else '❓'

                    line = f"    {i:2d}. {conf_icon} {comp['name']}"

                    if comp.get('code'):
                        line += f" [编码: {comp['code']}]"

                    if comp.get('specifications'):
                        specs = comp['specifications']
                        if 'dimensions' in specs:
                            dims = specs['dimensions']
                            if isinstance(dims, tuple):
                                line += f" (尺寸: {'×'.join(map(str, dims))})"
                            else:
                                line += f" (尺寸: {dims})"

                    report.append(line)

                if len(by_page[page]) > 15:
                    report.append(f"    ... 还有 {len(by_page[page]) - 15} 个")

        else:
            report.append(f"\n⚠️ 未找到任何元器件")

        report.append(f"\n{'=' * 80}")
        report.append("报告生成完成")
        report.append("=" * 80)

        report_text = "\n".join(report)

        # 保存报告
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ 报告已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")

        # 打印到控制台
        print(report_text)

        return report_text

    def export_data(self, output_file="ocr_components_data.json"):
        """
        导出数据
        """
        if not self.components:
            print("没有数据可导出")
            return

        data = {
            'metadata': {
                'total_components': len(self.components),
                'processed_images': self.stats.get('processed_images', 0),
                'tesseract_version': '5.3.1.20230401',
                'analysis_method': 'ocr_based'
            },
            'components': self.components,
            'statistics': dict(self.stats)
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 数据已导出到: {output_file}")
        except Exception as e:
            print(f"❌ 导出数据失败: {e}")


def main():
    """
    主程序
    """
    print("=" * 80)
    print("🚗 汽车线束图元器件OCR识别系统")
    print("=" * 80)
    print("说明: 使用新版Tesseract 5.x进行OCR识别")
    print("=" * 80)

    # 检查依赖
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import cv2
        import numpy as np
        print("✓ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("\n请安装:")
        print("pip install PyMuPDF pytesseract pillow opencv-python numpy")
        return

    # 查找PDF文件
    pdf_files = [
        "test.pdf",
        "一汽解放_新款J6L_整车线束图【含各类选装配置】【锡柴自主FA10-气驱罐】【国六】.pdf",
        "线束图.pdf",
        "harness.pdf",
    ]

    pdf_path = None
    for file in pdf_files:
        if os.path.exists(file):
            pdf_path = file
            print(f"✓ 找到PDF文件: {file}")
            break

    if not pdf_path:
        # 查找第一个PDF
        for file in os.listdir('.'):
            if file.lower().endswith('.pdf'):
                pdf_path = file
                print(f"✓ 找到PDF文件: {file}")
                break

    if not pdf_path:
        print("❌ 未找到PDF文件")
        print("当前目录:", os.listdir('.'))
        return

    # 创建提取器
    extractor = OptimizedComponentExtractor(
        tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )

    # 提取元器件
    print("\n" + "=" * 80)
    print("开始OCR识别元器件...")
    print("=" * 80)
    print("注意: 首次运行可能较慢，正在处理图像...")

    # 处理PDF（可以限制页数进行测试）
    components = extractor.extract_from_pdf(
        pdf_path=pdf_path,
        max_pages=None,  # None表示处理所有页，可以设为5进行测试
        save_images=True  # 保存处理后的图像用于调试
    )

    # 生成报告
    print("\n" + "=" * 80)
    print("生成详细报告...")
    print("=" * 80)

    extractor.generate_report("元器件OCR识别报告.txt")

    # 导出数据
    extractor.export_data("元器件OCR数据.json")

    # 显示关键结果
    if components:
        print("\n🎯 关键识别结果:")
        print("-" * 40)

        # 高置信度结果
        high_conf = [c for c in components if c['confidence'] == '高']
        if high_conf:
            print(f"\n✅ 高置信度元器件 ({len(high_conf)}个):")
            for comp in high_conf[:10]:
                print(f"  • {comp['name']}")
                if comp.get('page'):
                    print(f"    所在页: 第{comp['page']}页")
                if comp.get('specifications'):
                    specs = comp['specifications']
                    if 'dimensions' in specs:
                        print(f"    规格: {specs['dimensions']}")

        # 连接器信息
        connectors = [c for c in components if c['type'] == 'connector']
        if connectors:
            print(f"\n🔌 连接器识别 ({len(connectors)}个):")
            conn_stats = {}
            for conn in connectors:
                code = conn.get('connector')
                if code:
                    conn_stats[code] = conn_stats.get(code, 0) + 1

            for code, count in sorted(conn_stats.items()):
                print(f"  {code}系列: {count}个针脚")

        # 规格信息
        specs = [c for c in components if c['type'] == 'specification']
        if specs:
            print(f"\n📏 尺寸规格 ({len(specs)}个):")
            for spec in specs[:5]:
                print(f"  {spec['specification']}")

    print(f"\n✨ OCR识别完成！")


if __name__ == "__main__":
    main()