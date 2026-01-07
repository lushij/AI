import pdfplumber
import re
import json
from collections import defaultdict
from datetime import datetime
import os
import pandas as pd


class AutomotiveHarnessParser:
    def __init__(self):
        # 汽车线束专用术语词典
        self.component_dictionary = {
            # 连接器类型
            '连接器': ['连接器', '接插件', '插接件', '插座', '插头', '端子', 'PIN'],
            '线束': ['线束', '线缆', '电线', '导线', '电缆'],
            '传感器': ['传感器', '探头', '感应器', '探测器'],
            '开关': ['开关', '按钮', '旋钮', '按键', '拨杆'],
            '继电器': ['继电器', '电磁继电器', '固态继电器'],
            '保险': ['保险丝', '熔断器', '保险', '断路器'],
            '模块': ['模块', '控制模块', 'ECU', '电脑板', '控制器'],
            '电机': ['电机', '马达', '电动机', '电动马达'],
            '泵': ['泵', '油泵', '水泵', '燃油泵', '尿素泵'],
            '阀': ['阀', '电磁阀', '阀门', '气动阀', '液压阀'],
            '灯': ['灯', '灯泡', 'LED灯', '指示灯', '照明灯'],
            '仪表': ['仪表', '仪表盘', '显示屏', '显示器', '仪表板'],
            '传感器': ['传感器', '温度传感器', '压力传感器', '位置传感器', '速度传感器'],

            # 汽车系统
            '发动机系统': ['发动机', '引擎', '柴油机', '汽油机', 'FA10', '锡柴'],
            '排放系统': ['排放', '尾气', '废气', '催化器', 'DPF', 'SCR', '尿素', 'AdBlue'],
            '电气系统': ['电气', '电路', '电源', '蓄电池', '发电机', '起动机'],
            '制动系统': ['制动', '刹车', 'ABS', 'EBD', 'ESP'],
            '转向系统': ['转向', '方向盘', '转向机', '助力转向'],
            '空调系统': ['空调', '暖风', '制冷', '压缩机', '冷凝器'],
            '安全系统': ['安全', '气囊', '安全带', '防盗', '报警'],
        }

        # 连接器编号模式
        self.connector_patterns = [
            r'([A-Z][0-9]+)P([0-9]+)',  # C2P1
            r'([A-Z][0-9]+)-([0-9]+)',  # C2-1
            r'(J[0-9]+)',  # J100
            r'(X[0-9]+)',  # X1
            r'(S[0-9]+)',  # S1
        ]

        # 零件号模式
        self.part_number_patterns = [
            r'(CA\d+[A-Z0-9_\-]+)',  # CA1251P62K1L7T3E5_S100001_07
            r'([0-9]{8,}[A-Z]*)',  # 长数字编码
            r'([A-Z]{2,}\d+[A-Z0-9]+)',  # 字母数字混合编码
        ]

        # 日期模式
        self.date_patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # 2021-11-12
            r'(\d{8})',  # 20211112
            r'(\d{4}年\d{1,2}月\d{1,2}日)',  # 2021年11月12日
        ]

        # 已知的关键部件映射
        self.known_components = {
            'AdBlue': {
                'name': 'AdBlue尿素喷射系统',
                'type': '排放系统',
                'subcomponents': ['尿素泵', '尿素喷嘴', '尿素罐', '浓度传感器', '温度传感器', '控制单元']
            },
            'C2': {
                'name': '发动机ECU主连接器',
                'type': '连接器',
                'pin_count': 100,
                'function': '发动机控制单元连接'
            },
            'CA1251P62K1L7T3E5_S100001_07': {
                'name': '新款J6L整车主线束',
                'type': '线束总成',
                'description': '适用于各种选装配置'
            },
            'FA10': {
                'name': '锡柴自主FA10发动机',
                'type': '发动机系统',
                'spec': '国六排放标准'
            }
        }

    def extract_all_content(self, pdf_path):
        """从PDF中提取所有内容"""
        print(f"正在解析PDF文件: {os.path.basename(pdf_path)}")

        all_content = {
            'text': '',
            'tables': [],
            'pages': [],
            'metadata': {}
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_content['metadata']['total_pages'] = len(pdf.pages)
                all_content['metadata']['file_name'] = os.path.basename(pdf_path)

                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"  处理第 {page_num}/{len(pdf.pages)} 页...")

                    # 提取文本
                    page_text = page.extract_text()

                    # 提取表格
                    tables = page.extract_tables()

                    # 提取字符级别信息（用于精确位置）
                    chars = page.chars

                    page_data = {
                        'page_number': page_num,
                        'text': page_text,
                        'tables': tables,
                        'char_count': len(chars),
                        'bbox': page.bbox
                    }

                    all_content['pages'].append(page_data)
                    all_content['text'] += f"\n=== Page {page_num} ===\n{page_text}"

                    # 处理表格数据
                    for table_num, table in enumerate(tables, 1):
                        table_data = {
                            'page': page_num,
                            'table_number': table_num,
                            'rows': len(table),
                            'columns': len(table[0]) if table else 0,
                            'data': table
                        }
                        all_content['tables'].append(table_data)

                        # 将表格数据也添加到文本中
                        table_text = self._table_to_text(table)
                        all_content['text'] += f"\n[Table {page_num}-{table_num}]\n{table_text}"

        except Exception as e:
            print(f"解析PDF时出错: {e}")
            return None

        print(f"提取完成: {len(all_content['pages'])}页, {len(all_content['tables'])}个表格")
        return all_content

    def _table_to_text(self, table):
        """将表格转换为文本"""
        if not table:
            return ""

        lines = []
        for row in table:
            # 过滤空单元格并连接
            row_text = " | ".join([str(cell).strip() for cell in row if cell])
            if row_text:
                lines.append(row_text)

        return "\n".join(lines)

    def find_all_components(self, content):
        """查找所有元器件"""
        print("\n开始识别元器件...")

        components = {
            'connectors': [],  # 连接器
            'harnesses': [],  # 线束
            'sensors': [],  # 传感器
            'switches': [],  # 开关
            'relays': [],  # 继电器
            'fuses': [],  # 保险丝
            'modules': [],  # 模块
            'motors': [],  # 电机
            'pumps': [],  # 泵
            'valves': [],  # 阀
            'lights': [],  # 灯
            'gauges': [],  # 仪表
            'systems': [],  # 系统
            'other': []  # 其他
        }

        # 处理每一页
        for page_data in content['pages']:
            page_num = page_data['page_number']
            page_text = page_data['text']

            # 查找连接器
            connectors = self._find_connectors(page_text, page_num)
            components['connectors'].extend(connectors)

            # 查找零件号
            part_components = self._find_part_components(page_text, page_num)
            for comp in part_components:
                comp_type = comp.get('type', 'other')
                if comp_type in components:
                    components[comp_type].append(comp)
                else:
                    components['other'].append(comp)

            # 查找关键词相关的元器件
            keyword_components = self._find_by_keywords(page_text, page_num)
            for comp in keyword_components:
                comp_type = comp.get('type', 'other')
                if comp_type in components:
                    components[comp_type].append(comp)
                else:
                    components['other'].append(comp)

        # 处理表格中的元器件信息
        for table_data in content['tables']:
            table_components = self._parse_table_components(table_data)
            for comp in table_components:
                comp_type = comp.get('type', 'other')
                if comp_type in components:
                    components[comp_type].append(comp)
                else:
                    components['other'].append(comp)

        # 统计信息
        print(f"识别完成:")
        for category, comp_list in components.items():
            if comp_list:
                print(f"  {self._get_category_name(category)}: {len(comp_list)}个")

        return components

    def _find_connectors(self, text, page_num):
        """查找连接器"""
        connectors = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 查找连接器编号
            for pattern in self.connector_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        connector = match[0]
                        pin = match[1] if len(match) > 1 else None
                    else:
                        connector = match
                        pin = None

                    # 判断是否为新连接器
                    connector_info = {
                        'name': f"{connector}连接器",
                        'type': 'connectors',
                        'code': connector,
                        'pin': pin,
                        'page': page_num,
                        'line': line_num,
                        'full_text': line[:100],
                        'function': self._guess_connector_function(connector, line)
                    }

                    connectors.append(connector_info)

            # 查找连接器序列（如 C2P1 → C2P2 → ...）
            if '→' in line or '->' in line:
                sequence_info = self._parse_connector_sequence(line, page_num, line_num)
                if sequence_info:
                    connectors.append(sequence_info)

        return connectors

    def _guess_connector_function(self, connector_code, context):
        """猜测连接器功能"""
        context_lower = context.lower()

        if connector_code.startswith('C'):
            if '发动机' in context or 'engine' in context_lower:
                return '发动机相关连接'
            elif 'ECU' in context or '电脑' in context:
                return '控制单元连接'
            elif '传感器' in context or 'sensor' in context_lower:
                return '传感器连接'
            elif '电源' in context or 'power' in context_lower:
                return '电源连接'

        if connector_code.startswith('J'):
            return '跳线连接器'

        if connector_code.startswith('X'):
            return '特殊功能连接器'

        if connector_code.startswith('S'):
            return '传感器连接器'

        return '通用连接器'

    def _parse_connector_sequence(self, line, page_num, line_num):
        """解析连接器序列"""
        # 提取所有连接器针脚
        pins = re.findall(r'[A-Z][0-9]+P[0-9]+', line)
        if pins:
            connector = pins[0][:pins[0].find('P')]
            pin_numbers = [pin[pin.find('P') + 1:] for pin in pins]

            return {
                'name': f'{connector}连接器序列',
                'type': 'connectors',
                'code': connector,
                'page': page_num,
                'line': line_num,
                'total_pins': len(pin_numbers),
                'pin_range': f"{min(pin_numbers, key=int)}-{max(pin_numbers, key=int)}",
                'is_sequence': True,
                'sequence_text': line[:150]
            }

        return None

    def _find_part_components(self, text, page_num):
        """查找零件号对应的元器件"""
        components = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 查找零件号
            for pattern in self.part_number_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        part_num = match[0]
                    else:
                        part_num = match

                    # 跳过太短的匹配
                    if len(part_num) < 6:
                        continue

                    # 创建元器件信息
                    component = self._create_component_from_part(part_num, line, page_num, line_num)
                    if component:
                        components.append(component)

        return components

    def _create_component_from_part(self, part_num, context, page_num, line_num):
        """从零件号创建元器件信息"""
        # 检查是否为已知部件
        if part_num in self.known_components:
            known_info = self.known_components[part_num]
            return {
                'name': known_info['name'],
                'type': known_info['type'],
                'code': part_num,
                'page': page_num,
                'line': line_num,
                'is_known': True,
                'description': known_info.get('description', ''),
                'full_text': context[:100]
            }

        # 根据零件号模式推断
        component_info = {
            'name': '',
            'type': 'other',
            'code': part_num,
            'page': page_num,
            'line': line_num,
            'is_known': False,
            'full_text': context[:100]
        }

        # 分析零件号特征
        if 'CA' in part_num:
            component_info['type'] = 'harnesses'

            if 'CA1251' in part_num:
                component_info['name'] = '一汽解放J6L整车主线束'
                component_info['description'] = '新款J6L车型整车电气线束总成'
            elif 'CA1234' in part_num:
                component_info['name'] = '发动机控制线束'
                component_info['description'] = '发动机相关传感器和执行器线束'
            elif 'CA1181' in part_num:
                component_info['name'] = '驾驶室电气线束'
                component_info['description'] = '驾驶室内开关、仪表、控制面板线束'

        elif 'S100001' in part_num:
            component_info['name'] = 'FA10气驱罐尿素系统线束'
            component_info['type'] = 'harnesses'
            component_info['description'] = '锡柴自主FA10发动机气驱尿素罐专用线束'

        elif 'Z00231' in part_num:
            component_info['name'] = '整车线束图纸文件'
            component_info['type'] = 'other'
            component_info['description'] = '线束设计图纸文档'

        elif 'Q00070' in part_num:
            component_info['name'] = '国六排放系统线束'
            component_info['type'] = 'harnesses'
            component_info['description'] = '国六排放后处理系统专用线束'

        else:
            # 通用零件号处理
            component_info['name'] = f"零件_{part_num[:12]}"
            component_info['description'] = '汽车电气线束组件'

            # 根据特征进一步分类
            if any(keyword in part_num for keyword in ['P62', 'K1', 'L7', 'T3', 'E5']):
                component_info['description'] = '主线束组件，含控制线和电源线'

        return component_info

    def _find_by_keywords(self, text, page_num):
        """根据关键词查找元器件"""
        components = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # 检查是否包含元器件关键词
            for category, keywords in self.component_dictionary.items():
                for keyword in keywords:
                    if keyword in line:
                        # 提取包含关键词的完整名称
                        component_name = self._extract_component_name(line, keyword)

                        component_info = {
                            'name': component_name,
                            'type': self._map_category_to_type(category),
                            'code': f"KW_{keyword}_{page_num}_{line_num}",
                            'page': page_num,
                            'line': line_num,
                            'keyword': keyword,
                            'full_text': line[:150],
                            'description': self._get_description_by_keyword(keyword)
                        }

                        components.append(component_info)
                        break  # 每个关键词只匹配一次

        return components

    def _extract_component_name(self, line, keyword):
        """从文本行中提取元器件完整名称"""
        # 查找包含关键词的短语
        words = line.split()
        for i, word in enumerate(words):
            if keyword in word:
                # 尝试获取前后词汇组成完整名称
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                name = ' '.join(words[start:end])
                return name

        # 如果没找到合适的上下文，返回包含关键词的部分
        return line[:50]

    def _map_category_to_type(self, category):
        """将分类映射到元器件类型"""
        type_mapping = {
            '连接器': 'connectors',
            '线束': 'harnesses',
            '传感器': 'sensors',
            '开关': 'switches',
            '继电器': 'relays',
            '保险': 'fuses',
            '模块': 'modules',
            '电机': 'motors',
            '泵': 'pumps',
            '阀': 'valves',
            '灯': 'lights',
            '仪表': 'gauges',
            '发动机系统': 'systems',
            '排放系统': 'systems',
            '电气系统': 'systems',
            '制动系统': 'systems',
            '转向系统': 'systems',
            '空调系统': 'systems',
            '安全系统': 'systems',
        }

        return type_mapping.get(category, 'other')

    def _get_description_by_keyword(self, keyword):
        """根据关键词获取描述"""
        descriptions = {
            'AdBlue': '柴油机尾气处理液系统，用于减少氮氧化物排放',
            '尿素': '选择性催化还原系统(SCR)的还原剂',
            'FA10': '锡柴自主10升发动机，国六排放标准',
            'ECU': '电子控制单元，车辆控制核心',
            '传感器': '用于检测各种物理量的装置',
            '线束': '车辆电气系统的导线束总成',
            '连接器': '电气连接装置，用于连接不同线束或部件',
        }

        for key, desc in descriptions.items():
            if key in keyword or keyword in key:
                return desc

        return '汽车电气系统组件'

    def _parse_table_components(self, table_data):
        """解析表格中的元器件信息"""
        components = []

        if not table_data['data']:
            return components

        # 假设表格的第一行是表头
        table = table_data['data']

        # 常见的元器件表格列名
        component_columns = ['名称', '代号', '零件号', '规格', '数量', '备注']

        # 检查表头是否包含元器件信息
        header = table[0] if table else []
        header_lower = [str(cell).lower() if cell else '' for cell in header]

        for row_num, row in enumerate(table[1:], 1):  # 跳过表头
            row_dict = {}
            for col_num, cell in enumerate(row):
                if col_num < len(header):
                    col_name = header[col_num] or f"列{col_num + 1}"
                    row_dict[col_name] = str(cell) if cell else ''

            # 从行数据中提取元器件信息
            component = self._extract_component_from_table_row(row_dict, table_data['page'], row_num)
            if component:
                components.append(component)

        return components

    def _extract_component_from_table_row(self, row_data, page_num, row_num):
        """从表格行中提取元器件信息"""
        # 查找可能包含元器件信息的字段
        component_info = {}

        for field_name, value in row_data.items():
            if not value:
                continue

            # 检查字段名是否暗示元器件信息
            field_lower = field_name.lower()
            value_str = str(value)

            if any(key in field_lower for key in ['名称', 'name', 'desc']):
                component_info['name'] = value_str

            elif any(key in field_lower for key in ['代号', '代码', 'code', '编号']):
                component_info['code'] = value_str

            elif any(key in field_lower for key in ['零件号', 'part', '型号']):
                component_info['part_number'] = value_str

            elif any(key in field_lower for key in ['类型', 'type', '类别']):
                component_info['type'] = value_str

            elif any(key in field_lower for key in ['规格', 'spec', '参数']):
                component_info['spec'] = value_str

            elif any(key in field_lower for key in ['数量', 'qty', 'quantity']):
                component_info['quantity'] = value_str

            elif any(key in field_lower for key in ['备注', 'note', 'comment']):
                component_info['description'] = value_str

        # 如果有足够的信息，创建元器件对象
        if component_info.get('name') or component_info.get('code'):
            component = {
                'name': component_info.get('name', component_info.get('code', '未命名')),
                'type': component_info.get('type', 'other'),
                'code': component_info.get('code', component_info.get('part_number', f"TABLE_{page_num}_{row_num}")),
                'page': page_num,
                'row': row_num,
                'source': 'table',
                'spec': component_info.get('spec', ''),
                'quantity': component_info.get('quantity', ''),
                'description': component_info.get('description', '')
            }
            return component

        return None

    def analyze_systems(self, components):
        """分析整车系统架构"""
        print("\n分析整车系统架构...")

        systems = {
            '动力总成系统': {
                'components': [],
                'subsystems': ['发动机系统', '变速器系统', '传动系统']
            },
            '排放控制系统': {
                'components': [],
                'subsystems': ['SCR系统', 'DPF系统', 'EGR系统']
            },
            '电气系统': {
                'components': [],
                'subsystems': ['电源系统', '照明系统', '仪表系统']
            },
            '底盘系统': {
                'components': [],
                'subsystems': ['制动系统', '转向系统', '悬挂系统']
            },
            '车身系统': {
                'components': [],
                'subsystems': ['空调系统', '安全系统', '舒适系统']
            }
        }

        # 将元器件归类到系统
        for category, comp_list in components.items():
            if not isinstance(comp_list, list):
                continue

            for comp in comp_list:
                if isinstance(comp, dict):  # 确保comp是字典
                    system = self._classify_to_system(comp)
                    if system:
                        systems[system]['components'].append(comp)

        # 统计各系统元器件数量
        for system_name, system_data in systems.items():
            system_data['count'] = len(system_data['components'])

        return systems

    def _classify_to_system(self, component):
        """将元器件分类到系统"""
        if not isinstance(component, dict):
            return None

        name = component.get('name', '').lower()
        comp_type = component.get('type', '')
        description = component.get('description', '').lower()

        # 检查关键词
        if any(keyword in name or keyword in description
               for keyword in ['发动机', '引擎', 'fa10', '锡柴', '燃油']):
            return '动力总成系统'

        elif any(keyword in name or keyword in description
                 for keyword in ['尿素', 'adblue', '排放', '尾气', 'scr', '国六']):
            return '排放控制系统'

        elif any(keyword in name or keyword in description
                 for keyword in ['线束', '连接器', '电缆', '电源', '蓄电池']):
            return '电气系统'

        elif any(keyword in name or keyword in description
                 for keyword in ['制动', '刹车', '转向', '悬挂', '底盘']):
            return '底盘系统'

        elif any(keyword in name or keyword in description
                 for keyword in ['空调', '暖风', '安全', '气囊', '舒适']):
            return '车身系统'

        return None

    def generate_comprehensive_report(self, content, components, systems):
        """生成综合报告"""
        report = []
        report.append("=" * 100)
        report.append("一汽解放新款J6L整车线束图 - 元器件解析综合报告")
        report.append("=" * 100)

        # 基本信息
        report.append(f"\n📋 基本信息:")
        report.append(f"   文件: {content['metadata'].get('file_name', 'N/A')}")
        report.append(f"   总页数: {content['metadata'].get('total_pages', 0)}")
        report.append(f"   表格数量: {len(content['tables'])}")
        report.append(f"   解析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 元器件统计
        report.append(f"\n📊 元器件统计:")
        total_components = sum(len(comp_list) for comp_list in components.values()
                               if isinstance(comp_list, list))
        report.append(f"   元器件总数: {total_components}")

        for category, comp_list in components.items():
            if isinstance(comp_list, list) and comp_list:
                report.append(f"   {self._get_category_name(category)}: {len(comp_list)}个")

        # 系统架构分析
        report.append(f"\n🏗️  系统架构分析:")
        for system_name, system_data in systems.items():
            count = system_data.get('count', 0)
            if count > 0:
                report.append(f"   {system_name}: {count}个元器件")

        # 详细元器件列表（按类型）
        report.append(f"\n🔧 详细元器件列表:")

        # 连接器详情
        if components.get('connectors'):
            connectors = components['connectors']
            report.append(f"\n   连接器汇总 ({len(connectors)}个):")

            # 按连接器类型分组
            connector_types = {}
            for conn in connectors:
                if isinstance(conn, dict):
                    conn_type = conn.get('code', '')[0] if conn.get('code') else '其他'
                    if conn_type not in connector_types:
                        connector_types[conn_type] = []
                    connector_types[conn_type].append(conn)

            for conn_type, conn_list in connector_types.items():
                if conn_list:
                    report.append(f"\n     {conn_type}系列连接器 ({len(conn_list)}个):")
                    for i, conn in enumerate(conn_list[:5], 1):
                        pin_info = f", 针脚{conn.get('pin')}" if conn.get('pin') else ""
                        report.append(f"       {i}. {conn.get('name')} (代码: {conn.get('code')}{pin_info})")

                    if len(conn_list) > 5:
                        report.append(f"       ... 还有 {len(conn_list) - 5} 个")

        # 线束详情
        if components.get('harnesses'):
            harnesses = components['harnesses']
            report.append(f"\n   线束汇总 ({len(harnesses)}个):")
            for i, harness in enumerate(harnesses[:10], 1):
                if isinstance(harness, dict):
                    report.append(f"     {i}. {harness.get('name', '未命名线束')}")
                    if harness.get('description'):
                        report.append(f"        描述: {harness.get('description')}")
                    if harness.get('code'):
                        report.append(f"        编码: {harness.get('code')}")

        # 关键系统组件
        report.append(f"\n🎯 关键系统组件:")

        # AdBlue系统
        adblue_components = []
        for category, comp_list in components.items():
            if isinstance(comp_list, list):
                for comp in comp_list:
                    if isinstance(comp, dict):
                        if any(keyword in comp.get('name', '').lower()
                               for keyword in ['adblue', '尿素']):
                            adblue_components.append(comp)

        if adblue_components:
            report.append(f"\n   排放控制系统 (AdBlue/尿素系统):")
            for i, comp in enumerate(adblue_components[:5], 1):
                if isinstance(comp, dict):
                    report.append(f"     {i}. {comp.get('name', '未命名')}")
                    if comp.get('description'):
                        report.append(f"        {comp.get('description')}")

        # FA10发动机相关
        fa10_components = []
        for category, comp_list in components.items():
            if isinstance(comp_list, list):
                for comp in comp_list:
                    if isinstance(comp, dict):
                        if any(keyword in comp.get('name', '').lower()
                               for keyword in ['fa10', '锡柴', '发动机']):
                            fa10_components.append(comp)

        if fa10_components:
            report.append(f"\n   动力系统 (FA10发动机):")
            for i, comp in enumerate(fa10_components[:5], 1):
                if isinstance(comp, dict):
                    report.append(f"     {i}. {comp.get('name', '未命名')}")
                    if comp.get('description'):
                        report.append(f"        {comp.get('description')}")

        # 人员信息
        report.append(f"\n👥 相关设计人员:")

        # 从文本中提取人名
        all_text = content['text']
        chinese_names = re.findall(r'[\u4e00-\u9fff]{2,3}', all_text)
        unique_names = sorted(set(chinese_names))

        if unique_names:
            # 过滤常见非人名汉字
            common_names = [name for name in unique_names
                            if name not in ['一汽', '解放', '新款', '整车', '线束', '图例']]

            for name in common_names[:10]:
                report.append(f"   • {name}")

            if len(common_names) > 10:
                report.append(f"   ... 还有 {len(common_names) - 10} 人")

        # 零件号清单
        report.append(f"\n🏷️  重要零件号清单:")

        # 收集所有零件号
        part_numbers = set()
        for category, comp_list in components.items():
            if isinstance(comp_list, list):
                for comp in comp_list:
                    if isinstance(comp, dict):
                        if comp.get('code') and len(comp.get('code', '')) > 8:
                            part_numbers.add(comp['code'])

        for i, part in enumerate(sorted(part_numbers)[:15], 1):
            report.append(f"   {i:2d}. {part}")

        if len(part_numbers) > 15:
            report.append(f"   ... 还有 {len(part_numbers) - 15} 个零件号")

        report.append("\n" + "=" * 100)
        report.append("报告生成完成")
        report.append("=" * 100)

        return "\n".join(report)

    def _get_category_name(self, category):
        """获取分类的中文名称"""
        category_names = {
            'connectors': '连接器',
            'harnesses': '线束',
            'sensors': '传感器',
            'switches': '开关',
            'relays': '继电器',
            'fuses': '保险丝',
            'modules': '模块',
            'motors': '电机',
            'pumps': '泵',
            'valves': '阀',
            'lights': '灯具',
            'gauges': '仪表',
            'systems': '系统',
            'other': '其他'
        }
        return category_names.get(category, category)

    def export_detailed_data(self, components, systems, filename):
        """导出详细数据到JSON文件"""
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'component_categories': len(components),
                'total_components': sum(len(comp_list) for comp_list in components.values()
                                        if isinstance(comp_list, list)),
                'systems_analyzed': len(systems)
            },
            'components_by_category': components,
            'systems_architecture': systems,
            'summary': {
                'connector_count': len(components.get('connectors', [])),
                'harness_count': len(components.get('harnesses', [])),
                'sensor_count': len(components.get('sensors', [])),
                'key_systems': [
                    system for system, data in systems.items()
                    if data.get('count', 0) > 0
                ]
            }
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 详细数据已导出到: {filename}")
            return True
        except Exception as e:
            print(f"❌ 导出数据时出错: {e}")
            return False

    def export_component_list(self, components, filename):
        """导出元器件列表到CSV文件"""
        all_components = []

        for category, comp_list in components.items():
            if isinstance(comp_list, list):
                for comp in comp_list:
                    if isinstance(comp, dict):
                        component_row = {
                            '元器件名称': comp.get('name', ''),
                            '类型': self._get_category_name(category),
                            '编码': comp.get('code', ''),
                            '所在页': comp.get('page', ''),
                            '所在行': comp.get('line', comp.get('row', '')),
                            '描述': comp.get('description', ''),
                            '规格': comp.get('spec', ''),
                            '数量': comp.get('quantity', ''),
                            '来源': comp.get('source', 'text')
                        }
                        all_components.append(component_row)

        if all_components:
            df = pd.DataFrame(all_components)
            df = df.drop_duplicates(subset=['元器件名称', '编码'])
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ 元器件列表已导出到: {filename} ({len(df)}条记录)")
            return True

        print("没有元器件数据可导出")
        return False


def main():
    """主程序"""
    print("=" * 80)
    print("一汽解放J6L整车线束图元器件解析系统")
    print("=" * 80)

    # PDF文件路径
    pdf_files = [
        "一汽解放_新款J6L_整车线束图【含各类选装配置】【锡柴自主FA10-气驱罐】【国六】.pdf",
        "test.pdf"
    ]

    pdf_path = None
    for file in pdf_files:
        if os.path.exists(file):
            pdf_path = file
            print(f"找到PDF文件: {file}")
            break

    if not pdf_path:
        print("未找到PDF文件，请将文件放在当前目录下")
        print("当前目录文件:", os.listdir('.'))
        return

    # 创建解析器
    parser = AutomotiveHarnessParser()

    # 提取PDF内容
    print("\n步骤1: 提取PDF内容...")
    content = parser.extract_all_content(pdf_path)
    if not content:
        print("无法提取PDF内容")
        return

    # 查找元器件
    print("\n步骤2: 识别元器件...")
    components = parser.find_all_components(content)

    # 分析系统架构
    print("\n步骤3: 分析系统架构...")
    systems = parser.analyze_systems(components)

    # 生成报告
    print("\n步骤4: 生成报告...")
    report = parser.generate_comprehensive_report(content, components, systems)
    print(report)

    # 导出数据
    print("\n步骤5: 导出数据...")
    parser.export_detailed_data(components, systems, 'harness_analysis_detailed.json')
    parser.export_component_list(components, 'components_detailed.csv')

    # 显示关键发现
    print("\n" + "=" * 80)
    print("关键发现摘要:")
    print("=" * 80)

    # 连接器统计
    connector_count = len(components.get('connectors', []))
    if connector_count > 0:
        print(f"\n🔌 发现 {connector_count} 个连接器:")

        # 按连接器类型统计
        connector_types = {}
        for conn in components.get('connectors', []):
            if isinstance(conn, dict):
                conn_code = conn.get('code', '')
                if conn_code:
                    conn_type = conn_code[0] if conn_code else '其他'
                    connector_types[conn_type] = connector_types.get(conn_type, 0) + 1

        for conn_type, count in connector_types.items():
            print(f"   {conn_type}系列: {count}个")

    # 线束统计
    harness_count = len(components.get('harnesses', []))
    if harness_count > 0:
        print(f"\n🔌 发现 {harness_count} 个线束组件:")
        for i, harness in enumerate(components.get('harnesses', [])[:5], 1):
            if isinstance(harness, dict):
                print(f"   {i}. {harness.get('name', '未命名线束')}")

    # 关键系统
    print(f"\n🚗 关键系统识别:")
    for system_name, system_data in systems.items():
        count = system_data.get('count', 0)
        if count > 0:
            print(f"   • {system_name}: {count}个相关组件")

    total_components = sum(len(comp_list) for comp_list in components.values()
                           if isinstance(comp_list, list))
    print(f"\n📈 总计: {total_components} 个元器件被识别")


if __name__ == "__main__":
    # 检查依赖库
    try:
        import pdfplumber
        import pandas as pd

        print("✓ 所需库已安装")
    except ImportError as e:
        print(f"需要安装依赖库: {e}")
        print("请运行: pip install pdfplumber pandas")
        exit(1)

    main()