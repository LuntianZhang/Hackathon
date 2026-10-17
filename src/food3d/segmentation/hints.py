"""Validated, model-independent input shared by photo and video recognition."""
from __future__ import annotations

import math
import re

from .recognition import NAMES_ZH
from ..food_catalog import FoodIndex,visual_label,normalize_name

REJECT_LABELS = ('table', 'plate', 'background', 'unknown')
ALIASES = {**{value: key for key, value in NAMES_ZH.items()},
    '米饭': 'rice', '鸡肉': 'chicken', '鸡腿': 'chicken', '西兰花': 'broccoli',
    '鸡蛋': 'egg', '土豆': 'potato', '三文鱼': 'salmon', '胡萝卜': 'carrot',
    '黄瓜': 'cucumber', '番茄': 'tomato', '西红柿': 'tomato', '豆腐': 'tofu',
    '牛肉': 'beef', '猪肉': 'pork', '牛奶': 'milk', '酸奶': 'yogurt',
    '草莓': 'strawberry', '蓝莓': 'blueberry', '李子': 'plum', '青李子': 'green plum',
    '紫李子': 'purple plum', '猕猴桃': 'kiwi', '桃子': 'peach', '桃': 'peach',
    '柠檬': 'lemon', '西瓜': 'watermelon', '橘子': 'mandarin orange',
    '青苹果': 'green apple', '红苹果': 'red apple', '葡萄': 'grapes'}


def parse_foods(text: str, catalog: list[dict] = ()) -> list[dict]:
    """Chinese aliases or `display name|English description`; no hidden translation API."""
    if not isinstance(text, str) or len(text) > 2000:
        raise ValueError('食物清单最多 2000 字符')
    index=FoodIndex(catalog)
    aliases={normalize_name(name):label for name,label in ALIASES.items()}
    foods = []
    seen = set()
    for token in re.split(r'[,，、;；\n]+', text):
        token = token.strip()
        if not token:
            continue
        parts = token.split('|')
        if len(parts) > 2:
            raise ValueError('自定义名称格式为：中文名称|English description')
        name = parts[0].strip()
        key=index.match(name)
        query = (parts[1].strip() if len(parts) == 2 else
            visual_label(index.foods[key]) if key else aliases.get(normalize_name(name),name)).lower()
        if not name or not query or len(name) > 80 or len(query) > 100:
            raise ValueError('名称最多 80 字符，英文描述最多 100 字符')
        if not re.fullmatch(r"[a-z][a-z0-9 '\-]*", query):
            raise ValueError(f'“{name}”尚无英文映射，请填写“{name}|English name”')
        if query in REJECT_LABELS or name in ('背景', '桌面', '盘子', '未知'):
            raise ValueError('食物清单不应包含背景、桌面、盘子或未知')
        if query not in seen:
            foods.append({'name': name, 'text': query})
            seen.add(query)
    if len(foods) > 20:
        raise ValueError('最多输入 20 类食物；同类多个对象会分别分配实例 ID')
    return foods


def validate_assistance(value: dict | None, catalog: list[dict] = (), *, allow_none=False) -> dict:
    if value is None:
        return {'mode': 'auto', 'foods': [], 'instances': []}
    if not isinstance(value, dict) or set(value) - {'mode', 'food_text', 'foods', 'instances'}:
        raise ValueError('辅助识别配置格式不正确')
    mode = value.get('mode', 'list')
    if mode not in {'auto', 'list', 'grounded', 'manual', 'llm'} and not (allow_none and mode == 'none'):
        raise ValueError('请选择清单辅助、文字定位、点击框选或大模型识别')
    if mode == 'none':
        if value.get('food_text') or value.get('foods') or value.get('instances'):
            raise ValueError('不使用语义分割时无需提供食物提示或区域')
        return {'mode': 'none', 'foods': [], 'instances': []}
    if 'food_text' in value:
        foods = parse_foods(value['food_text'], catalog)
    else:
        raw = value.get('foods', [])
        if not isinstance(raw, list) or any(not isinstance(f, dict) or set(f) != {'name', 'text'} for f in raw):
            raise ValueError('候选食物格式不正确')
        foods = parse_foods('\n'.join(f"{f['name']}|{f['text']}" for f in raw), catalog)
    if mode in {'list', 'grounded'} and not foods:
        raise ValueError('请先输入可能有哪些食物')
    if mode == 'llm' and foods:
        raise ValueError('大模型识别会自动判断对象，无需输入食物清单')
    raw_instances = value.get('instances', [])
    if not isinstance(raw_instances, list) or len(raw_instances) > 20:
        raise ValueError('最多指定 20 个食物实例')
    instances = []
    used = set()
    for index, entry in enumerate(raw_instances, 1):
        if not isinstance(entry, dict) or set(entry) - {'instance_id', 'name', 'point', 'box'}:
            raise ValueError('点击或框选提示格式不正确')
        name = entry.get('name', '')
        identifier = entry.get('instance_id', index)
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError('每个区域需要填写名称，最多 80 字符')
        if name.strip() in ('背景', '桌面', '盘子', '未知'):
            raise ValueError('请标记食物区域')
        if type(identifier) is not int or not 0 < identifier < 65535 or identifier in used:
            raise ValueError('实例 ID 必须唯一且介于 1 和 65534')
        if ('point' in entry) == ('box' in entry):
            raise ValueError('每个实例需指定一个点击或一个框选')
        kind = 'point' if 'point' in entry else 'box'
        coordinates = entry[kind]
        if (not isinstance(coordinates, list) or len(coordinates) != (2 if kind == 'point' else 4)
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in coordinates)):
            raise ValueError('区域坐标必须是照片内的归一化坐标（0 至 1）')
        if kind == 'box' and not (coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]):
            raise ValueError('框选区域必须具有正面积')
        instances.append({'instance_id': identifier, 'name': name.strip(), kind: coordinates})
        used.add(identifier)
    if mode == 'manual' and not instances:
        raise ValueError('请先点击或框选至少一个食物')
    if mode != 'manual' and instances:
        raise ValueError('只有点击框选方式接受人工区域')
    return {'mode': mode, 'foods': foods, 'instances': instances}


def candidate_names(assistance: dict, defaults: list[str]) -> list[str]:
    # Non-hinted known foods compete as rejection alternatives. A wrong hint
    # must not automatically turn a banana into the only requested "bread".
    labels = [f['text'] for f in assistance['foods']] + defaults
    return list(dict.fromkeys([*labels, *REJECT_LABELS]))


def allowed_food(assistance: dict, label: str) -> bool:
    return label not in REJECT_LABELS and (assistance['mode'] not in {'list', 'grounded'}
        or label in {f['text'] for f in assistance['foods']})


def display_names(assistance: dict, catalog: list[dict] = ()) -> dict[str, str]:
    return {**NAMES_ZH, **{f['key']: f['name'] for f in catalog},
        **{visual_label(f):f['name'] for f in catalog}, **{f['text']: f['name'] for f in assistance['foods']}}
