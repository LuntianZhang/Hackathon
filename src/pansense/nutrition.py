from pathlib import Path

from food3d.io import read_json

FIELDS = ('kcal', 'protein', 'carbs', 'fat')

class Catalog:
    def __init__(self, path: Path):
        self.data = read_json(path)
        self.foods = {entry['key']: entry for entry in self.data['foods']}

    def match(self, name: str) -> str | None:
        aliases = {'米饭': 'rice', '面包': 'bread', '鸡腿': 'chicken', '鸡肉': 'chicken',
                   '西兰花': 'broccoli', '鸡蛋': 'egg', '土豆': 'potato', '三文鱼': 'salmon',
                   '胡萝卜': 'carrot', '黄瓜': 'cucumber'}
        if name in aliases:
            return aliases[name]
        for key, food in self.foods.items():
            if name.strip().casefold() in (key.casefold(), food['name'].casefold()):
                return key
        return None

    def estimate(self, key: str | None) -> dict | None:
        if key is None:
            return None
        if key not in self.foods:
            raise ValueError('请选择食物目录中的名称，或使用自定义名称')
        food = self.foods[key]
        return {**{field: round(food['per_100g'][field]*food['reference_grams']/100, 3) for field in FIELDS},
                'reference_grams': food['reference_grams'], 'basis': food['portion_basis'],
                'source': self.data['source'], 'ndb_number': food['ndb_number'],
                'source_description': food['source_description']}

def total_nutrition(items: list[dict]) -> dict:
    known = [item['nutrition'] for item in items if item.get('nutrition') is not None]
    return {**{field: round(sum(value[field] for value in known), 3) for field in FIELDS},
            'known_items': len(known), 'unknown_items': len(items)-len(known),
            'estimated': True, 'complete': len(known) == len(items)}
