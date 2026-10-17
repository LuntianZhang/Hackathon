from pathlib import Path

from food3d.io import read_json
from food3d.food_catalog import FoodIndex,normalize_name

FIELDS = ('kcal', 'protein', 'carbs', 'fat')

class Catalog:
    def __init__(self, path: Path):
        self.data = read_json(path)
        self.foods = {entry['key']: entry for entry in self.data['foods']}
        self.index=FoodIndex(self.data['foods'])

    def match(self, name: str) -> str | None:
        return self.index.match(name)

    def estimate(self, key: str | None) -> dict | None:
        if key is None:
            return None
        if key not in self.foods:
            raise ValueError('请选择食物目录中的名称，或使用自定义名称')
        food = self.foods[key]
        if food.get('per_100g') is None:return None
        return {**{field: round(food['per_100g'][field]*food['reference_grams']/100, 3) for field in FIELDS},
                'reference_grams': food['reference_grams'], 'basis': food['portion_basis'],
                'source': self.data['source'], 'ndb_number': food['ndb_number'],
                'reference_name':food['name'],
                'source_description': food['source_description']}

    def search(self,query='',category=None):
        query=normalize_name(query)
        return [food for food in self.foods.values() if (not category or food.get('category')==category)
            and (not query or any(query in normalize_name(value) for value in
                [food['name'],food['key'],food.get('recognition_label',''),*food.get('aliases',[])]))]

    def categories(self):
        return [{**category,'count':sum(f.get('category')==category['key'] for f in self.foods.values())}
            for category in self.data.get('categories',[])]

def total_nutrition(items: list[dict]) -> dict:
    known = [item['nutrition'] for item in items if item.get('nutrition') is not None]
    return {**{field: round(sum(value[field] for value in known), 3) for field in FIELDS},
            'known_items': len(known), 'unknown_items': len(items)-len(known),
            'volume_density_items':sum(value.get('portion_method')=='volume_density' for value in known),
            'reference_items':sum(value.get('portion_method')!='volume_density' for value in known),
            'quantity_pending_items':sum(item.get('reference_nutrition') is not None and item.get('nutrition') is None and item.get('mass_estimate',{}).get('status') in {'density_pending','volume_unavailable'} for item in items),
            'estimated': True, 'complete': len(known) == len(items)}
