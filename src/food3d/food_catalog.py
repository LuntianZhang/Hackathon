"""Shared exact-name and alias matching for nutrition and visual recognition."""
import re
import unicodedata


def normalize_name(name):
    if not isinstance(name,str):return ''
    return re.sub(r'[\s_]+',' ',unicodedata.normalize('NFKC',name).strip().casefold())


def visual_label(food):
    return food.get('recognition_label') or food['key'].replace('_',' ')


class FoodIndex:
    def __init__(self,foods):
        self.foods={food['key']:food for food in foods}
        if len(self.foods)!=len(foods):raise ValueError('食物库包含重复标识')
        self.names={}
        for food in foods:
            for name in [food['key'],food['name'],visual_label(food),*food.get('aliases',[])]:
                name=normalize_name(name)
                if not name:continue
                if name in self.names and self.names[name]!=food['key']:
                    raise ValueError(f'食物别名重复：{name}')
                self.names[name]=food['key']

    def match(self,name):
        # Never use substring or fuzzy matching: dishes, packages and specific
        # cooking methods must not silently inherit another food's nutrients.
        return self.names.get(normalize_name(name))

    def labels(self):
        return [visual_label(food) for food in self.foods.values()]
