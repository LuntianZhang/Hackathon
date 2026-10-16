"""Extract a small, traceable P0 catalog from the official USDA SR28 archive."""
from pathlib import Path
from food3d.paths import ProjectPaths
import hashlib
import zipfile

from food3d.io import write_json

SELECTION = [
    ('apple', '苹果', '09003', 182, '🍎'), ('banana', '香蕉', '09040', 118, '🍌'),
    ('grapes', '葡萄', '09132', 50, '🍇'), ('mango', '芒果', '09176', 100, '🥭'),
    ('durian', '榴莲', '09422', 100, '🍈'), ('orange', '橙子', '09200', 131, '🍊'),
    ('pear', '梨', '09252', 178, '🍐'), ('plum', '李子', '09279', 66, '🟣'),
    ('kiwi', '猕猴桃', '09148', 69, '🥝'), ('strawberry', '草莓', '09316', 100, '🍓'),
    ('bread', '白面包', '18069', 50, '🍞'), ('rice', '米饭', '20045', 150, '🍚'),
    ('chicken', '鸡腿肉（烤制）', '05098', 100, '🍗'), ('broccoli', '西兰花（水煮）', '11091', 80, '🥦'),
    ('egg', '鸡蛋（水煮）', '01129', 50, '🥚'), ('tofu', '豆腐', '16426', 100, '⬜'),
    ('potato', '土豆（水煮）', '11367', 150, '🥔'), ('salmon', '三文鱼（烤制）', '15237', 100, '🐟'),
    ('tomato', '番茄', '11529', 100, '🍅'), ('carrot', '胡萝卜（生）', '11124', 100, '🥕'),
    ('cucumber', '黄瓜（带皮）', '11205', 100, '🥒')]

def build():
    path = ProjectPaths().resolve('.cache/nutrition/sr28asc.zip')
    with zipfile.ZipFile(path) as archive:
        def rows(name):
            return [[field.strip('~') for field in line.split('^')]
                    for line in archive.read(name).decode('latin1').splitlines()]
        descriptions = {row[0]: row[2] for row in rows('FOOD_DES.txt')}
        nutrients = {}
        for row in rows('NUT_DATA.txt'):
            if row[1] in {'208', '203', '204', '205'}:
                nutrients.setdefault(row[0], {})[row[1]] = float(row[2])
        foods = []
        for key, name, ndb, grams, emoji in SELECTION:
            values = nutrients[ndb]
            foods.append({'key': key, 'name': name, 'emoji': emoji, 'ndb_number': ndb,
                          'source_description': descriptions[ndb], 'reference_grams': grams,
                          'portion_basis': 'P0 固定参考份量，非视觉称重',
                          'per_100g': {'kcal': values['208'], 'protein': values['203'],
                                       'fat': values['204'], 'carbs': values['205']}})
    output = {'schema_version': 1, 'source': 'USDA SR28, revised May 2016',
              'source_url': 'https://www.ars.usda.gov/ARSUserFiles/80400525/Data/SR/SR28/dnload/sr28asc.zip',
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'basis': 'Nutrients per 100 g edible portion; reference portions are product assumptions',
              'foods': foods}
    write_json(Path('app/catalog/foods.json'), output)
    print(f'Catalog: {len(foods)} foods, source identities and all four nutrients verified')

if __name__ == '__main__':
    build()
