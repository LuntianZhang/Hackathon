"""Build a traceable ingredient catalog from USDA SR28 and curated names."""
from pathlib import Path
from food3d.paths import ProjectPaths
import hashlib
import zipfile

from food3d.io import read_json,write_json

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

CATEGORY_EMOJI = {
    'fruit': '🍎', 'vegetable': '🥬', 'meat': '🍖', 'staple': '🍚',
    'dairy': '🥛', 'soy': '🫘', 'nut': '🥜', 'drink': '🥤',
    'snack': '🍪', 'dish': '🍲', 'condiment': '🧂', 'other': '🍽️',
}

def build(path=None,output_path=None,selection_path=None):
    project=ProjectPaths().project
    path=Path(path) if path else ProjectPaths().resolve('.cache/nutrition/sr28asc.zip')
    selection=read_json(selection_path or project/'app/catalog/selection.json')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=selection['source_sha256']:raise ValueError('USDA 源归档校验失败，拒绝生成营养数据')
    categories={entry['key']:entry['name'] for entry in selection['categories']}
    records=[]
    for key,name,ndb,grams,emoji in SELECTION:
        metadata=selection['legacy_metadata'][key]
        records.append({'key':key,'name':name,'ndb_number':ndb,'reference_grams':grams,
            'emoji':emoji,'recognition_label':key,**metadata})
    records.extend(selection['foods'])
    if len({f['key'] for f in records})!=len(records):raise ValueError('食物 key 重复')
    if len({f['recognition_label'] for f in records})!=len(records):raise ValueError('识别标签重复')
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
        for record in records:
            ndb=record['ndb_number'];category=record['category']
            if category not in categories:raise ValueError('食物分类不存在')
            values=nutrients.get(ndb)
            if ndb and (not values or set(values)!={'208','203','204','205'}):
                raise ValueError(f'{record["key"]} 缺少已核验的营养字段')
            if ndb and record.get('source_description',descriptions[ndb])!=descriptions[ndb]:
                raise ValueError(f'{record["key"]} 的来源条目已变化')
            foods.append({**record,'emoji':record.get('emoji',CATEGORY_EMOJI.get(category,'🍽️')),
                'category_label':categories[category],
                'source_description':descriptions[ndb] if ndb else None,
                'nutrient_status':'verified' if ndb else 'unavailable',
                'portion_basis':'P0 固定参考份量，非视觉称重',
                'per_100g':{'kcal':values['208'],'protein':values['203'],'fat':values['204'],'carbs':values['205']} if ndb else None})
    output = {'schema_version': 1, 'source': 'USDA SR28, revised May 2016',
              'source_url': 'https://www.ars.usda.gov/ARSUserFiles/80400525/Data/SR/SR28/dnload/sr28asc.zip',
              'source_sha256': digest,
              'basis': 'Nutrients per 100 g edible portion; reference portions are product assumptions',
              'categories':selection['categories'],
              'coverage':selection['coverage'],
              'nutrient_policy':'Only verified source values; unavailable data stays null, never zero or a substitute food.',
              'foods': foods}
    write_json(output_path or project/'app/catalog/foods.json', output)
    return output

if __name__ == '__main__':
    result=build()
    print(f'Catalog: {len(result["foods"])} foods, {sum(f["nutrient_status"]=="verified" for f in result["foods"])} verified nutrient records')
