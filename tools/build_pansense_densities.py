"""Build category fallback estimates from traceable household-volume samples.

These are portion/bulk estimates, not measured solid-object densities. They must
remain distinguishable from a food-specific reference and a manual override.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import re
from statistics import mean, median
import zipfile

from food3d.io import read_json, write_json
from food3d.paths import ProjectPaths


# US customary volume convention, made explicit rather than guessing a cup size.
# NIST customary volume conversion: US gallon = 3.785411784 L, cup = gallon / 16.
GALLON_ML = 3785.411784
UNITS_ML = {'cup': GALLON_ML / 16, 'cups': GALLON_ML / 16,
            'tbsp': GALLON_ML / 256, 'tsp': GALLON_ML / 768}
NIST_URL = 'https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b9'
FAO_URL = 'https://www.fao.org/fileadmin/templates/food_composition/documents/density_DB_v2_0_01.pdf'
# The Chinese prepared-dish entries have no SR28 IDs. Use a diverse, explicit
# subset of FAO soups/mixed dishes, without claiming these are Chinese recipes.
DISH_SAMPLES = [
    ('Chicken noodle soup', 1.0, 'S&W', 21),
    ('Egg drop soup', 1.017, 'S&W', 21),
    ('Mushroom soup', 1.017, 'S&W', 21),
    ('Soup, vegetable', .99, 'KEN', 21),
    ('Beef and noodles (no sauce)', .65, 'S&W', 21),
    ('Beef stew, canned', 1.05, 'S&W', 21),
    ('Broccoli casserole', .95, 'S&W', 21),
    ('Chicken and dumplings (mixture)', 1.04, 'S&W', 21),
    ('Lasagna', 1.04, 'S&W', 22),
    ('Rice + beans', .75, 'KEN', 22),
    ('Rice + meat', .73, 'KEN', 22),
    ('Rice + meat + vegetable', .83, 'KEN', 23),
    ('Rice + vegetable', .65, 'KEN', 23),
    ('Stew, meat', 1.12, 'KEN', 23),
    ('Stew, vegetable', .68, 'KEN', 23),
    ('Stew, vegetable + meat', .86, 'KEN', 23),
    ('Stew: eggs + veg', .68, 'KEN', 23),
]


def build(archive_path=None, catalog_path=None, output_path=None):
    project = ProjectPaths().project
    catalog = read_json(catalog_path or project / 'app/catalog/foods.json')
    archive_path = Path(archive_path or project / '.cache/nutrition/sr28asc.zip')
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != catalog['source_sha256']:
        raise ValueError('USDA 源归档校验失败，拒绝生成分类密度')
    weights = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for line in archive.read('WEIGHT.txt').decode('latin1').splitlines():
            row = [field.strip('~') for field in line.split('^')]
            match = re.fullmatch(r'(cups?|tbsp|tsp)(?:, .*)?', row[3], re.I)
            # Reject unit counts, dimensions, heaped spoons and reconstitution
            # instructions. They are not a direct occupied-volume measurement.
            if not match or re.search(r'(?i)yield|makes?|prepared|reconstitut|heaped|rounded|sifted before', row[3]):
                continue
            amount, grams = float(row[2]), float(row[4])
            if amount <= 0 or grams <= 0:
                continue
            ml = amount * UNITS_ML[match[1].lower()]
            weights[row[0]].append({'sequence': row[1], 'amount': amount, 'measure': row[3],
                'grams': grams, 'volume_ml': ml, 'value_g_ml': grams / ml})
    defaults = []
    for category in catalog['categories']:
        samples, seen = [], set()
        for food in catalog['foods']:
            ndb = food['ndb_number']
            if food['category'] != category['key'] or not weights[ndb] or ndb in seen:
                continue
            seen.add(ndb)
            measures = weights[ndb]
            samples.append({'food_key': food['key'], 'food_name': food['name'], 'ndb_number': ndb,
                'source_description': food['source_description'], 'measures': measures,
                'value_g_ml': median(row['value_g_ml'] for row in measures)})
        source, url = 'USDA SR28 量杯份量计算', 'https://fdc.nal.usda.gov/data-documentation.html'
        method = '同一 USDA 食品的各量杯密度取中位数，按来源食品编号去重后等权算术平均'
        if not samples and category['key'] == 'dish':
            samples = [{'source_description': name, 'value_g_ml': value,
                        'biblio_id': biblio, 'pdf_page': page,
                        'source_url': f'{FAO_URL}#page={page}'}
                       for name, value, biblio, page in DISH_SAMPLES]
            source, url = 'FAO/INFOODS v2 菜肴样本计算', FAO_URL
            method = '17 条汤、炖菜、混合主食参考密度的等权算术平均；并非各道中式菜的实测值'
        if not samples:
            raise ValueError(f"分类 {category['key']} 缺少密度样本")
        values = [row['value_g_ml'] for row in samples]
        defaults.append({'category': category['key'], 'category_label': category['name'],
            'name': f"{category['name']}分类默认估值", 'status': 'category_default',
            'value_g_ml': round(mean(values), 3), 'unit': 'g/mL', 'kind': 'category_bulk_estimate',
            'sample_count': len(samples), 'sample_range_g_ml': [round(min(values), 3), round(max(values), 3)],
            'method': method, 'source': source, 'source_url': url,
            'note': f'分类默认估值，基于 {len(samples)} 个食品样本；量杯／堆积体积含空隙，整颗、切块及生熟状态会有差异，可手动修改。',
            'samples': samples})
    result = {'schema_version': 1, 'policy': 'Manual selection > food-specific reference > category estimate; never applied to packaging or non-food.',
        'basis': 'Household-volume/bulk fallback estimates, not measured material densities or nutrition substitutes.',
        'usda_source_sha256': digest, 'volume_units_ml': UNITS_ML, 'unit_source_url': NIST_URL,
        'categories': defaults}
    write_json(output_path or project / 'app/catalog/category_densities.json', result)
    return result


if __name__ == '__main__':
    data = build()
    for row in data['categories']:
        print(f"{row['category']}: {row['value_g_ml']} g/mL ({row['sample_count']} samples)")
