"""Keep density, geometric volume and reference-portion nutrition distinct."""
from copy import deepcopy
import math
from pathlib import Path
from statistics import median

from food3d.food_catalog import normalize_name
from food3d.io import read_json
from .nutrition import FIELDS


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


class FoodQuantities:
    def __init__(self, catalog, density_path, measurement_reader=None):
        self.catalog = catalog
        self.references = {}
        self.category_defaults = {}
        if Path(density_path).is_file():
            data = read_json(Path(density_path))
            for entry in data['references']:
                if not positive(entry['value_g_ml']) or entry['id'] in self.references:
                    raise ValueError('密度资料库包含无效数值或重复标识')
                if not set(entry['catalog_keys']) <= set(catalog.foods):
                    raise ValueError('密度资料库引用了不存在的食物')
                self.references[entry['id']] = entry
        category_path = Path(density_path).with_name('category_densities.json')
        if category_path.is_file():
            for entry in read_json(category_path)['categories']:
                key = entry['category']
                if not positive(entry['value_g_ml']) or key in self.category_defaults:
                    raise ValueError('分类密度包含无效数值或重复分类')
                # Calculation samples stay in the versioned source file; send
                # only the estimate and provenance to clients.
                self.category_defaults[key] = {k: deepcopy(v) for k, v in entry.items() if k != 'samples'}
        if measurement_reader is None:
            from food3d.volume import published_measurements
            measurement_reader = published_measurements
        self.measurement_reader = measurement_reader

    def candidates(self, key):
        return [deepcopy(entry) for entry in self.references.values() if key in entry['catalog_keys']]

    def default_density(self, key):
        refs = self.candidates(key)
        if refs:
            entry = refs[0]
            if len(refs) > 1:
                entry = {**entry, 'name': '食品参考密度中位数', 'kind': 'food_reference_estimate',
                         'value_g_ml': median(ref['value_g_ml'] for ref in refs),
                         'sample_count': len(refs),
                         'note': f'同一食品的 {len(refs)} 条参考密度取中位数；请在修改中核对生熟、形态及装填状态。'}
            return {**entry, 'status': 'food_reference', 'automatic': True, 'catalog_key': key,
                    'message': '自动使用食品参考密度，非本餐实测'}
        food = self.catalog.foods.get(key)
        entry = self.category_defaults.get(food.get('category')) if food else None
        if entry:
            return {**deepcopy(entry), 'status': 'category_default', 'automatic': True, 'catalog_key': key,
                    'message': '缺少食品专属密度，使用分类默认估值，非本餐实测'}
        return {'status': 'missing', 'value_g_ml': None, 'candidate_count': 0,
                'message': '未匹配食品分类，请选择参考食品或手动填写密度'}

    @staticmethod
    def identity(item):
        return [normalize_name(item['name']), item.get('catalog_key')]

    def selection(self, item, choice):
        if choice is None: return item.get('density_selection')
        mode = choice.get('mode')
        if mode == 'clear': return None
        if item.get('object_kind') in {'non_food', 'packaged_food'}:
            raise ValueError('非食物对象和包装外形不能用于食物密度估重')
        if mode == 'reference':
            entry = self.references.get(choice.get('reference_id'))
            if not entry or item.get('catalog_key') not in entry['catalog_keys']:
                raise ValueError('所选参考密度与当前食品不匹配，请重新选择')
            return {'mode': mode, 'reference_id': entry['id'], 'food_identity': self.identity(item)}
        if mode == 'custom':
            value = choice.get('value_g_ml')
            if not positive(value) or value > 10: raise ValueError('密度须为大于 0 且不超过 10 的有限数，单位 g/mL')
            return {'mode': mode, 'value_g_ml': value, 'note': str(choice.get('note', ''))[:200],
                    'food_identity': self.identity(item)}
        raise ValueError('请选择参考密度、手动密度或清除密度')

    def density(self, item):
        if item.get('object_kind') in {'non_food', 'packaged_food'}:
            return {'status': 'blocked', 'value_g_ml': None, 'message': '非食物或包装外形，不换算食物重量'}
        choice = item.get('density_selection') or {}
        if choice.get('food_identity') == self.identity(item):
            if choice.get('mode') == 'reference':
                entry = self.references.get(choice.get('reference_id'))
                if entry and item.get('catalog_key') in entry['catalog_keys']:
                    return {**deepcopy(entry), 'status': 'reference', 'message': '已选择参考密度，非本餐实测'}
            if choice.get('mode') == 'custom' and positive(choice.get('value_g_ml')) and choice['value_g_ml'] <= 10:
                return {'status': 'custom', 'value_g_ml': choice['value_g_ml'], 'source': '用户填写',
                        'source_url': None, 'note': choice.get('note', ''), 'kind': 'user_reference',
                        'message': '用户填写密度，非系统实测'}
        # Older meals can predate a catalog entry. Match exact names/aliases at
        # read time, without changing their saved identity or manual overrides.
        key = item.get('catalog_key') or self.catalog.match(item['name'])
        return self.default_density(key)

    @staticmethod
    def scaled_nutrition(reference, grams, density, volume):
        original = reference.get('reference_grams')
        if not positive(original): return None
        if any(not isinstance(reference.get(field), (int, float)) or not math.isfinite(reference[field]) or reference[field] < 0 for field in FIELDS): return None
        return {**reference, **{field: round(reference[field]*grams/original, 3) for field in FIELDS},
                'reference_grams': round(grams, 3), 'portion_method': 'volume_density',
                'basis': '已标定体积 × 密度估重，非称重', 'volume_ml': volume,
                'density_g_ml': density['value_g_ml'], 'density_source': density['source'],
                'density_status': density['status']}

    def enrich(self, meal):
        report = self.measurement_reader(Path(meal['bundle_ref'])) if meal.get('bundle_ref') else {'status': 'unavailable', 'instances': []}
        meal['_measurements'] = report
        by_id = {row['instance_id']: row for row in report['instances']}
        for item in meal['items']:
            volume = by_id.get(item.get('instance_id'))
            item['volume'] = volume
            density = self.density(item)
            item['density'] = density
            item['reference_nutrition'] = deepcopy(item.get('nutrition'))
            mass = {'status': 'volume_unavailable', 'weight_grams': None, 'message': '暂无已标定体积，暂不能估重'}
            if density['status'] == 'blocked':
                item['nutrition'] = item['reference_nutrition'] = None
                mass.update(status='blocked', message=density['message'])
            elif item.get('instance_id') and meal.get('bundle_ref'):
                # A 3D food's actual nutrient estimate must not silently fall
                # back to the old fixed reference portion when density is absent.
                item['nutrition'] = None
                ml = (volume or {}).get('volume_ml')
                if report.get('status') == 'calibrated' and (volume or {}).get('status') == 'estimated' and positive(ml):
                    if positive(density.get('value_g_ml')):
                        grams = ml * density['value_g_ml']
                        if positive(grams):
                            mass.update(status='estimated', weight_grams=round(grams, 3), volume_ml=ml,
                                        density_g_ml=density['value_g_ml'], message='体积 × 密度估重，非称重')
                            if item['reference_nutrition']:
                                item['nutrition'] = self.scaled_nutrition(item['reference_nutrition'], grams, density, ml)
                    else:
                        mass.update(status='density_pending', message=density['message'])
            item['mass_estimate'] = mass
        return meal
