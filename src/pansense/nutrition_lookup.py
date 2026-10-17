"""Background nutrition searches with persistent, exact-name, per-100g results."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
import re
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from food3d.food_catalog import normalize_name


PENDING = {'queued', 'searching'}
MESSAGES = {
    'queued': '营养数据查询中，正在等待联网检索',
    'searching': '营养数据查询中…',
    'found': '已查到在线营养参考，请核对食品与处理方式',
    'not_found': '未查到名称和营养字段均匹配的数据，可补充品牌、生熟或配方后再查',
    'ambiguous': '查到多个营养版本，请补充品牌、生熟或配方后再查',
    'error': '暂时无法联网查询，可稍后重试',
}


def exact_name(value):
    value = unicodedata.normalize('NFKD', normalize_name(value))
    return ''.join(c for c in value if c.isalnum() and not unicodedata.combining(c))


def valid_nutrients(values):
    """Missing fields and NaN are unknown, never zero-filled."""
    try:
        if set(values) != {'kcal', 'protein', 'carbs', 'fat'}: return False
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values()): return False
        return values['kcal'] <= 1000 and all(values[k] <= 100 for k in ('protein', 'carbs', 'fat')) and sum(values[k] for k in ('protein', 'carbs', 'fat')) <= 105
    except (TypeError, ValueError):
        return False


class PublicNutritionProvider:
    """USDA generic foods, then exact-name Open Food Facts label references."""
    def __init__(self):
        self.stop = threading.Event()
        self.last_off = 0

    def _json(self, endpoint, parameters):
        request = Request(endpoint + '?' + urlencode(parameters, doseq=True),
                          headers={'User-Agent': 'PanSense/0.1 (local food journal)', 'Accept': 'application/json'})
        for attempt in range(2):
            if 'openfoodfacts.org' in endpoint: self.last_off = time.monotonic()
            try:
                with urlopen(request, timeout=12) as response:
                    raw = response.read(2_000_001)
                break
            except HTTPError as error:
                if attempt or error.code not in {500, 502, 503, 504}: raise
                # One bounded retry for a temporary upstream outage. Respect
                # the product search limit on the retry as well as new tasks.
                if self.stop.wait(6.3 if 'openfoodfacts.org' in endpoint else 1): raise
        if len(raw) > 2_000_000: raise ValueError('查询返回数据过大')
        return json.loads(raw)

    def usda(self, query):
        data = self._json('https://api.nal.usda.gov/fdc/v1/foods/search', {
            'api_key': os.environ.get('USDA_FDC_API_KEY') or 'DEMO_KEY',
            'query': query, 'pageSize': 25, 'dataType': ['Foundation', 'SR Legacy']})
        results = []
        for food in data.get('foods', []):
            if exact_name(food.get('description', '')) != exact_name(query): continue
            nutrients = {}
            mapping = {1008: ('kcal', 'KCAL'), 1003: ('protein', 'G'), 1004: ('fat', 'G'), 1005: ('carbs', 'G')}
            for nutrient in food.get('foodNutrients', []):
                spec = mapping.get(nutrient.get('nutrientId'))
                if spec and str(nutrient.get('unitName', '')).upper() == spec[1]:
                    nutrients[spec[0]] = nutrient.get('value')
            identifier = food.get('fdcId')
            if not isinstance(identifier, int) or not valid_nutrients(nutrients): continue
            results.append({'per_100g': nutrients, 'source': 'USDA FoodData Central',
                'source_url': f'https://fdc.nal.usda.gov/food-details/{identifier}/nutrients',
                'source_id': str(identifier), 'reference_name': food['description'],
                'source_description': food['description'], 'quality': 'official_reference'})
        return results

    def open_food_facts(self, query):
        # Official search limit is 10/minute. Do not run search-as-you-type.
        delay = max(0, self.last_off + 6.2 - time.monotonic())
        if self.stop.wait(delay): return []
        self.last_off = time.monotonic()
        fields = 'code,product_name,product_name_zh,product_name_en,brands,nutriments,nutrition_data_per,no_nutrition_data,data_quality_errors_tags'
        data = self._json('https://world.openfoodfacts.org/cgi/search.pl', {
            'search_terms': query, 'search_simple': 1, 'action': 'process',
            'json': 1, 'page_size': 25, 'fields': fields})
        results = []
        for product in data.get('products', []):
            names = [product.get(k, '') for k in ('product_name', 'product_name_zh', 'product_name_en')]
            if not any(exact_name(name) == exact_name(query) for name in names): continue
            # A per-100ml drink cannot be silently converted into 100 grams.
            if product.get('nutrition_data_per') != '100g' or product.get('no_nutrition_data') in ('on', '1', 1, True): continue
            if product.get('data_quality_errors_tags'): continue
            code = str(product.get('code', ''))
            if not re.fullmatch(r'\d{8,14}', code): continue
            source = product.get('nutriments') or {}
            kcal = source.get('energy-kcal_100g')
            if kcal is None and isinstance(source.get('energy-kj_100g'), (int, float)):
                kcal = round(source['energy-kj_100g'] / 4.184, 3)
            nutrients = {'kcal': kcal, 'protein': source.get('proteins_100g'),
                         'carbs': source.get('carbohydrates_100g'), 'fat': source.get('fat_100g')}
            if not valid_nutrients(nutrients): continue
            name = next(name for name in names if exact_name(name) == exact_name(query))
            description = name + (' · ' + product['brands'] if product.get('brands') else '')
            results.append({'per_100g': nutrients, 'source': 'Open Food Facts（产品标签参考）',
                'source_url': f'https://world.openfoodfacts.org/product/{code}',
                'source_id': code, 'reference_name': description, 'source_description': description,
                'quality': 'product_label_reference', 'license': 'ODbL'})
        return results

    def __call__(self, name, english_query):
        failed = False
        sources = [(self.usda, english_query), (self.open_food_facts, name)]
        # USDA descriptions are English; avoid spending a limited key request on
        # an unknown Chinese name that has no local English recognition label.
        if re.search(r'[\u4e00-\u9fff]', english_query): sources = sources[1:]
        for function, query in sources:
            if self.stop.is_set(): break
            try:
                results = function(query)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, KeyError):
                failed = True
                continue
            if results:
                # Never select an arbitrary brand/recipe when numerical references differ.
                profiles = {tuple(round(r['per_100g'][k], 2) for k in ('kcal', 'protein', 'carbs', 'fat')) for r in results}
                if len(profiles) > 1: return {'status': 'ambiguous'}
                return {'status': 'found', 'result': results[0]}
        return {'status': 'error' if failed else 'not_found'}


class NutritionLookup:
    def __init__(self, store, *, enabled=True, provider=None):
        self.store = store
        self.enabled = enabled
        self.provider = provider or PublicNutritionProvider()
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nutrition-search')
        self.futures = {}
        self.epoch = 0
        self.closed = False
        with store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS nutrition_queries (id TEXT PRIMARY KEY, value TEXT NOT NULL)')
            # Pending work is resumed only when the corresponding food is used again.
            for row in db.execute('SELECT id,value FROM nutrition_queries').fetchall():
                value = json.loads(row['value'])
                if value['status'] in PENDING:
                    value.update(status='error', message='上次查询因服务重启中断，可重试', expires_at=0)
                    db.execute('UPDATE nutrition_queries SET value=? WHERE id=?', (json.dumps(value, ensure_ascii=False), row['id']))

    def identity(self, name, catalog_key=None):
        name = name.strip()
        if not 1 <= len(name) <= 80: raise ValueError('食品名称需要 1 至 80 个字符')
        if catalog_key is not None and catalog_key not in self.store.catalog.foods: raise ValueError('参考食物不存在')
        key = catalog_key or self.store.catalog.match(name)
        food = self.store.catalog.foods.get(key, {})
        reference_name = food.get('name', name)
        query = food.get('recognition_label', name)
        identifier = sha256(json.dumps([normalize_name(reference_name), key], ensure_ascii=False).encode()).hexdigest()[:32]
        return identifier, reference_name, query, key

    def get(self, identifier):
        with self.store.connection() as db:
            row = db.execute('SELECT value FROM nutrition_queries WHERE id=?', (identifier,)).fetchone()
        if row is None: raise KeyError('这次营养查询不存在')
        return json.loads(row['value'])

    def _write(self, value):
        with self.store.connection() as db:
            db.execute('INSERT OR REPLACE INTO nutrition_queries VALUES(?,?)',
                       (value['id'], json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def start(self, name, catalog_key=None, *, retry=False):
        identifier, name, query, key = self.identity(name, catalog_key)
        known = self.store.catalog.estimate(key)
        if known:
            return {'id': identifier, 'status': 'found', 'message': '已使用本机食物库', 'nutrition': known}
        with self.lock:
            try: cached = self.get(identifier)
            except KeyError: cached = None
            if cached and (cached['status'] in PENDING or cached.get('expires_at', 0) > time.time() and not retry): return cached
            if self.closed: return {'id': identifier, 'status': 'error', 'message': '服务正在停止，请稍后重试'}
            value = {'id': identifier, 'name': name, 'query': query, 'catalog_key': key,
                     'status': 'queued', 'message': MESSAGES['queued'], 'nutrition': None,
                     'requested_at': time.time(), 'expires_at': 0}
            if len(self.futures) >= 100:
                value.update(status='error', message='查询较多，请稍后重试', expires_at=time.time()+60)
                self._write(value)
                return value
            self._write(value)
            if self.enabled:
                epoch = self.epoch
                future = self.pool.submit(self._run, value, epoch)
                self.futures[identifier] = future
            return value

    def _run(self, value, epoch):
        with self.lock:
            if self.closed or epoch != self.epoch: return
            value = {**value, 'status': 'searching', 'message': MESSAGES['searching']}
            self._write(value)
        try:
            answer = self.provider(value['name'], value['query'])
            status = answer['status']
            if status not in {'found', 'not_found', 'ambiguous', 'error'}: raise ValueError('Unknown provider status')
            nutrition = None
            if status == 'found':
                result = answer['result']
                if not valid_nutrients(result['per_100g']): raise ValueError('Invalid nutrition')
                grams = self.store.catalog.foods.get(value['catalog_key'], {}).get('reference_grams', 100)
                nutrition = {**{k: round(v * grams / 100, 3) for k, v in result['per_100g'].items()},
                             **{k: v for k, v in result.items() if k != 'per_100g'},
                             'reference_grams': grams, 'basis': '在线参考数据，固定参考份量，非视觉称重',
                             'retrieved_at': time.time()}
            expiry = {'found': 30 * 86400, 'not_found': 86400, 'ambiguous': 86400, 'error': 60}[status]
            value.update(status=status, message=MESSAGES[status], nutrition=nutrition,
                         completed_at=time.time(), expires_at=time.time() + expiry)
        except Exception:
            # Never send remote payloads, credentials or internal tracebacks to the UI.
            value.update(status='error', message=MESSAGES['error'], nutrition=None, expires_at=time.time() + 60)
        with self.lock:
            if not self.closed and epoch == self.epoch: self._write(value)
            self.futures.pop(value['id'], None)

    @staticmethod
    def eligible(item):
        return item.get('nutrition') is None and item.get('object_kind') not in {'non_food', 'packaged_food'} and 1 <= len(item.get('name', '').strip()) <= 80

    def ensure_items(self, items):
        for item in items:
            if self.eligible(item): self.start(item['name'], item.get('catalog_key'))

    def enrich_items(self, items):
        result = deepcopy(items)
        for item in result:
            if not self.eligible(item): continue
            identifier, _, _, _ = self.identity(item['name'], item.get('catalog_key'))
            try: value = self.get(identifier)
            except KeyError: continue
            item['nutrition_lookup'] = {k: value[k] for k in ('id', 'status', 'message')}
            if value['status'] == 'found' and value.get('nutrition'):
                item['nutrition'] = value['nutrition']
                item['nutrition_needs_review'] = True
        return result

    def clear(self):
        with self.lock:
            self.epoch += 1
            for future in self.futures.values(): future.cancel()
            self.futures.clear()
            with self.store.connection() as db: db.execute('DELETE FROM nutrition_queries')

    def close(self):
        with self.lock:
            self.closed = True
            if hasattr(self.provider, 'stop'): self.provider.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
