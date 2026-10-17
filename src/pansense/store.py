"""SQLite meal documents; revisions prevent concurrent edits from being lost."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .nutrition import Catalog, total_nutrition

DEFAULT_PROFILE = {'onboarded': False, 'display_name': '记录生活', 'goal_kcal': None,
                   'timezone': 'Asia/Shanghai', 'retain_media': True}

class ConflictError(ValueError):
    pass

def now():
    return datetime.now(timezone.utc).isoformat()

def parse_timestamp(value: str):
    # Browser toISOString() uses Z; Python 3.10 needs an explicit UTC offset.
    try:
        result=datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    except ValueError as error:
        raise ValueError('请填写有效的记录时间') from error
    if result.tzinfo is None:
        raise ValueError('记录时间必须包含时区')
    return result

class Store:
    def __init__(self, root: Path, catalog: Catalog):
        self.root, self.catalog = root.resolve(), catalog
        self.nutrition_lookup = None
        self.quantities = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root/'pansense.sqlite3'
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS meals (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, value TEXT NOT NULL,
                    saved_at TEXT, occurred_at TEXT NOT NULL, created_at TEXT NOT NULL);
            ''')
            db.execute('INSERT OR IGNORE INTO settings VALUES(1, ?)', (json.dumps(DEFAULT_PROFILE),))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def profile(self):
        with self.connection() as db:
            return json.loads(db.execute('SELECT value FROM settings WHERE id=1').fetchone()[0])

    def update_profile(self, patch: dict):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = json.loads(db.execute('SELECT value FROM settings WHERE id=1').fetchone()[0])
            value.update(patch)
            try:
                ZoneInfo(value['timezone'])
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError('请选择有效的记录时区') from error
            db.execute('UPDATE settings SET value=? WHERE id=1', (json.dumps(value, ensure_ascii=False),))
            return value

    def new_item(self, name: str, catalog_key=None, instance_id=None, **metadata):
        if not name.strip(): raise ValueError('请输入食物名称')
        key = catalog_key if catalog_key is not None else self.catalog.match(name)
        if metadata.get('object_kind') in {'packaged_food', 'non_food'}:
            key = None
        return {'id': uuid4().hex, 'name': name.strip(), 'catalog_key': key,
                'instance_id': instance_id, 'nutrition': self.catalog.estimate(key), **metadata}

    def create(self, source_kind: str, **fields):
        identifier, timestamp = uuid4().hex, now()
        meal = {'id': identifier, 'revision': 1, 'title': '这一餐', 'meal_type': 'snack',
                'occurred_at': timestamp, 'note': '', 'created_at': timestamp, 'updated_at': timestamp,
                'saved_at': None, 'source_kind': source_kind, 'source_photo': None,
                'bundle_ref': None, 'items': [], 'analysis': {'status': 'ready', 'stage': '等待添加食物',
                'progress': 0, 'error': None}, **fields}
        with self.connection() as db:
            db.execute('INSERT INTO meals VALUES(?,?,?,?,?,?)',
                       (identifier, 1, json.dumps(meal, ensure_ascii=False), None, meal['occurred_at'], timestamp))
        if self.nutrition_lookup: self.nutrition_lookup.ensure_items(meal['items'])
        return self.get(identifier)

    def get(self, identifier: str):
        with self.connection() as db:
            row = db.execute('SELECT value FROM meals WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise KeyError('这条餐食不存在或已删除')
        value = json.loads(row[0])
        if self.nutrition_lookup: value['items'] = self.nutrition_lookup.enrich_items(value['items'])
        if self.quantities: value = self.quantities.enrich(value)
        value['totals'] = total_nutrition(value['items'])
        return value

    def exists(self, identifier: str):
        with self.connection() as db:
            return db.execute('SELECT 1 FROM meals WHERE id=?', (identifier,)).fetchone() is not None

    def mutate(self, identifier: str, operation, revision: int | None = None):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT value FROM meals WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise KeyError('这条餐食不存在或已删除')
            value = json.loads(row[0])
            if revision is not None and revision != value['revision']:
                raise ConflictError('餐食已更新，请刷新后重试')
            operation(value)
            value['revision'] += 1
            value['updated_at'] = now()
            value.pop('totals', None)
            db.execute('UPDATE meals SET revision=?,value=?,saved_at=?,occurred_at=? WHERE id=?',
                       (value['revision'], json.dumps(value, ensure_ascii=False, allow_nan=False),
                        value['saved_at'], value['occurred_at'], identifier))
        if self.nutrition_lookup: self.nutrition_lookup.ensure_items(value['items'])
        return self.get(identifier)

    def fields(self, identifier: str, **fields):
        return self.mutate(identifier, lambda value: value.update(fields))

    def edit_item(self, identifier: str, item_id: str, name: str, catalog_key, revision: int, *, density=None):
        if not name.strip(): raise ValueError('请输入食物名称')
        def edit(value):
            item = next((item for item in value['items'] if item['id'] == item_id), None)
            if item is None:
                raise KeyError('这项食物已不存在')
            key = catalog_key if catalog_key is not None else self.catalog.match(name)
            previous_identity = [item['name'].strip(), item.get('catalog_key')]
            item.update(name=name.strip(), catalog_key=key, nutrition=self.catalog.estimate(key),
                        name_source='manual_correction', needs_review=False)
            if previous_identity != [item['name'], key]: item.pop('density_selection', None)
            if density is not None:
                if not self.quantities: raise ValueError('密度服务暂不可用')
                item['density_selection'] = self.quantities.selection(item, density)
        return self.mutate(identifier, edit, revision)

    def add_item(self, identifier: str, name: str, catalog_key, revision: int, region=None, *, density=None):
        def add(value):
            if len(value['items']) >= 40:
                raise ValueError('一餐最多记录 40 项食物')
            item = self.new_item(name, catalog_key, source='manual', region=region, needs_review=False)
            if density is not None:
                if not self.quantities: raise ValueError('密度服务暂不可用')
                item['density_selection'] = self.quantities.selection(item, density)
            value['items'].append(item)
        return self.mutate(identifier, add, revision)

    def delete_item(self, identifier: str, item_id: str, revision: int):
        def delete(value):
            found = [item for item in value['items'] if item['id'] == item_id]
            if not found:
                raise KeyError('这项食物已不存在')
            value['items'] = [item for item in value['items'] if item['id'] != item_id]
        return self.mutate(identifier, delete, revision)

    def save(self, identifier: str, metadata: dict, revision: int):
        if 'title' in metadata and not metadata['title'].strip(): raise ValueError('请输入餐食标题')
        def save(value):
            if not value['items']:
                raise ValueError('请至少添加一项食物再保存')
            occurred = parse_timestamp(metadata.get('occurred_at', value['occurred_at']))
            value.update(metadata)
            value['occurred_at'] = occurred.astimezone(timezone.utc).isoformat()
            value['saved_at'] = value['saved_at'] or now()
        return self.mutate(identifier, save, revision)

    def list(self, date: str | None = None, include_drafts=False):
        profile = self.profile()
        zone = ZoneInfo(profile['timezone'])
        with self.connection() as db:
            rows = db.execute('SELECT id FROM meals ' + ('' if include_drafts else 'WHERE saved_at IS NOT NULL ')
                              + 'ORDER BY occurred_at DESC,created_at DESC').fetchall()
        values = [self.get(row[0]) for row in rows]
        if date is not None:
            datetime.strptime(date, '%Y-%m-%d')
            values = [value for value in values if parse_timestamp(value['occurred_at']).astimezone(zone).date().isoformat() == date]
        return values

    def summary(self, date: str):
        values = self.list(date)
        return {'date': date, 'count': len(values),
                'totals': total_nutrition([item for meal in values for item in meal['items']]),
                'goal_kcal': self.profile()['goal_kcal']}

    def delete(self, identifier: str):
        meal = self.get(identifier)
        with self.connection() as db:
            db.execute('DELETE FROM meals WHERE id=?', (identifier,))
        return meal

    def purge(self):
        if self.nutrition_lookup: self.nutrition_lookup.clear()
        with self.connection() as db:
            db.execute('DELETE FROM meals')
            db.execute('UPDATE settings SET value=? WHERE id=1', (json.dumps(DEFAULT_PROFILE),))
        # Erase deleted JSON from SQLite pages after checkpointing the WAL.
        with self.connection() as db:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            db.execute('VACUUM')
