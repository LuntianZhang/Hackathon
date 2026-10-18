"""Find usable local examples without depending on a separate demo server."""
from pathlib import Path

from food3d.paths import ProjectPaths


def local_examples(store, project):
    data = ProjectPaths(project).data_root
    roots = [data.resolve(), (store.root/'uploads').resolve()]

    def allowed_file(value):
        if not value:
            return None
        path = Path(value).resolve()
        return path if any(path.is_relative_to(root) for root in roots) and path.is_file() else None

    photo = model = None
    # Prefer completed, saved meals; comparison sessions must not become their
    # own example, and a stale bundle must not be advertised as a working model.
    meals = sorted(store.list(include_drafts=True), key=lambda m: bool(m.get('saved_at')), reverse=True)
    for meal in meals:
        if meal.get('comparison_parent') or meal['source_kind'] == 'assist-session':
            continue
        source = allowed_file(meal.get('source_photo'))
        if source is None:
            continue
        if photo is None:
            photo = {'path': source, 'meal': meal}
        bundle = meal.get('bundle_ref')
        if bundle and all(allowed_file(Path(bundle)/name) for name in ('scene.json', 'model.json', 'gaussians.npz')):
            model = {'path': source, 'meal': meal}
            break
    if model:
        photo = model
    if photo is None:
        source = allowed_file(data/'outputs/fruit2_view_counts_run01/n30/independent_colmap_scene/images/000018.jpg')
        if source:
            photo = {'path': source, 'meal': None}
    legacy = data/'outputs/fruit2_bundle'
    legacy_available = all(allowed_file(legacy/name) for name in ('scene.json', 'model.json', 'gaussians.npz', 'semantics/instances.json', 'semantics/fusion.json'))
    return {'photo': photo, 'model': model, 'legacy_demo_available': legacy_available}


def example_status(examples):
    photo, model = examples['photo'], examples['model']
    meal = model['meal'] if model else None
    return {'photo_available': photo is not None,
            'photo_title': photo['meal']['title'] if photo and photo['meal'] else '参考照片',
            'meal_id': meal['id'] if meal else None,
            'model_title': meal['title'] if meal else None,
            'model_url': f'/#plate/{meal["id"]}' if meal else None,
            'legacy_demo_available': examples['legacy_demo_available']}
