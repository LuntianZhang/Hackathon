"""Configured locations for code, data, caches and Python environments."""
from __future__ import annotations

import json
import os
from pathlib import Path


class ProjectPaths:
    def __init__(self, project: Path | None = None):
        self.project = (project or Path(__file__).resolve().parents[2]).resolve()
        configuration = self.project / 'configs/paths.json'
        settings = json.loads(configuration.read_text(encoding='utf-8')) if configuration.is_file() else {}
        defaults = {'data_root': 'data', 'cache_root': '.cache', 'tools_root': '.tools',
                    'recognition_env': '.venv', 'reconstruction_env': '.venv-gs'}
        for name, default in defaults.items():
            value = Path(settings.get(name, default))
            resolved = (self.project / value).resolve()
            if not resolved.is_relative_to(self.project):
                raise ValueError(f'资源目录必须位于工程根目录内：{name}')
            setattr(self, name, resolved)

    @property
    def recognition_python(self) -> Path:
        return self.recognition_env / 'Scripts/python.exe'

    @property
    def reconstruction_python(self) -> Path:
        return self.reconstruction_env / 'Scripts/python.exe'

    def resolve(self, value: Path | str) -> Path:
        """Resolve resource-prefixed CLI paths to their configured locations."""
        path = Path(value)
        if not path.is_absolute() and path.parts:
            roots = {'data': self.data_root, '.cache': self.cache_root,
                     '.tools': self.tools_root, '.venv': self.recognition_env,
                     '.venv-gs': self.reconstruction_env}
            root = roots.get(path.parts[0])
            if root is not None:
                return root.joinpath(*path.parts[1:]).resolve()
        return path.resolve()

    def environment(self) -> dict[str, str]:
        source = str(self.project / 'src')
        entries = [source, *os.environ.get('PYTHONPATH', '').split(os.pathsep)]
        return dict(os.environ, PYTHONUTF8='0', PYTHONIOENCODING='utf-8', PYTHONNOUSERSITE='1',
                    PYTHONUNBUFFERED='1',
                    PYTHONPATH=os.pathsep.join(dict.fromkeys(entry for entry in entries if entry)),
                    HF_HOME=str(self.cache_root / 'models/huggingface'),
                    TORCH_HOME=str(self.cache_root / 'models/torch'))


def resolve_arguments(namespace):
    paths = ProjectPaths()
    for name, value in vars(namespace).items():
        if isinstance(value, Path):
            setattr(namespace, name, paths.resolve(value))
    return namespace
