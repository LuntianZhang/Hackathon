import argparse
from pathlib import Path

import uvicorn
from food3d.paths import ProjectPaths
from .api import create_app

parser = argparse.ArgumentParser(description='PanSense P0 local PWA server')
parser.add_argument('--port',type=int,default=8000)
parser.add_argument('--data-dir',type=Path)
args = parser.parse_args()
data_dir = ProjectPaths().resolve(args.data_dir) if args.data_dir else None
uvicorn.run(create_app(data_dir=data_dir),host='127.0.0.1',port=args.port)
