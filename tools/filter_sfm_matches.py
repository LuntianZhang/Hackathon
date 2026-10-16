import argparse
from pathlib import Path
from food3d.paths import resolve_arguments
from food3d.data.match_filter import filter_database
p=argparse.ArgumentParser()
p.add_argument('--source',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--segmentation',required=True,type=Path)
p.add_argument('--focal',type=float,default=1152)
a=resolve_arguments(p.parse_args())
print(filter_database(a.source,a.output,a.segmentation,a.focal))
