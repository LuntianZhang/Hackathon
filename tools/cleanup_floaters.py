import argparse
from pathlib import Path
from food3d.paths import resolve_arguments
from food3d.reconstruction.cleanup import cleanup_background_floaters
p=argparse.ArgumentParser()
p.add_argument('--bundle',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=resolve_arguments(p.parse_args())
print(cleanup_background_floaters(a.bundle,a.output))
