"""Run all three SAM 2 seed/propagation adapters on a small real-view subset.

Does not train, fuse, or replace the original Gaussian model. Run only after
the web recognition queue is idle to avoid competing for GPU memory.
"""
import argparse
from dataclasses import replace
from pathlib import Path
from food3d.paths import ProjectPaths, resolve_arguments
import time

from food3d.io import read_json, write_json
from food3d.schemas import load_scene, save_scene
from food3d.segmentation.sam2 import segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manual', type=Path, required=True, help='第三种方式的 assistance JSON')
    parser.add_argument('--food-text', default='香蕉、青苹果、李子')
    args = resolve_arguments(parser.parse_args())
    if args.output.exists():
        raise ValueError('验证输出目录必须尚未存在')
    data, frames = load_scene(args.scene/'scene.json')
    reference = next(i for i,f in enumerate(frames) if f.frame_id == '000018')
    frames = frames[max(0,reference-1):reference+2]
    frames = [replace(frame,image_path=str((args.scene/frame.image_path).resolve())) for frame in frames]
    save_scene(args.output/'scene/scene.json',frames,source=data['source'],world_units=data['world_units'])
    records = []
    specs = [{'mode':'list','food_text':args.food_text}, {'mode':'grounded','food_text':args.food_text},
             read_json(args.manual)]
    for spec in specs:
        started=time.monotonic()
        result=segment(args.output/'scene',args.output/spec['mode'],ProjectPaths().resolve('.tools/weights/sam2.1_hiera_small.pt'),
                       reference_frame='000018',assistance=spec)
        records.append({'mode':spec['mode'],'elapsed_seconds':time.monotonic()-started,'result':result})
        write_json(args.output/'report.json',records)
    print('All three seed and propagation adapters completed; Gaussian geometry unchanged.',flush=True)


if __name__ == '__main__': main()
