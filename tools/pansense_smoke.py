"""Submit a real photo/video to a running, isolated PanSense validation server."""
import argparse
import json
from pathlib import Path
from food3d.paths import resolve_arguments
import time
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--kind', choices=['photo', 'video'], default='photo')
    parser.add_argument('--meal-id')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = resolve_arguments(parser.parse_args())
    with httpx.Client(base_url=args.url, timeout=120) as client:
        if args.meal_id:
            identifier = args.meal_id
        else:
            if not args.input:
                parser.error('--input or --meal-id is required')
            with args.input.open('rb') as file:
                response = client.post('/api/meals/upload', params={'kind':args.kind},
                    files={'file':(args.input.name,file,'video/mp4' if args.kind == 'video' else 'image/jpeg')})
            response.raise_for_status()
            identifier = response.json()['id']
        last = None
        while True:
            response = client.get(f'/api/meals/{identifier}')
            response.raise_for_status()
            meal = response.json()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(meal,ensure_ascii=False,indent=2), encoding='utf-8')
            analysis = meal['analysis']
            signature = (analysis['status'],analysis['stage'],analysis['progress'])
            if signature != last:
                print(json.dumps({'id':identifier,'analysis':analysis,
                    'items':[item['name'] for item in meal['items']],
                    'three_d_available':meal['three_d_available']},ensure_ascii=False), flush=True)
                last = signature
            if not args.wait or analysis['status'] in {'ready','failed'}:
                if analysis['status'] == 'failed': raise SystemExit(1)
                break
            time.sleep(3)


if __name__ == '__main__':
    main()
