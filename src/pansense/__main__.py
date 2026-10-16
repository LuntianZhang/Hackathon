import argparse
from pathlib import Path

import uvicorn
from food3d.paths import ProjectPaths
from .api import create_app

parser = argparse.ArgumentParser(description='PanSense P0 local PWA server')
parser.add_argument('--port',type=int,default=8000)
parser.add_argument('--data-dir',type=Path)
parser.add_argument('--mobile',action='store_true',help='开启手机 HTTPS 拍摄入口')
parser.add_argument('--lan-host',help='手机可访问的电脑局域网 IP')
parser.add_argument('--https-port',type=int,default=8443)
parser.add_argument('--setup-port',type=int,default=8081)
args = parser.parse_args()
data_dir = ProjectPaths().resolve(args.data_dir) if args.data_dir else None
app=create_app(data_dir=data_dir)
if args.mobile:
    scans=app.state.scans
    scans.gateway.start(args.lan_host,args.https_port,args.setup_port)
    print(f'手机首次证书设置：{scans.setup_url}',flush=True)
    print(f'手机 HTTPS 相机：{scans.public_url}',flush=True)
    print(f'证书 SHA-256：{scans.certificate_fingerprint}',flush=True)
try:
    uvicorn.run(app,host='127.0.0.1',port=args.port)
finally:
    app.state.scans.gateway.close()
