"""Separate LAN camera gateway; meal diary remains bound to loopback."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .scan_geometry import board_svg, square_board_png, square_board_svg
from .mobile_https import lan_addresses
from .store import ConflictError


class ScanCreate(BaseModel):
    model_config=ConfigDict(extra='forbid')
    title:str=Field('手机扫描餐盘',min_length=1,max_length=80)


class ScanState(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1)
    status:str=Field(pattern='^(capturing|paused|complete)$')


class ScanBuild(BaseModel):
    model_config=ConfigDict(extra='forbid')
    assistance:dict
    paper_present:bool=False
    prepare:bool=False
    board_background:bool=False


class ScanConnection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    host:str|None=None
    https_port:int=Field(8443,ge=1,le=65535)
    setup_port:int=Field(8081,ge=1,le=65535)


def register_scan_routes(app,scans,public_meal=None,*,gateway=False):
    router=APIRouter()

    def authorization(identifier:str,authorization:str=Header('')):
        if gateway:
            if not authorization.startswith('Bearer ') or not scans.authenticate(identifier,authorization[7:]):
                raise HTTPException(401,'手机配对已失效，请在电脑上重新获取二维码')

    @router.get('/api/scans/board.svg')
    def board(): return Response(board_svg(),media_type='image/svg+xml',headers={'Content-Disposition':'inline; filename="PanSense-A3-board.svg"'})

    @router.get('/api/scans/board-20cm.svg')
    def board_square_svg(): return Response(square_board_svg(),media_type='image/svg+xml',headers={'Content-Disposition':'inline; filename="PanSense-20cm-board.svg"'})

    @router.get('/api/scans/board-20cm.png')
    def board_square_png(): return Response(square_board_png(),media_type='image/png',headers={'Content-Disposition':'inline; filename="PanSense-20cm-board.png"'})

    @router.get('/api/scans/{identifier}',dependencies=[Depends(authorization)])
    def scan(identifier:str): return scans.public(scans.get(identifier))

    @router.patch('/api/scans/{identifier}',dependencies=[Depends(authorization)])
    def state(identifier:str,body:ScanState): return scans.update(identifier,body.status,body.revision)

    @router.post('/api/scans/{identifier}/frames',dependencies=[Depends(authorization)])
    async def upload(identifier:str,file:UploadFile=File(...),frame_id:str=Form(...),timestamp:float=Form(...),manual_sector:int|None=Form(None)):
        parts=[];size=0
        try:
            while chunk:=await file.read(256*1024):
                size+=len(chunk)
                if size>4*1024*1024: raise HTTPException(413,'单张采集图像不能超过 4 MB')
                parts.append(chunk)
            if not size: raise ValueError('照片为空')
            return await run_in_threadpool(scans.add_frame,identifier,frame_id,b''.join(parts),timestamp,manual_sector)
        finally: await file.close()

    @router.get('/api/scans/{identifier}/frames/{frame_id}',dependencies=[Depends(authorization)])
    def photo(identifier:str,frame_id:str): return FileResponse(scans.frame_path(identifier,frame_id),media_type='image/jpeg')

    @router.delete('/api/scans/{identifier}/frames/{frame_id}',dependencies=[Depends(authorization)])
    def remove(identifier:str,frame_id:str,revision:int): return scans.remove_frame(identifier,frame_id,revision)

    if not gateway:
        @router.post('/api/scans/connection')
        def connect(body:ScanConnection):
            scans.gateway.start(body.host,body.https_port,body.setup_port)
            return listing()

        @router.delete('/api/scans/connection')
        def disconnect():
            scans.gateway.close();return listing()

        @router.get('/api/scans')
        def listing(): return {'scans':scans.list(),'connection':{
            'enabled':bool(scans.public_url),'url':scans.public_url,
            'setup_url':getattr(scans,'setup_url',None),'setup_qr':getattr(scans,'setup_qr',None),
            'certificate_fingerprint':scans.certificate_fingerprint,
            'addresses':lan_addresses()}}

        @router.post('/api/scans',status_code=201)
        def create(body:ScanCreate): return scans.create(body.title)

        @router.post('/api/scans/{identifier}/pair')
        def pair(identifier:str): return scans.pair(identifier)

        @router.post('/api/scans/{identifier}/build')
        def build(identifier:str,body:ScanBuild):
            from food3d.segmentation.hints import validate_assistance
            options=validate_assistance(body.assistance,list(scans.store.catalog.foods.values()),allow_none=True)
            if options['mode']=='manual': raise ValueError('请先准备扫描参考帧，再在参考图中点击或框选食物')
            return public_meal(scans.make_meal(identifier,options,body.paper_present,body.prepare,body.board_background))
    app.include_router(router)


def capture_app(scans,hosts):
    app=FastAPI(title='PanSense camera',docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=hosts)

    @app.middleware('http')
    async def headers(request,call_next):
        origin=request.headers.get('origin')
        if request.method not in {'GET','HEAD','OPTIONS'} and origin and urlsplit(origin).netloc!=request.headers.get('host'):
            return JSONResponse({'detail':'不接受其他网站的采集请求'},status_code=403)
        response=await call_next(request)
        response.headers.update({'Cache-Control':'no-store','Referrer-Policy':'no-referrer',
            'X-Content-Type-Options':'nosniff','Permissions-Policy':'camera=(self), microphone=()',
            'Content-Security-Policy':"default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    @app.exception_handler(KeyError)
    async def missing(request,error): return JSONResponse({'detail':str(error.args[0])},status_code=404)
    @app.exception_handler(ConflictError)
    async def conflict(request,error): return JSONResponse({'detail':str(error)},status_code=409)
    @app.exception_handler(ValueError)
    async def invalid(request,error): return JSONResponse({'detail':str(error)},status_code=400)

    register_scan_routes(app,scans,gateway=True)
    def static_asset(filename):
        def asset(): return FileResponse(scans.project/'app/web'/filename)
        return asset
    for filename in ('capture.html','capture.js','capture.css','scan-map.js','frame-diff.js','icon.svg'):
        app.add_api_route('/'+filename,static_asset(filename),methods=['GET'])

    @app.get('/')
    def index(): return FileResponse(scans.project/'app/web/capture.html')
    @app.get('/health')
    def health(): return {'status':'ok','service':'PanSense camera'}
    return app


def certificate_app(scans):
    """HTTP is used only to install the public certificate, never for photos."""
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    @app.get('/ca.cer')
    def certificate(): return FileResponse(scans.ca_path,media_type='application/x-x509-ca-cert',filename='PanSense-local-camera.cer')
    @app.get('/',response_class=HTMLResponse)
    def instructions():
        fingerprint=scans.certificate_fingerprint
        return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连接盘感摄像头</title>
        <style>body{{font:16px/1.8 system-ui;max-width:650px;margin:32px auto;padding:20px;color:#28483c;background:#f7f8f3}}a{{color:#276b55}}code{{overflow-wrap:anywhere}}</style>
        <h1>首次连接这台电脑</h1><p>手机与电脑连接同一 Wi-Fi。只需首次安装这台电脑的局域网相机证书；手机无需安装盘感 App。</p>
        <ol><li><a href="/ca.cer">下载此电脑的相机 CA 证书</a>。</li><li>安卓：设置 → 安全与隐私 → 更多安全设置 → 加密与凭据 → 安装证书 → CA 证书，选择刚下载的文件。不同品牌的菜单名称可能不同。</li>
        <li>返回电脑，再扫描“拍摄二维码”，授权浏览器使用后置摄像头。</li></ol>
        <p>此 CA 用于信任本机签发的 HTTPS 证书。请先与电脑上显示的指纹核对；用完后可在手机“受信任的凭据 → 用户”中移除 PanSense Local Camera。</p>
        <p>SHA-256 指纹：<br><code>{fingerprint}</code></p><p>此页面只提供公共证书，不接收照片。</p></html>'''
    return app
