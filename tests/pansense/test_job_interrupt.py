"""Cancellation stops work, preserves completed data, and isolates retries."""
import ctypes
import os
from pathlib import Path
import sys
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from pansense.api import create_app
from pansense.jobs import Jobs
from pansense.nutrition import Catalog
from pansense.store import Store


def wait_until(condition, timeout=8):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if condition(): return
        time.sleep(.02)
    raise AssertionError('Worker did not reach the expected state')


@pytest.mark.parametrize('kind,prepare', [('photo',False),('video',False),('video',True),('scan',True)])
def test_interrupt_api_preserves_assets_and_retries_same_input(tmp_path, monkeypatch, kind, prepare):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        store,jobs=client.app.state.store,client.app.state.jobs
        item=store.new_item('已确认的食物',instance_id=7)
        meal=store.create(kind,items=[item],video_preparing=prepare,
            analysis={'status':'running','stage':'计算中','progress':42,'error':None})
        root=tmp_path/'uploads'/meal['id'];root.mkdir(parents=True)
        photo=root/'source.jpg';Image.new('RGB',(40,30),'orange').save(photo)
        source=root/'photos' if kind=='scan' else photo if kind=='photo' else root/'source.mp4'
        if kind=='scan': source.mkdir()
        elif kind=='video': source.write_bytes(b'original video')
        store.fields(meal['id'],source_photo=str(photo),input_file=str(source))
        url='/api/meals/'+meal['id']
        stopped=client.post(url+'/interrupt')
        assert stopped.status_code==200,stopped.text
        value=stopped.json()
        assert value['analysis']['status']=='cancelled' and value['analysis']['progress']==42
        assert value['items'][0]['id']==item['id'] and source.exists()
        assert client.get(value['photo_url']).status_code==200
        assert client.post(url+'/interrupt').json()['revision']==value['revision']
        calls=[]
        monkeypatch.setattr(jobs,'submit',lambda *args:calls.append(args))
        assert client.post(url+'/retry').status_code==200
        assert calls==[(meal['id'],kind+('-prepare' if prepare else ''),source)]
        assert client.post('/api/meals/missing/interrupt').status_code==404


def test_interrupt_3d_keeps_ready_recognition_and_completed_meals_are_untouched(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        store=client.app.state.store
        meal=store.create('photo',items=[store.new_item('苹果',instance_id=3)],
            three_d_analysis={'status':'queued','stage':'等待三维关联','progress':0,'error':None})
        url='/api/meals/'+meal['id']+'/interrupt'
        value=client.post(url).json()
        assert value['analysis']['status']=='ready' and store.get(meal['id'])['items']==meal['items']
        assert value['three_d_analysis']['status']=='cancelled'
        assert client.post(url).json()['revision']==value['revision']
        client.app.state.jobs.submit_3d(meal['id'],{})
        assert store.get(meal['id'])['three_d_analysis']['status']=='queued'
        finished=store.create('photo')
        assert client.post('/api/meals/'+finished['id']+'/interrupt').json()['revision']==finished['revision']


def test_cancel_waiting_for_gpu_does_not_block_other_jobs(tmp_path, monkeypatch):
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    gpu=threading.Lock();gpu.acquire()
    jobs=Jobs(store,Path.cwd(),gpu)
    waiting=threading.Event()
    def run(identifier,*args):
        waiting.set()
        with jobs.use_gpu(identifier): pytest.fail('Cancelled job acquired the GPU')
    monkeypatch.setattr(jobs,'run',run)
    meal=store.create('photo')
    try:
        jobs.submit(meal['id'],'photo',tmp_path/'input.jpg')
        assert waiting.wait(3)
        jobs.interrupt(meal['id'])
        jobs.futures[meal['id']].result(timeout=2)
        assert gpu.locked() and store.get(meal['id'])['analysis']['status']=='cancelled'
    finally:
        gpu.release();jobs.close()


def test_queued_cancel_never_runs_and_immediate_retry_rejects_late_publication(tmp_path, monkeypatch):
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    jobs=Jobs(store,Path.cwd(),threading.Lock())
    started,release=threading.Event(),threading.Event()
    meal=store.create('photo');queued=store.create('photo')
    calls=[]
    def run(identifier,kind,*args):
        calls.append((identifier,kind))
        if kind=='old':
            started.set();assert release.wait(5)
            jobs.fields(identifier,bundle_ref='late-result-must-not-publish')
            pytest.fail('Old operation was allowed to publish')
        jobs.fields(identifier,analysis={'status':'ready','stage':'new result','progress':100,'error':None})
    monkeypatch.setattr(jobs,'run',run)
    try:
        jobs.submit(meal['id'],'old',tmp_path/'input.jpg');assert started.wait(3)
        old=jobs.futures[meal['id']]
        jobs.submit(queued['id'],'never',tmp_path/'input.jpg')
        jobs.interrupt(queued['id'])
        assert jobs.futures[queued['id']].cancelled()
        jobs.interrupt(meal['id'])
        jobs.submit(meal['id'],'new',tmp_path/'input.jpg')
        release.set();old.result(timeout=3);jobs.futures[meal['id']].result(timeout=3)
        value=store.get(meal['id'])
        assert not value['bundle_ref'] and value['analysis']['stage']=='new result'
        assert calls==[(meal['id'],'old'),(meal['id'],'new')]
    finally:
        release.set();jobs.close()


def windows_process_running(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        code=ctypes.c_ulong()
        assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
        return code.value==259
    finally:kernel.CloseHandle(handle)


def test_stop_after_recognition_prevents_automatic_3d_queue(tmp_path, monkeypatch):
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    jobs=Jobs(store,Path.cwd(),threading.Lock())
    meal=store.create('photo',items=[store.new_item('苹果',instance_id=1)])
    ready,release=threading.Event(),threading.Event()
    def run(identifier,*args):
        jobs.fields(identifier,analysis={'status':'ready','stage':'识别完成','progress':100,'error':None})
        ready.set();assert release.wait(5)
        jobs.submit_3d(identifier,{})
    monkeypatch.setattr(jobs,'run',run)
    try:
        jobs.submit(meal['id'],'photo',tmp_path/'input.jpg');assert ready.wait(3)
        jobs.interrupt(meal['id']);release.set();jobs.futures[meal['id']].result(timeout=3)
        assert store.get(meal['id'])['analysis']['status']=='ready'
        assert not store.get(meal['id']).get('three_d_analysis')
    finally:release.set();jobs.close()


def test_interrupt_terminates_real_worker_and_its_child(tmp_path,monkeypatch):
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    jobs=Jobs(store,Path.cwd(),threading.Lock())
    meal=store.create('video');pid_file=tmp_path/'child.pid'
    worker=tmp_path/'worker.py'
    worker.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\nchild.wait()\n",encoding='utf-8')
    def run(identifier,*args):
        jobs.stage(identifier,'运行子进程',45)
        jobs.command(identifier,[sys.executable,str(worker),str(pid_file)],tmp_path/'worker.log')
        jobs.fields(identifier,bundle_ref='must-not-publish')
    monkeypatch.setattr(jobs,'run',run)
    try:
        jobs.submit(meal['id'],'video',tmp_path/'input.mp4')
        wait_until(pid_file.is_file)
        process=jobs.processes[meal['id']];child_pid=int(pid_file.read_text())
        if os.name=='nt': assert windows_process_running(child_pid)
        jobs.interrupt(meal['id'])
        jobs.futures[meal['id']].result(timeout=5)
        assert process.poll() is not None
        if os.name=='nt': assert not windows_process_running(child_pid)
        value=store.get(meal['id'])
        assert value['analysis']['status']=='cancelled' and not value['bundle_ref']
    finally:jobs.close()
