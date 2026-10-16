"""Inspect the actual 3D Gaussian contributors behind a semantic click."""
import argparse
from pathlib import Path
from food3d.paths import resolve_arguments
import numpy as np
from food3d.schemas import load_scene
from food3d.viewer.renderer import GaussianRenderer,resize_camera
from food3d.fusion.visibility import splat_contributions
from food3d.io import write_json
p=argparse.ArgumentParser()
p.add_argument('--bundle',required=True,type=Path)
p.add_argument('--frame',required=True)
p.add_argument('--x',required=True,type=int)
p.add_argument('--y',required=True,type=int)
p.add_argument('--output',required=True,type=Path)
a=resolve_arguments(p.parse_args())
r=GaussianRenderer(a.bundle)
_,frames=load_scene(a.bundle/'scene.json')
c=resize_camera(next(f for f in frames if f.frame_id==a.frame),640)
rendered,alpha,meta=r.render_geometry(c)
g,pix,w=splat_contributions(r,c,meta)
sel=pix==a.y*c.width+a.x
g,w=g[sel].cpu().numpy(),w[sel].cpu().numpy()
order=np.argsort(w)[::-1][:20]
with np.load(a.bundle/'semantics/semantics.npz',allow_pickle=False) as s:
    top=[{'gaussian':int(g[i]),'contribution':float(w[i]),'instance':int(s['instance_ids'][g[i]]),'confidence':float(s['confidence'][g[i]]),'support_views':int(s['support_views'][g[i]]),'votes':s['votes'][g[i]].tolist()} for i in order]
result={'frame':a.frame,'pixel':[a.x,a.y],'alpha':float(alpha[0,a.y,a.x,0]),'expected_depth':float(rendered[0,a.y,a.x,3]),'top_contributors':top}
write_json(a.output,result)
print(result)
