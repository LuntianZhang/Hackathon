"""Remove opaque, poorly supported background floaters; preserve stable IDs."""
import os
from pathlib import Path
from dataclasses import replace
import numpy as np
from ..io import initialize_scene,read_json,write_json
from ..schemas import UNKNOWN_ID,load_scene,save_scene
from .gaussians import GaussianSet,load_bundle,file_hash
from .export import export_gaussian_ply


def background_floater_mask(opacities,labels,confidence,support,votes,instance_values):
    bg=np.flatnonzero(np.asarray(instance_values)==0)
    if len(bg)!=1:
        raise ValueError('Exactly one background channel is required')
    total=votes.sum(axis=1)
    background_fraction=votes[:,bg[0]]/np.maximum(total,1e-12)
    # Only background-only evidence may remove a splat. Unknown food,
    # mixed boundaries, low-opacity splats, and supported surfaces stay intact.
    return (labels==UNKNOWN_ID)&(confidence>=0.98)&(support>0)&(support<3)&(background_fraction>=0.98)&(total>=1.5)&(opacities>=0.5)


def cleanup_background_floaters(bundle:Path,output:Path)->dict:
    bundle,output=bundle.resolve(),output.resolve()
    model,metadata=load_bundle(bundle)
    fusion=read_json(bundle/'semantics/fusion.json')
    if fusion['model_sha256']!=metadata['model_sha256']:
        raise ValueError('Cleanup evidence belongs to a different model')
    with np.load(bundle/'semantics/semantics.npz',allow_pickle=False) as semantics:
        if not np.array_equal(semantics['gaussian_ids'],model.gaussian_ids):
            raise ValueError('Semantic Gaussian identities do not match the model')
        remove=background_floater_mask(model.opacities,semantics['instance_ids'],semantics['confidence'],semantics['support_views'],semantics['votes'],semantics['instance_values'])
    if remove.all():
        raise ValueError('Refusing cleanup that would remove every Gaussian')
    initialize_scene(output)
    filtered=GaussianSet(**{key:value[~remove] for key,value in model.__dict__.items()})
    filtered.save(output/'gaussians.npz')
    export_gaussian_ply(output/'gaussians.ply',filtered)
    np.save(output/'cleanup_removed_gaussian_ids.npy',model.gaussian_ids[remove])
    scene,frames=load_scene(bundle/'scene.json')
    frames=[replace(f,image_path=Path(os.path.relpath(bundle/f.image_path,output)).as_posix()) for f in frames]
    save_scene(output/'scene.json',frames,source=scene['source'],world_units=scene['world_units'],metadata=scene.get('metadata',{}))
    result={'source_bundle':str(bundle),'source_model_sha256':metadata['model_sha256'],'removed_gaussians':int(remove.sum()),'retained_gaussians':len(filtered.means),'rule':'unknown_label_background_fraction_ge_0.98_support_1_or_2_vote_mass_ge_1.5_opacity_ge_0.5','voting_views':fusion['voting_views'],'holdout_views':fusion['holdout_views'],'stable_gaussian_ids_preserved':True,'semantics_requires_refusion':True}
    metadata={**metadata,'model_sha256':file_hash(output/'gaussians.npz'),'gaussians':len(filtered.means),'semantics_verified':False,'postprocess':result}
    write_json(output/'model.json',metadata)
    write_json(output/'cleanup.json',result)
    return result
