"""Optional SfM recovery for repeated objects using reviewed instance tracks."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from ..io import read_json, write_json


def filter_database(source: Path, destination: Path, segmentation: Path, focal: float = 1152) -> dict:
    import pycolmap as p
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    segmentation_meta = read_json(segmentation/'segmentation.json')
    before = after = 0
    with p.Database.open(destination) as db:
        images = {im.image_id: im for im in db.read_all_images()}
        key_labels = {}
        for image_id, im in images.items():
            camera = db.read_camera(im.camera_id)
            mask = np.array(Image.open(segmentation/segmentation_meta['masks'][Path(im.name).stem]))
            if mask.shape != (camera.height, camera.width):
                raise ValueError('匹配过滤只支持掩码和原始特征像素完全一致的无畸变输入')
            xy = np.floor(db.read_keypoints(image_id)[:, :2]).astype(int)
            x, y = xy[:,0].clip(0,camera.width-1), xy[:,1].clip(0,camera.height-1)
            key_labels[image_id] = mask[y,x]
        pairs, matches = db.read_all_matches()
        db.clear_matches()
        db.clear_two_view_geometries()
        pair_lines = []
        for pair, match in zip(pairs,matches):
            a,b = p.pair_id_to_image_pair(pair)
            la,lb = key_labels[a][match[:,0]], key_labels[b][match[:,1]]
            keep = (la==lb) & (la!=65535)
            filtered = match[keep]
            before += len(match)
            after += len(filtered)
            if len(filtered)>=15:
                db.write_matches(a,b,filtered)
                pair_lines.append(images[a].name+' '+images[b].name)
        for camera in db.read_all_cameras():
            replacement = p.Camera.create_from_model_id(camera.camera_id,p.CameraModelId.SIMPLE_PINHOLE,focal,camera.width,camera.height)
            replacement.has_prior_focal_length = True
            db.update_camera(replacement)
    pairs_file = destination.with_suffix('.pairs.txt')
    pairs_file.write_text('\n'.join(pair_lines),encoding='utf-8')
    options = p.TwoViewGeometryOptions()
    options.ransac.random_seed=7
    p.verify_matches(destination,pairs_file,options)
    result = {'matches_before':before,'matches_after':after,'pairs':len(pair_lines),'camera_model':'SIMPLE_PINHOLE','initial_focal':focal,'segmentation':str(segmentation.resolve()),'semantic_tracks_used_in_sfm':True}
    write_json(destination.with_suffix('.filter.json'),result)
    return result
