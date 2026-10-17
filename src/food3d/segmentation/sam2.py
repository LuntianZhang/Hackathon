from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import initialize_scene, read_json, write_json
from ..paths import ProjectPaths
from ..schemas import UNKNOWN_ID, load_scene
from ..visuals import contact_sheet
from .recognition import ClipRecognizer, NAMES_ZH


def select_proposals(proposals: list[dict], classifications: list[dict], image_area: int, max_instances: int = 20, max_fraction: float = 0.20) -> list[tuple[dict, dict]]:
    foods = []
    excluded = {"table", "plate", "background", "unknown"}
    for proposal, result in zip(proposals, classifications):
        fraction = proposal["area"] / image_area
        if not 0.001 <= fraction <= max_fraction or result["label"] in excluded or result["similarity"] < 0.16:
            continue
        quality = float(proposal["predicted_iou"]) * float(proposal["stability_score"])
        foods.append((proposal, result, quality))
    # Prefer large coherent objects over contained parts (spikes, stems, leaves).
    foods.sort(key=lambda entry: entry[0]["area"], reverse=True)
    chosen = []
    for proposal, result, quality in foods:
        mask = proposal["segmentation"]
        redundant = False
        for existing, _, _ in chosen:
            intersection = np.logical_and(mask, existing["segmentation"]).sum()
            union = np.logical_or(mask, existing["segmentation"]).sum()
            if intersection / max(union, 1) > 0.45 or intersection / max(min(proposal["area"], existing["area"]), 1) > 0.85:
                redundant = True
                break
        if not redundant:
            chosen.append((proposal, result, quality))
        if len(chosen) >= max_instances:
            break
    chosen.sort(key=lambda entry: (entry[0]["bbox"][1], entry[0]["bbox"][0]))
    return [(proposal, result) for proposal, result, _ in chosen]


def segment(scene: Path, output: Path, checkpoint: Path, reference_frame: str | None = None, prompts: Path | None = None, candidates: list[str] | None = None, assistance: dict | None = None, seed_masks: Path | None = None) -> dict:
    if assistance and assistance.get('mode') == 'llm':
        from .hints import validate_assistance
        validate_assistance(assistance)
        if prompts or seed_masks:
            raise ValueError('大模型多视图识别不能同时使用单帧人工种子')
        from .llm_segment import segment_llm
        return segment_llm(scene, output, checkpoint, reference_frame)
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from .hints import validate_assistance, candidate_names, display_names, allowed_food
    assistance = validate_assistance(assistance)
    if prompts and assistance['mode'] != 'auto':
        raise ValueError('旧版 --prompts 和 --assistance 不能同时使用')
    if seed_masks and prompts:
        raise ValueError('不能同时使用种子掩码与点框提示')
    manual_names = bool(prompts or seed_masks) or assistance['mode'] == 'manual'
    catalog_path=ProjectPaths().project/'app/catalog/foods.json'
    catalog=read_json(catalog_path)['foods'] if catalog_path.is_file() else []
    from ..food_catalog import FoodIndex
    food_index=FoodIndex(catalog)
    names = display_names(assistance,catalog)
    scene, output, checkpoint = scene.resolve(), output.resolve(), checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM 2 权重不存在：{checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 SAM 2 传播适配器需要 CUDA；几何模拟测试仍可在 CPU 运行")
    _, frames = load_scene(scene / "scene.json")
    dimensions = {(frame.width, frame.height) for frame in frames}
    if len(dimensions) != 1:
        raise ValueError("视频传播需要一致的图像尺寸；请统一处理相机图像")
    ids = [frame.frame_id for frame in frames]
    seed_index = ids.index(reference_frame) if reference_frame else 0
    cache = ProjectPaths().cache_root / 'models'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    config = "configs/sam2.1/sam2.1_hiera_s.yaml"
    with Image.open(scene / frames[seed_index].image_path) as source:
        image = source.convert("RGB")
    recognizer = ClipRecognizer(candidate_names(assistance, candidates or food_index.labels() or ["apple", "banana", "durian", "bread", "orange", "pear", "mango", "grapes"]), cache)
    if seed_masks:
        seeds, chosen = load_seed_masks(seed_masks, image.size)
    elif prompts:
        seeds = read_json(prompts)["instances"]
        chosen = []
    else:
        if assistance['mode'] in {'grounded', 'manual'}:
            from .prompted import prompted_masks
            proposals = prompted_masks(image, assistance, checkpoint, cache)
        else:
            sam = build_sam2(config, str(checkpoint), device="cuda", apply_postprocessing=False)
            generator = SAM2AutomaticMaskGenerator(sam, points_per_side=24, points_per_batch=16, pred_iou_thresh=0.85, stability_score_thresh=0.92, min_mask_region_area=0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                proposals = generator.generate(np.asarray(image))
            del generator, sam
            torch.cuda.empty_cache()
        maximum_fraction = .75 if assistance['mode'] == 'manual' else .20
        proposals = [proposal for proposal in proposals if 0.001 <= proposal["area"] / (image.width*image.height) <= maximum_fraction]
        if assistance['mode'] == 'manual':
            chosen = [(p, {'label':p['seed']['name']}) for p in proposals]
        else:
            classifications = []
            for start in range(0, len(proposals), 16):
                classifications.extend(recognizer.classify(image, [proposal["segmentation"] for proposal in proposals[start:start+16]]))
            if assistance['mode'] != 'auto':
                from pansense.photo import reliable_proposals
                reliable = reliable_proposals(proposals, classifications, image.width, image.height)
                reliable = [(p,c) for p,c in reliable if allowed_food(assistance, c['label'])]
                proposals, classifications = [p for p,_ in reliable], [c for _,c in reliable]
            chosen = select_proposals(proposals, classifications, image.width*image.height)
        if not chosen:
            raise RuntimeError("SAM 2 / CLIP 未得到可靠的食物候选；可提供人工 points/box 初始提示")
        seeds = [{"instance_id": proposal['seed']['instance_id'] if manual_names else index+1,
                  "name": result['label'] if manual_names else names.get(result["label"], result["label"]),
                  "name_source": 'manual_prompt' if manual_names else 'automatic',
                  "auto_label": result["label"], "recognition": result,
                  "seed_quality": float(proposal["predicted_iou"])*float(proposal["stability_score"])} for index, (proposal, result) in enumerate(chosen)]
    instance_ids = [int(seed["instance_id"]) for seed in seeds]
    if len(set(instance_ids)) != len(instance_ids) or not instance_ids or any(value <= 0 or value >= UNKNOWN_ID for value in instance_ids):
        raise ValueError("初始实例 ID 必须唯一且介于 1 和 65534")
    initialize_scene(output)
    for name in ["frames", "masks", "confidence", "debug"]:
        (output / name).mkdir()
    for index, frame in enumerate(frames):
        with Image.open(scene / frame.image_path) as source:
            source.convert("RGB").save(output / "frames" / f"{index:06d}.jpg", quality=95)
    write_json(output / "instances.json", {"schema_version": 1, "instances": [{"instance_id": 0, "name": "背景"}, *seeds], "unknown_id": UNKNOWN_ID, "initialization": "selected_result_masks" if seed_masks else "manual_prompts" if prompts else assistance['mode'], "assistance": assistance, "reference_frame": frames[seed_index].frame_id, "names_manually_correctable": True})
    predictor = build_sam2_video_predictor(config, str(checkpoint), device="cuda", apply_postprocessing=False)
    previews = []
    summaries = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(output / "frames"), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for index, seed in enumerate(seeds):
            if chosen:
                predictor.add_new_mask(state, frame_idx=seed_index, obj_id=int(seed["instance_id"]), mask=chosen[index][0]["segmentation"])
            else:
                kwargs = {"box": np.asarray(seed["box"], dtype=np.float32)} if "box" in seed else {"points": np.asarray(seed["points"], dtype=np.float32), "labels": np.asarray(seed.get("labels", [1]*len(seed["points"])), dtype=np.int32)}
                predictor.add_new_points_or_box(state, frame_idx=seed_index, obj_id=int(seed["instance_id"]), **kwargs)
        for reverse in [False, True]:
            for index, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed_index, reverse=reverse):
                score = logits[:, 0].float().cpu().numpy()
                top = np.argmax(score, axis=0)
                best = np.take_along_axis(score, top[None, :, :], axis=0)[0]
                label = np.asarray(obj_ids, dtype=np.uint16)[top]
                label[best <= 0] = 0
                probabilities = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
                best_probability = np.max(probabilities, axis=0)
                second = np.partition(probabilities, -2, axis=0)[-2] if len(obj_ids) > 1 else np.zeros_like(best_probability)
                ambiguous = (label > 0) & ((best_probability - second) < 0.1)
                label[ambiguous] = UNKNOWN_ID
                confidence = np.where(label == 0, 1-best_probability, best_probability)
                confidence[ambiguous] = 0
                frame = frames[index]
                Image.fromarray(label).save(output / "masks" / f"{frame.frame_id}.png")
                np.save(output / "confidence" / f"{frame.frame_id}.npy", confidence.astype(np.float16))
                summaries[frame.frame_id] = {str(obj_id): int((label == obj_id).sum()) for obj_id in instance_ids}
                print(f"SAM 2 frame {index+1}/{len(frames)} reverse={reverse}", flush=True)
    # Aggregate CLIP names over several propagated views; retain manual names.
    observations = {obj_id: [] for obj_id in instance_ids}
    for index in np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int):
        frame = frames[index]
        with Image.open(scene / frame.image_path) as source:
            picture = source.convert("RGB")
        with Image.open(output / "masks" / f"{frame.frame_id}.png") as source:
            labels = np.asarray(source)
        valid_ids = [obj_id for obj_id in instance_ids if (labels == obj_id).sum() >= 100]
        for obj_id, result in zip(valid_ids, recognizer.classify(picture, [labels == obj_id for obj_id in valid_ids])):
            observations[obj_id].append(result["scores"])
        colored = np.array(picture)
        for obj_id in instance_ids:
            color = np.array([(obj_id*71)%255, (obj_id*139)%255, (obj_id*193)%255])
            mask = labels == obj_id
            colored[mask] = (0.6*colored[mask] + 0.4*color).astype(np.uint8)
        preview = Image.fromarray(colored)
        preview.save(output / "debug" / f"{frame.frame_id}.png")
        preview.thumbnail((248, 188))
        previews.append((frame.frame_id, preview))
    instance_data = read_json(output / "instances.json")
    for entry in instance_data["instances"]:
        scores = observations.get(entry["instance_id"], [])
        if scores:
            average = {label: float(np.mean([score[label] for score in scores])) for label in recognizer.candidates}
            label = max(average, key=average.get)
            entry["multi_view_scores"] = average
            entry["recognition_views"] = len(scores)
            if not manual_names and allowed_food(assistance, label):
                entry["auto_label"] = label
                entry["name"] = names.get(label, label)
            elif not manual_names:
                entry['needs_review'] = True
                entry['name_evidence'] = 'multi_view_rejected_or_outside_hint_list'
    write_json(output / "instances.json", instance_data)
    contact_sheet(previews, output / "debug" / "mask_sheet.png")
    result = {"schema_version": 1, "source_scene": str(scene), "frames": len(frames), "instance_ids": instance_ids, "masks": {frame.frame_id: f"masks/{frame.frame_id}.png" for frame in frames}, "confidence": {frame.frame_id: f"confidence/{frame.frame_id}.npy" for frame in frames}, "pixel_counts": summaries, "initialization": instance_data["initialization"], "status": "propagated_requires_identity_review"}
    write_json(output / "segmentation.json", result)
    return {key: result[key] for key in ["source_scene", "frames", "instance_ids", "initialization", "status"]}


def load_seed_masks(path: Path, size: tuple[int, int]):
    """Propagate the exact selected 2D result, rather than detecting again."""
    seeds, chosen = [], []
    entries = read_json(path)['instances']
    if not entries or len(entries) > 20:
        raise ValueError('请选择至少一个食物，最多 20 个实例')
    for entry in entries:
        with Image.open(entry['mask_path']) as source:
            mask = np.asarray(source.convert('L').resize(size, Image.Resampling.NEAREST)) > 0
        if mask.sum() < 16:
            raise ValueError('选定的食物掩码为空或过小')
        seeds.append({'instance_id': entry['instance_id'], 'name': entry['name'],
                      'name_source': 'selected_result', 'seed_quality': entry.get('sam_score', 1)})
        chosen.append(({'segmentation': mask}, {}))
    return seeds, chosen
