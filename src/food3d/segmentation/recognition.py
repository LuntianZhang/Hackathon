from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ClipRecognizer:
    def __init__(self, candidates: list[str], cache: Path, device: str = "cuda"):
        import open_clip
        import torch
        self.torch = torch
        self.device = device
        self.candidates = candidates
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device, cache_dir=str(cache))
        self.model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        prompts = [f"a photo of {name}" for name in candidates] + [f"a 3D render of {name}" for name in candidates]
        with torch.inference_mode():
            features = self.model.encode_text(tokenizer(prompts).to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            features = (features[:len(candidates)] + features[len(candidates):]) / 2
            self.text = features / features.norm(dim=-1, keepdim=True)

    def classify(self, image: Image.Image, masks: list[np.ndarray]) -> list[dict]:
        tensors = []
        for mask in masks:
            y, x = np.nonzero(mask)
            if not len(x):
                raise ValueError("不能识别空掩码")
            padding = max(4, int(max(x.max()-x.min(), y.max()-y.min()) * 0.08))
            box = (max(0, int(x.min())-padding), max(0, int(y.min())-padding), min(image.width, int(x.max())+padding+1), min(image.height, int(y.max())+padding+1))
            masked = Image.composite(image, Image.new("RGB", image.size, "white"), Image.fromarray(mask.astype(np.uint8)*255))
            tensors.append(self.preprocess(masked.crop(box)))
        if not tensors:
            return []
        with self.torch.inference_mode():
            features = self.model.encode_image(self.torch.stack(tensors).to(self.device))
            features /= features.norm(dim=-1, keepdim=True)
            scores = (features @ self.text.T).float().cpu().numpy()
        results = []
        for row in scores:
            order = np.argsort(row)[::-1]
            results.append({"label": self.candidates[int(order[0])], "similarity": float(row[order[0]]), "margin": float(row[order[0]] - row[order[1]]), "scores": {label: float(score) for label, score in zip(self.candidates, row)}})
        return results


NAMES_ZH = {"apple": "苹果", "banana": "香蕉", "durian": "榴莲", "bread": "面包", "orange": "橙子", "pear": "梨", "mango": "芒果", "grapes": "葡萄", "table": "桌面", "plate": "盘子", "background": "背景", "unknown": "未知"}
