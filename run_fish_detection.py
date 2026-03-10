#!/usr/bin/env python3
"""
Fish Detection & Classification Pipeline

Downloads pre-trained models (if needed) and runs end-to-end fish detection
and species classification on input images.

Usage:
    python run_fish_detection.py --image path/to/fish_image.jpg
    python run_fish_detection.py --image path/to/fish_image.jpg --output results/
    python run_fish_detection.py --image_dir path/to/images/ --output results/
"""

import argparse
import json
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Optional imports for kNN search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DETECTOR_DIR = MODELS_DIR / "detector"
CLASSIFIER_DIR = MODELS_DIR / "classifier"
LABELS_PATH = BASE_DIR / "labels.json"

DETECTOR_MODEL_PATH = DETECTOR_DIR / "model.pt"
CLASSIFIER_CKPT_PATH = CLASSIFIER_DIR / "model.ckpt"
DATABASE_PATH = CLASSIFIER_DIR / "database.pt"

# ─── Download URLs ───────────────────────────────────────────────────────────

DETECTOR_URL = "https://storage.googleapis.com/fishial-ml-resources/detector_v26_n3.zip"
CLASSIFIER_URL = "https://storage.googleapis.com/fishial-ml-resources/classification_model_v0.10.zip"


def download_and_extract(url: str, extract_dir: Path, name: str):
    """Download a zip file and extract it."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path = extract_dir / "download.zip"

    print(f"  Downloading {name}...")
    urllib.request.urlretrieve(url, str(zip_path))

    print(f"  Extracting {name}...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_dir))
    zip_path.unlink()
    print(f"  {name} ready.")


def ensure_models():
    """Download models if they don't exist."""
    if not DETECTOR_MODEL_PATH.exists():
        print("[1/2] Detector model not found. Downloading...")
        download_and_extract(DETECTOR_URL, DETECTOR_DIR, "YOLO Fish Detector")
    else:
        print("[1/2] Detector model found.")

    if not CLASSIFIER_CKPT_PATH.exists() or not DATABASE_PATH.exists():
        print("[2/2] Classifier model not found. Downloading...")
        download_and_extract(CLASSIFIER_URL, CLASSIFIER_DIR, "BEiTv2 Classifier")
    else:
        print("[2/2] Classifier model found.")


# ─── ViT Attention Pooling ───────────────────────────────────────────────────

class ViTAttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(in_features // 4, 128)
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        attention_scores = self.attention_net(x)
        weights = F.softmax(attention_scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled


# ─── ArcFace Head ────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=64.0, m=0.2):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, normalized_emb, labels=None):
        normalized_w = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_emb, normalized_w)
        return cosine * self.s


# ─── Classification Model ───────────────────────────────────────────────────

class FishClassificationModel(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=775,
                 backbone_model_name='beitv2_base_patch16_224.in1k_ft_in22k_in1k',
                 arcface_s=64.0, arcface_m=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_model_name, pretrained=False, num_classes=0)
        self.backbone_out_features = self.backbone.embed_dim
        self.pooling = ViTAttentionPooling(self.backbone_out_features)
        self.embedding_fc = nn.Sequential(nn.Linear(self.backbone_out_features, embedding_dim))
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if hasattr(self.backbone, 'cls_token'):
            patch_tokens = features[:, 1:, :]
        else:
            patch_tokens = features
        pooled = self.pooling(patch_tokens)
        emb_raw = self.embedding_fc(pooled)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.arcface_head(emb_norm)
        return emb_norm, logits


# ─── Fish Classifier (loads checkpoint + database) ──────────────────────────

class FishClassifier:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._load_labels()
        self._load_database()
        self._load_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"  Classifier ready on {device} ({self.num_classes} species)")

    def _load_labels(self):
        with open(LABELS_PATH, 'r') as f:
            self.labels_json = json.load(f)

    def _load_database(self):
        data = torch.load(str(DATABASE_PATH), map_location="cpu", weights_only=False)
        self.db_embeddings = data['embeddings'].numpy().astype("float32")
        self.db_labels = np.array(data['labels'])
        self.keys = data['labels_keys']
        self.num_classes = len(self.keys)

        self.id_to_label = {int(k): v['label'] for k, v in self.keys.items()}

        # Build centroids
        unique_labels = np.unique(self.db_labels)
        centroids = []
        self.centroid_labels = []
        for lbl in unique_labels:
            embs = self.db_embeddings[self.db_labels == lbl]
            c = embs.mean(axis=0)
            c /= (np.linalg.norm(c) + 1e-10)
            centroids.append(c)
            self.centroid_labels.append(lbl)
        self.centroid_matrix = np.stack(centroids).astype("float32")
        print(f"  Database loaded: {len(self.db_embeddings)} embeddings, {len(unique_labels)} classes")

    def _load_model(self):
        ckpt = torch.load(str(CLASSIFIER_CKPT_PATH), map_location="cpu", weights_only=False)
        hparams = ckpt.get('hyper_parameters', {})

        self.model = FishClassificationModel(
            embedding_dim=hparams.get('embedding_dim', 512),
            num_classes=hparams.get('num_classes', 775),
            backbone_model_name=hparams.get('backbone_model_name', 'beitv2_base_patch16_224.in1k_ft_in22k_in1k'),
            arcface_s=hparams.get('arcface_s', 64.0),
            arcface_m=hparams.get('arcface_m', 0.2),
        )

        state_dict = ckpt['state_dict']
        # Remove 'model.' prefix from Lightning checkpoint keys
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_sd[k[6:]] = v
            else:
                new_sd[k] = v
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Classification model loaded from checkpoint")

    def classify(self, crop_bgr: np.ndarray, topk: int = 5) -> List[Dict]:
        """Classify a single fish crop (BGR numpy array)."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb, logits = self.model(tensor)

        # ArcFace top-k
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, topk)

        results = []
        for i in range(topk):
            idx = top_indices[0][i].item()
            label = self.id_to_label.get(idx, f"class_{idx}")
            score = top_probs[0][i].item()
            results.append({"species": label, "confidence": round(score * 100, 2)})

        return results


# ─── Fish Detector (YOLO) ───────────────────────────────────────────────────

class FishDetector:
    def __init__(self, device: str = "cpu", conf: float = 0.25):
        self.device = device
        self.conf = conf
        self.model = YOLO(str(DETECTOR_MODEL_PATH))
        self.model.to(device)
        print(f"  Detector ready on {device}")

    def detect(self, image_bgr: np.ndarray) -> List[Dict]:
        """Detect fish in an image. Returns list of bounding boxes."""
        results = self.model.predict(
            source=image_bgr,
            imgsz=640,
            conf=self.conf,
            iou=0.45,
            device=self.device,
            verbose=False,
            save=False
        )

        detections = []
        if results and len(results) > 0:
            boxes_data = results[0].boxes.data.cpu().numpy()
            for box in boxes_data:
                x1, y1, x2, y2, confidence = box[:5]
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(float(confidence), 3)
                })
        return detections


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(image_path: str, detector: FishDetector, classifier: FishClassifier,
                 output_dir: str = None, topk: int = 5) -> List[Dict]:
    """Run full detection + classification pipeline on a single image."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  ERROR: Could not read image: {image_path}")
        return []

    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"  Size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # Step 1: Detect fish
    detections = detector.detect(image_bgr)
    print(f"  Fish detected: {len(detections)}")

    if not detections:
        print("  No fish detected in this image.")
        return []

    all_results = []

    # Step 2: Classify each detected fish
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        # Add margin around crop
        h, w = image_bgr.shape[:2]
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(w, x2 + margin_x)
        cy2 = min(h, y2 + margin_y)

        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        species = classifier.classify(crop, topk=topk)

        result = {
            "fish_id": i + 1,
            "bbox": det["bbox"],
            "detection_confidence": det["confidence"],
            "species_predictions": species
        }
        all_results.append(result)

        top = species[0] if species else {"species": "unknown", "confidence": 0}
        print(f"\n  Fish #{i+1} (conf: {det['confidence']:.2f})")
        print(f"    Box: [{x1}, {y1}, {x2}, {y2}]")
        for j, sp in enumerate(species[:3]):
            marker = ">>>" if j == 0 else "   "
            print(f"    {marker} {sp['species']}: {sp['confidence']:.1f}%")

    # Save annotated image
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        annotated = image_bgr.copy()

        for res in all_results:
            x1, y1, x2, y2 = res["bbox"]
            top_sp = res["species_predictions"][0]
            label = f"{top_sp['species']} ({top_sp['confidence']:.1f}%)"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        out_name = Path(image_path).stem + "_result.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"\n  Annotated image saved: {out_path}")

        # Save JSON results
        json_path = os.path.join(output_dir, Path(image_path).stem + "_result.json")
        with open(json_path, 'w') as f:
            json.dump({"image": image_path, "results": all_results}, f, indent=2)
        print(f"  JSON results saved: {json_path}")

    return all_results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fish Detection & Species Classification")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to a directory of images")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--topk", type=int, default=5, help="Number of top species predictions")
    parser.add_argument("--skip_download", action="store_true", help="Skip model download check")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image_dir")

    # Step 0: Ensure models are downloaded
    if not args.skip_download:
        print("Checking models...")
        ensure_models()

    # Step 1: Initialize models
    print("\nLoading models...")
    detector = FishDetector(device=args.device, conf=args.conf)
    classifier = FishClassifier(device=args.device)

    # Step 2: Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for f in sorted(Path(args.image_dir).iterdir()):
            if f.suffix.lower() in exts:
                image_paths.append(str(f))

    if not image_paths:
        print("No images found.")
        return

    print(f"\nProcessing {len(image_paths)} image(s)...")

    # Step 3: Run pipeline
    all_results = {}
    for img_path in image_paths:
        results = run_pipeline(img_path, detector, classifier, args.output, args.topk)
        all_results[img_path] = results

    # Summary
    total_fish = sum(len(r) for r in all_results.values())
    print(f"\n{'='*60}")
    print(f"Done! Processed {len(image_paths)} image(s), detected {total_fish} fish total.")
    if args.output:
        print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
