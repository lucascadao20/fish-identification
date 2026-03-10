#!/usr/bin/env python3
"""
Batch Fish Identification - Processa todas as imagens de uma pasta e salva
os resultados (imagens anotadas + CSV + JSON) em outra pasta.

Uso:
    python batch_identify.py --input pasta_entrada/ --output pasta_saida/
    python batch_identify.py --input fotos/ --output resultados/ --device cuda
    python batch_identify.py --input fotos/ --output resultados/ --conf 0.3 --topk 3
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from run_fish_detection import (
    FishDetector,
    FishClassifier,
    ensure_models,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def collect_images(input_dir: str) -> list[str]:
    """Coleta recursivamente todas as imagens de uma pasta."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"ERRO: '{input_dir}' nao e um diretorio valido.")
        sys.exit(1)

    images = []
    for f in sorted(input_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(str(f))
    return images


def process_single_image(
    image_path: str,
    detector: FishDetector,
    classifier: FishClassifier,
    output_dir: str,
    topk: int,
    crop_margin: float = 0.05,
) -> list[dict]:
    """Processa uma imagem: detecta peixes, classifica e salva resultados."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return []

    h, w = image_bgr.shape[:2]

    # Detectar peixes
    detections = detector.detect(image_bgr)
    if not detections:
        return []

    results = []
    annotated = image_bgr.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]

        # Margem ao redor do crop
        mx = int((x2 - x1) * crop_margin)
        my = int((y2 - y1) * crop_margin)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)

        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # Classificar especie
        species_preds = classifier.classify(crop, topk=topk)
        top_species = species_preds[0] if species_preds else {"species": "unknown", "confidence": 0}

        results.append({
            "fish_id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "detection_confidence": det["confidence"],
            "top_species": top_species["species"],
            "top_confidence": top_species["confidence"],
            "all_predictions": species_preds,
        })

        # Desenhar na imagem anotada
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{top_species['species']} ({top_species['confidence']:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Salvar crop individual do peixe
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        stem = Path(image_path).stem
        crop_path = os.path.join(crops_dir, f"{stem}_fish{i+1}.jpg")
        cv2.imwrite(crop_path, crop)

    # Salvar imagem anotada
    annotated_dir = os.path.join(output_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    out_img_path = os.path.join(annotated_dir, Path(image_path).stem + "_annotated.jpg")
    cv2.imwrite(out_img_path, annotated)

    return results


def print_progress(current: int, total: int, image_name: str, n_fish: int, elapsed: float):
    """Exibe barra de progresso no terminal."""
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    avg_time = elapsed / current if current > 0 else 0
    remaining = avg_time * (total - current)

    fish_info = f"{n_fish} peixe(s)" if n_fish > 0 else "nenhum peixe"
    print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%) | {image_name}: {fish_info} | "
          f"Restante: {remaining:.0f}s", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Identificacao de peixes em lote - processa pasta de imagens"
    )
    parser.add_argument("--input", "-i", required=True, help="Pasta com imagens de entrada")
    parser.add_argument("--output", "-o", required=True, help="Pasta para salvar resultados")
    parser.add_argument("--device", default="cpu", help="Dispositivo: 'cpu' ou 'cuda' (default: cpu)")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confianca do detector (default: 0.25)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K especies por peixe (default: 5)")
    parser.add_argument("--skip_download", action="store_true", help="Pular verificacao de modelos")
    args = parser.parse_args()

    # Coletar imagens
    image_paths = collect_images(args.input)
    if not image_paths:
        print(f"Nenhuma imagem encontrada em '{args.input}'")
        return

    print(f"Encontradas {len(image_paths)} imagens em '{args.input}'")

    # Verificar modelos
    if not args.skip_download:
        print("\nVerificando modelos...")
        ensure_models()

    # Criar pasta de saida
    os.makedirs(args.output, exist_ok=True)

    # Carregar modelos
    print("\nCarregando modelos...")
    detector = FishDetector(device=args.device, conf=args.conf)
    classifier = FishClassifier(device=args.device)

    # Processar imagens
    print(f"\nProcessando {len(image_paths)} imagens...\n")

    all_csv_rows = []
    all_json_results = {}
    total_fish = 0
    images_with_fish = 0
    start_time = time.time()

    for idx, img_path in enumerate(image_paths, 1):
        results = process_single_image(img_path, detector, classifier, args.output, args.topk)

        n_fish = len(results)
        total_fish += n_fish
        if n_fish > 0:
            images_with_fish += 1

        # Acumular resultados para CSV
        for res in results:
            all_csv_rows.append({
                "image": os.path.basename(img_path),
                "image_path": img_path,
                "fish_id": res["fish_id"],
                "bbox_x1": res["bbox"][0],
                "bbox_y1": res["bbox"][1],
                "bbox_x2": res["bbox"][2],
                "bbox_y2": res["bbox"][3],
                "detection_confidence": res["detection_confidence"],
                "species": res["top_species"],
                "species_confidence": res["top_confidence"],
            })

        # Acumular resultados para JSON
        all_json_results[img_path] = results

        elapsed = time.time() - start_time
        print_progress(idx, len(image_paths), os.path.basename(img_path), n_fish, elapsed)

    elapsed_total = time.time() - start_time
    print("\n")

    # Salvar CSV com todos os resultados
    csv_path = os.path.join(args.output, "results.csv")
    if all_csv_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_csv_rows)

    # Salvar JSON completo
    json_path = os.path.join(args.output, "results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_json_results, f, indent=2, ensure_ascii=False)

    # Resumo final
    print("=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  Imagens processadas:     {len(image_paths)}")
    print(f"  Imagens com peixes:      {images_with_fish}")
    print(f"  Total de peixes:         {total_fish}")
    print(f"  Tempo total:             {elapsed_total:.1f}s")
    if len(image_paths) > 0:
        print(f"  Tempo medio por imagem:  {elapsed_total/len(image_paths):.2f}s")
    print(f"")
    print(f"  Resultados salvos em:    {args.output}/")
    print(f"    - annotated/           Imagens com bounding boxes e nomes")
    print(f"    - crops/               Recortes individuais de cada peixe")
    print(f"    - results.csv          Tabela com todas as deteccoes")
    print(f"    - results.json         Resultados completos em JSON")
    print("=" * 60)


if __name__ == "__main__":
    main()
