"""
Train a YOLOv8 model on the Roboflow leaf detection dataset.
"""

import argparse
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DATA_YAML = Path(__file__).resolve().parent.parent / "models" / "leaf-detection" / "data.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on the leaf detection dataset.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base YOLO model (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if OOM).")
    parser.add_argument("--device", default=None, help="Device: 'cpu', '0', '0,1', etc. Auto-detected if omitted.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--name", default="leaf-detection", help="Run name (saved under runs/detect/).")
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("GPU detectada: %s (%.1f GB VRAM)", gpu, vram)
    else:
        log.warning("CUDA não disponível — treinando na CPU")

    log.info("Modelo base: %s", args.model)
    log.info("Dataset: %s", DATA_YAML)
    log.info("Epochs: %d | Batch: %d | Imgsz: %d | Workers: %d", args.epochs, args.batch, args.imgsz, args.workers)

    model = YOLO(args.model)

    log.info("Iniciando treinamento...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project="runs/detect",
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    log.info("Treinamento finalizado! Resultados salvos em: %s", save_dir)
    log.info("Melhor modelo: %s", save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
