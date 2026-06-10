#!/usr/bin/env python3
"""
Train / export the FloatPro ball detector (YOLOv8n).

Train:
    python ml/train_yolo.py --data ml/dataset/data.yaml --epochs 100

Export best weights to ONNX (then trtexec on the Jetson → TensorRT):
    python ml/train_yolo.py --export runs/detect/train/weights/best.pt

Requires: pip install ultralytics
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="base weights (n=nano is right for Orin Nano)")
    ap.add_argument("--export", metavar="WEIGHTS",
                    help="export trained weights to ONNX instead of training")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return 1

    if args.export:
        model = YOLO(args.export)
        path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"\nONNX → {path}")
        print("On the Jetson:")
        print(f"  /usr/src/tensorrt/bin/trtexec --onnx={path} "
              f"--saveEngine=ball.engine --fp16")
        return 0

    if not args.data:
        ap.error("--data required for training (or use --export)")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=-1,            # auto batch size
        patience=20,         # early stop
        degrees=10,          # mild rotation aug — ball is rotation-invariant
        scale=0.5,           # serves vary a lot in apparent ball size
        fliplr=0.5,
        mosaic=1.0,
    )
    print("\nTraining done. Validate, then export:")
    print("  python ml/train_yolo.py --export runs/detect/train/weights/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
