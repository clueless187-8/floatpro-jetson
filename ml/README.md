# FloatPro ML — YOLOv8 ball detector

Replaces `detect_ball_simple` (threshold + blob) with a trained
detector once real gym footage exists. The simple detector works on
bright balls against dark backgrounds; gyms have white walls, white
jerseys, and white lights — a trained model is the fix.

## Pipeline

```
phone/Jetson clips → extract frames → label → train → export → deploy
```

### 1. Collect

15-20 serve clips covering the variation you'll see in production:
- both gyms you'll film in
- different jersey colors, lighting, ball types
- some clips with multiple balls visible (carts, shaggers)

### 2. Extract frames

```bash
python ml/extract_frames.py captures/  --every 5 --out ml/dataset/images/raw
```

Every 5th frame avoids near-duplicate frames poisoning the val split.
Target ~500 labeled frames for a solid first model.

### 3. Label

Use [Roboflow](https://roboflow.com) (free tier handles this size) or
[Label Studio](https://labelstud.io) locally. One class: `ball`.
Export in **YOLOv8 format** to `ml/dataset/` so it looks like:

```
ml/dataset/
├── data.yaml
├── images/{train,val}/*.png
└── labels/{train,val}/*.txt
```

Suggested split: 80/20, split by **clip** not by frame (frames from
the same clip in both splits inflates val accuracy).

### 4. Train

```bash
pip install ultralytics
python ml/train_yolo.py --data ml/dataset/data.yaml --epochs 100
```

Defaults to YOLOv8n (nano) — the right size for Jetson Orin Nano.
Trains fine on a laptop GPU or even CPU overnight at this dataset size.

### 5. Export for Jetson

```bash
python ml/train_yolo.py --export runs/detect/train/weights/best.pt
```

Produces ONNX. On the Jetson, convert to TensorRT:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=ball.engine --fp16
```

### 6. Deploy

Drop-in: implement `detect_ball_yolo()` matching the
`detect_ball_simple()` signature in `floatpro/spin_estimator.py`
(returns `Detection(cx, cy, r, frame_index)`), then switch the call in
`estimate_session_spin`. The interface was designed for this swap from
day one.

## Decision gate (from the README)

Run `python ingest_video.py <real_clip> --preview` first:
- **>80% detection hit rate** → simple detector is fine, skip ML for now
- **30-80%** → label + train before buying more hardware
- **<30%** → fix lighting/exposure first; ML can't rescue motion blur
