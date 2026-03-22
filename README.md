# Hierarchical Grocery Detector (hYOLO V4)

End-to-end hierarchical object detection for fine-grained grocery product recognition on retail shelf images. Built by Ryan Marinelli for NM i AI 2026 (Norway's National AI Championship), Task 3 — NorgesGruppen Data.

## Overview

Standard object detectors assign a single flat class label per detection. On a grocery shelf with hundreds of visually similar products this falls short — the model has no way to express that misidentifying a Nescafé variant is a less severe error than misidentifying it as a completely different brand.

This repo implements a 4-level hierarchical detection model: a YOLOv8x backbone and bbox regression head paired with a custom hierarchical classification head that simultaneously predicts each detected product at four levels of specificity.

| Level | Description | Classes |
|-------|-------------|---------|
| L0 | Section (e.g. hot drinks, breakfast) | 4 |
| L1 | Format / type (e.g. filter coffee, capsules) | 30 |
| L2 | Brand (e.g. Nescafé, Friele, Evergood) | 149 |
| L3 | Specific product | 323 |

## Repository Structure

```
hyolo_finetune.py      # Model definition, loss, training loop
hyolo_posttrain.py     # Phase 3: HierSupCon contrastive post-training
                       # Phase 4: prototype calibration
build_submission.py    # Export to ONNX + package submission zip
check_data.py          # Dataset validation before training
run.py                 # ONNX inference entry point
hierarchy.json         # 4-level product taxonomy
```

## Architecture

The model extends the hYOLO architecture (Tsenkova et al., arXiv 2510.23278) from a classification-only system into a full end-to-end detector:

- **Backbone**: YOLOv8x, pretrained on SKU-110K (generic shelf product detection)
- **Neck**: Standard YOLOv8 PANet
- **Bbox head**: Standard YOLOv8 anchor-free regression (DFL + CIoU)
- **Hierarchical cls head** (hYOLO V4):
  - Per-level Conv branches with bidirectional cross-level information flow
  - Each level receives the previous level's prediction as context
  - 4 independent classification outputs (L0–L3)

Key differences from the original hYOLO paper:
- Full detection pipeline — localises and classifies simultaneously from raw shelf images
- Bidirectional gradient flow between hierarchy levels
- Per-level loss weighting to compensate for cascaded gradients
- IoU-matched validation (not raw anchor top-k)

## Installation

```bash
pip install ultralytics==8.1.0 torch torchvision
pip install ensemble-boxes  # for WBF inference
```

## Training

**Step 1 — Prepare data**

Your dataset should follow YOLO format. Class IDs in label files must correspond to L3 leaf indices in `hierarchy.json`. Run the data check before training:

```bash
python check_data.py \
    --data      /path/to/dataset \
    --hierarchy hierarchy.json \
    --yaml      /path/to/dataset.yaml
```

**Step 2 — Pretrain on SKU-110K** (optional but recommended)

Train a standard YOLOv8x on SKU-110K first to get a strong shelf-aware backbone, then pass those weights via `--weights`.

**Step 3 — Fine-tune**

```bash
python hyolo_finetune.py \
    --data      dataset.yaml \
    --hierarchy hierarchy.json \
    --weights   sku110k_pretrain/weights/best.pt \
    --epochs    300 \
    --imgsz     1280 \
    --batch     4 \
    --device    0
```

Training runs in two phases:
- Phase 1 (default 50 epochs): backbone frozen, only hierarchical cls head trains
- Phase 2 (remaining epochs): full fine-tune with gradual backbone unfreeze

Best checkpoint saved by EMA F1Hier to `runs/hyolo/train/best.pt`.

**Key training arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--freeze-epochs` | 50 | Phase 1 duration |
| `--alpha` | 25.0 | Hierarchy consistency penalty strength |
| `--label-smoothing` | 0.1 | BCE label smoothing |
| `--pos-weight-cap` | 10.0 | Max inverse-frequency weight for rare classes |
| `--reward-weight` | 0.0 | Symmetric penalty reward (0 = asymmetric only) |
| `--copy-paste` | 0.3 | Copy-paste augmentation probability |

## Post-training (optional)

After main training, run prototype calibration to correct frequency bias on rare classes:

```bash
python hyolo_posttrain.py \
    --checkpoint runs/hyolo/train/best.pt \
    --data       dataset.yaml \
    --hierarchy  hierarchy.json \
    --epochs     0 \
    --device     0
```

`--epochs 0` skips contrastive post-training and only builds the prototype bank. Set `--epochs 20` to also run HierSupCon contrastive fine-tuning (recommended only with larger datasets — at least 5 images per leaf class).

## Inference

**Build submission zip (ONNX export)**

```bash
python build_submission.py \
    --checkpoint runs/hyolo/train/best.pt \
    --hierarchy  hierarchy.json \
    --coco-map   coco_to_l3.json \
    --annotations annotations.json \
    --imgsz      1024 \
    --output     submission.zip
```

**Run inference**

```bash
python run.py \
    --input  /path/to/images \
    --output predictions.json
```

Output is a COCO-format JSON array:

```json
[
  {
    "image_id": 42,
    "category_id": 177,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

## Performance

Evaluated on held-out validation set using hierarchical F1 (Kiritchenko et al. 2006):

| Metric | Value |
|--------|-------|
| F1Hier | 0.9426 |
| L0 accuracy (section) | 0.972 |
| L1 accuracy (format) | 0.949 |
| L2 accuracy (brand) | 0.926 |
| L3 accuracy (product) | 0.884 |

Training hardware: NVIDIA A100 (Google Colab Pro+), 248 training images.

## Limitations

- Trained on Norwegian grocery products — performance on other retailers will vary
- 1 out of 323 leaf classes has no training examples and cannot be predicted
- Optimised for shelf-level photography; overhead or very close-up angles reduce accuracy

## Relationship to hYOLO Paper

The hierarchical classification head is inspired by Tsenkova et al. (arXiv 2510.23278), which proposes hYOLO for hierarchical classification of pre-cropped images. This repo extends that work into a full detection system — the bbox regression pipeline, Task-Aligned Assigner, IoU-matched validation, bidirectional gradient flow, and all training improvements are original contributions.

## Citation

If you use this code, please cite:

```bibtex
@misc{marinelli2026hyolo,
  title  = {Hierarchical Grocery Detector (hYOLO V4)},
  author = {Ryan Marinelli},
  year   = {2026},
  url    = {https://github.com/zrmarine/hierarchical-grocery-detecto}
}
```

If building on the hYOLO architecture, also cite the original paper:

```bibtex
@misc{tsenkova2025hyolo,
  title         = {hYOLO Model: Enhancing Object Classification with Hierarchical Context in YOLOv8},
  author        = {Veska Tsenkova and Peter Stanchev and Daniel Petrov and Deyan Lazarov},
  year          = {2025},
  eprint        = {2510.23278},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2510.23278}
}
```

## Acknowledgements

- hYOLO architecture: Tsenkova et al., arXiv 2510.23278 (2025)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- SKU-110K: Goldman et al., "Precise Detection in Densely Packed Scenes" (CVPR 2019)
- Hierarchical F1: Kiritchenko et al. (2006)
- NM i AI 2026 organised by [Astar Technologies](https://astarconsulting.no)

## License

MIT
