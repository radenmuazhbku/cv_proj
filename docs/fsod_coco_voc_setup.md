# Standard Few-Shot Object Detection (FSOD) Benchmark Setup: MS COCO & PASCAL VOC

This document details the standard setup for Few-Shot Object Detection (FSOD) established in literature (Kang et al. ICCV 2019, Meta R-CNN ICCV 2019, and standardized by TFA / Wang et al. ICML 2020 via `fsdet`). Subsequent benchmarks (DeFRCN, FSCE, MFDC, Meta-DETR, De-DETR, etc.) follow this exact protocol.

---

## 1. PASCAL VOC Setup

### 1.1 Datasets Used
* **Training & Sampling Pool:** VOC 2007 `trainval` + VOC 2012 `trainval` (combined: $\approx 16{,}551$ images).
* **Testing:** VOC 2007 `test` ($4{,}952$ images).

### 1.2 Class Splits (Base vs. Novel)
PASCAL VOC consists of 20 categories. The literature evaluates on **3 standard random splits**, each dividing the dataset into **15 Base** and **5 Novel** classes:

```python
# All 20 PASCAL VOC categories
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Split 1
SPLIT_1_NOVEL = ['bird', 'bus', 'cow', 'motorbike', 'sofa']
SPLIT_1_BASE  = [c for c in VOC_CLASSES if c not in SPLIT_1_NOVEL]

# Split 2
SPLIT_2_NOVEL = ['aeroplane', 'bottle', 'cow', 'horse', 'sofa']
SPLIT_2_BASE  = [c for c in VOC_CLASSES if c not in SPLIT_2_NOVEL]

# Split 3
SPLIT_3_NOVEL = ['boat', 'cat', 'motorbike', 'sheep', 'sofa']
SPLIT_3_BASE  = [c for c in VOC_CLASSES if c not in SPLIT_3_NOVEL]
```

### 1.3 Shot Settings ($K$) and Random Seeds
* **Shots ($K$):** $K \in \{1, 2, 3, 5, 10\}$.
* **Seeds:** Performance varies significantly depending on which instances are sampled. Literature evaluates across **10 random seeds** (seeds `0` to `9` from TFA / `fsdet`) and reports the **mean $\text{AP}_{50}$**.
* Standard reporting: Splits 1, 2, and 3 are either reported individually across seeds or averaged.

### 1.4 Metrics
* **Primary Metric:** $\text{AP}_{50}$ (Average Precision at $\text{IoU} = 0.50$, evaluated using VOC 2007 11-point interpolation or VOC 2012 continuous AUC).
* Report breakdown:
  * **$\text{nAP}_{50}$** (Novel class $\text{AP}_{50}$) $\rightarrow$ Primary benchmark metric.
  * **$\text{bAP}_{50}$** (Base class $\text{AP}_{50}$).
  * **$\text{mAP}_{50}$** (Overall / all 20 classes $\text{AP}_{50}$).

---

## 2. MS COCO Setup

### 2.1 Datasets Used
* **Base Training & Sampling Pool:** COCO `trainval35k` (combination of `train2014` and `val2014` minus the $5{,}000$ minival images) or COCO `train2017`.
* **Testing:** COCO `minival5k` (identical to COCO `val2017`, $5{,}000$ images).

### 2.2 Class Splits (60 Base / 20 Novel)
The **20 Novel classes** in MS COCO are chosen to be **the 20 classes that overlap with PASCAL VOC**. The remaining **60 classes** serve as Base classes.

```python
# 20 Novel Classes (using standard COCO 80 category naming)
COCO_NOVEL_CLASSES = [
    'airplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'couch',
    'cow', 'dining table', 'dog', 'horse', 'motorcycle',
    'person', 'potted plant', 'sheep', 'train', 'tv'
]

# Standard COCO category IDs for Novel classes (1-indexed COCO category_ids):
COCO_NOVEL_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
]

# 60 Base Classes: All other 60 classes in the 80-class COCO ontology
```

### 2.3 Shot Settings ($K$) and Random Seeds
* **Shots ($K$):** $K \in \{1, 2, 3, 5, 10, 30\}$.
  * $K = 10$ and $K = 30$ are the most universally reported; comprehensive papers include all 6.
* **Seeds:** Standard TFA benchmark provides **10 predefined seeds** (`seed0` to `seed9`). Literature reports the **mean over the 10 seeds**.

### 2.4 Metrics
Standard COCO evaluation protocol (averaged across 10 IoU thresholds from $0.50$ to $0.95$ with step $0.05$):
* **Primary Metric:** $\text{nAP}$ ($\text{AP}_{0.50:0.95}$ on Novel classes).
* Additional metrics:
  * $\text{nAP}_{50}$ and $\text{nAP}_{75}$ (Novel classes at $\text{IoU} = 0.50$ and $0.75$).
  * $\text{bAP}$ / $\text{bAP}_{50}$ (Base classes).
  * $\text{AP}_{\text{all}}$ (All 80 classes).

---

## 3. Implementation Details & Two-Stage Pipeline

```
[Phase 1: Base Training]
Full dataset filtered to Base classes only ───► Pretrained Base Detector
                                                       │
                                                       ▼
[Phase 2: Few-Shot Fine-Tuning]
Balanced K-shot data (Base + Novel)         ───► Freeze backbone/head layers
                                                 & Fine-tune classifier/regressor
```

### 3.1 What Exactly is a "Shot"?
* In FSOD, **$K$-shot means exactly $K$ bounding box instances per category**, **NOT** $K$ images.
* Because an image may contain multiple instances or classes:
  1. An image is chosen to provide an instance of class $C$.
  2. If the image has extra instances of other classes or multiple instances of $C$, standard sampling algorithms (such as TFA's) truncate or filter the annotation file so that **each class has exactly $K$ bounding box annotations** in the few-shot annotation file.
* **Reproducibility Note:** Do not generate random shots from scratch when comparing against published literature. Instead, use the exact `.json` annotation files released by the [TFA / fsdet repository](https://github.com/ucbdrive/few-shot-object-detection).

### 3.2 Phase 1: Base Training
1. **Filtering Annotations:**
   * Keep **only** annotations of Base classes.
   * Strip out all Novel class bounding boxes.
   * Discard any images that contain **only** novel classes (zero base annotations).
   * If an image has both base and novel objects, keep the image and its base bounding boxes, but discard novel bounding boxes (they are treated as background).
2. **Backbone & Architecture:**
   * Standard benchmark backbone: **Faster R-CNN with ResNet-101 + FPN** (or Deformable DETR / RF-DETR for Transformer-based FSOD).
3. **Training Schedule:**
   * VOC: Standard 18k iterations / $\sim 12$ epochs.
   * COCO: Standard $1\times$ ($90\text{k}$ iterations) or $2\times$ ($180\text{k}$ iterations) schedule.

### 3.3 Phase 2: Few-Shot Fine-Tuning (Class-Balanced)
1. **Dataset Construction:**
   * Build a balanced few-shot set with **$K$ instances for ALL classes** (both Base and Novel):
     * VOC: $15 \text{ base} \times K + 5 \text{ novel} \times K = 20 \times K$ instances.
     * COCO: $60 \text{ base} \times K + 20 \text{ novel} \times K = 80 \times K$ instances.
   * *Rationale:* Class-balanced fine-tuning prevents catastrophic forgetting of base categories.
2. **Parameter Freezing / Updating:**
   * **TFA (Baseline):** Freeze the backbone, FPN, and RPN. Only fine-tune the final classification & bounding box regression linear layers in the RoI head.
   * **DeFRCN / FSCE:** Allow gradient flow through the backbone/RPN with Gradient Decoupled Layers (GDL) or reduced learning rates (e.g., $0.1\times$ or $0.01\times$ backbone LR).
   * **DETR-based FSOD:** Fine-tune decoder queries, cross-attention, and classification/box projection heads while freezing or setting a small LR on the backbone/encoder.
3. **Iterations & Learning Rate:**
   * Training budget is intentionally kept short to prevent overfitting:
     * VOC: typically $1{,}000$ to $4{,}000$ iterations.
     * COCO: typically $2{,}000$ to $8{,}000$ iterations.
   * Learning rate is typically dropped by $10\times$ to $100\times$ compared to base training.

---

## 4. Summary Table

| Attribute | PASCAL VOC | MS COCO |
| :--- | :--- | :--- |
| **Total Classes** | 20 | 80 |
| **Base / Novel Split** | 15 Base / 5 Novel (3 standard splits) | 60 Base / 20 Novel (20 novel = 20 VOC classes) |
| **Base Training Data** | VOC 2007 trainval + VOC 2012 trainval | COCO train2014 (trainval35k) or train2017 |
| **Test Set** | VOC 2007 test (4,952 images) | COCO minival5k (val2017, 5,000 images) |
| **Shots ($K$)** | 1, 2, 3, 5, 10 | 1, 2, 3, 5, 10, 30 |
| **Number of Seeds** | 10 seeds per split | 10 seeds (seed 0 to 9) |
| **Primary Metric** | $\text{nAP}_{50}$ (Novel AP at IoU 0.50) | $\text{nAP}$ (Novel AP at IoU 0.50:0.95) |
| **Secondary Metrics** | $\text{bAP}_{50}$, $\text{mAP}_{50}$ | $\text{nAP}_{50}$, $\text{nAP}_{75}$, $\text{bAP}$, $\text{AP}_{\text{all}}$ |
| **Annotation Rule** | $K$ box instances per class in fine-tuning | $K$ box instances per class in fine-tuning |

---

## 5. In-Context Learning (ICL) Paradigm in RF-DETR PCS

Unlike traditional two-stage fine-tuning (TFA) which optimizes model weights on the few-shot set at test time, RF-DETR PCS employs **Training-Free In-Context Learning (ICL)**:
* **Base Training**: The model is trained on Base classes to learn universal visual prompt conditioning via support-query episodes.
* **Test Time (Zero Training)**: Model parameters remain completely frozen ($\nabla \theta = 0$). The $K$-shot support exemplars are passed through the visual prompt encoder, conditioning the queries to detect novel categories directly in feed-forward inference.

### Implemented Pipeline & Scripts
1. **VOC XML to COCO Format**:
   ```bash
   python scripts/convert_voc_to_coco.py --voc-dir datasets/VOC2012 --output-dir datasets/VOC2012/annotations --splits train val trainval
   ```
2. **FSOD Benchmark Split Generator**:
   ```bash
   # MS COCO
   python scripts/rfdetr_pcs/generate_fsod_splits.py --dataset coco --shots 1 2 3 5 10 30 --seeds 0
   # PASCAL VOC
   python scripts/rfdetr_pcs/generate_fsod_splits.py --dataset voc --shots 1 2 3 5 10 --seeds 0
   ```
3. **Configurations (200k-step Compute Budget)**:
   - VOC 1-Shot: [`configs/rfdetr_pcs/rfdetr_pcs_voc_split1_fs1.py`](file:///home/muaz/code/cv_proj/configs/rfdetr_pcs/rfdetr_pcs_voc_split1_fs1.py)
   - VOC 5-Shot: [`configs/rfdetr_pcs/rfdetr_pcs_voc_split1_fs5.py`](file:///home/muaz/code/cv_proj/configs/rfdetr_pcs/rfdetr_pcs_voc_split1_fs5.py)
   - COCO 1-Shot: [`configs/rfdetr_pcs/rfdetr_pcs_coco_fs1.py`](file:///home/muaz/code/cv_proj/configs/rfdetr_pcs/rfdetr_pcs_coco_fs1.py)
   - COCO 5-Shot: [`configs/rfdetr_pcs/rfdetr_pcs_coco_fs5.py`](file:///home/muaz/code/cv_proj/configs/rfdetr_pcs/rfdetr_pcs_coco_fs5.py)
4. **Full Training Command**:
   ```bash
   python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/rfdetr_pcs_coco_fs5.py
   ```
5. **Training-Free In-Context Evaluation Command**:
   ```bash
   python scripts/rfdetr_pcs/evaluate_in_context_fsod.py \
       --checkpoint <path_to_checkpoint.pt> \
       --config configs/rfdetr_pcs/rfdetr_pcs_coco_fs5.py \
       --support-annotation datasets/mscoco_fsod_splits/coco_fsod_novel_5shot_seed0.json \
       --support-image-dir datasets/mscoco/train2017 \
       --test-annotation datasets/mscoco_fsod_splits/coco_val_full.json \
       --test-image-dir datasets/mscoco/val2017 \
       --novel-categories 1 2 3 4 5 6 7 9 16 17 18 19 20 21 44 62 63 64 67 72 \
       --shots 5 \
       --segmentation \
       --output-json logs/coco_fs5_icl_eval.json
   ```


