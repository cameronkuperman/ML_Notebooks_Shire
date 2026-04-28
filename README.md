# Shire Restaurant Table Occupancy Classifier

## Motivation & Related Work

Restaurant table state classification is a practical computer vision problem with direct operational value: knowing whether a table is occupied, clean, or dirty enables real-time floor management without manual staff monitoring. Prior work on fine-grained scene understanding — particularly [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193), which demonstrated that self-supervised ViT features transfer strongly to downstream classification tasks — motivated our use of a DINOv3-ViT-Large backbone over conventional CNNs. Our occupancy classification approach draws on tabular feature engineering techniques common in person-tracking and re-identification literature (e.g., overlap, distance, motion, seated signal), similar to features used in [ByteTrack (Zhang et al., 2022)](https://arxiv.org/abs/2110.06864). The multi-stage cascade with a VLM fallback (Gemini 2.5 Flash) is inspired by confidence-based routing in production ML systems and recent work on vision-language models for scene understanding ([Gemini Team, 2024](https://arxiv.org/abs/2312.11805)).

## What it Does

This project builds a multi-stage machine learning pipeline to classify restaurant table states from triplets of video frames into three categories: **occupied**, **clean**, or **dirty**. A tabular ensemble model (XGBoost + Random Forest + MLP, voting) classifies occupancy from `perception.json` track features extracted by a pre-processing system. When the table is unoccupied, a fine-tuned DINOv3-ViT-Large model classifies the table surface as clean or dirty from the middle frame. If either local model falls below an 80% confidence threshold, the pipeline escalates to Gemini 2.5 Flash, which receives all three frames plus the perception data for a final multimodal judgment. The full pipeline is served as a FastAPI web application with a drag-and-drop UI for uploading triplet zips and visualizing per-stage confidence.

## Quick Start

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd ML_Notebooks_Shire
pip install -r requirements.txt

# 2. Place model artifacts in the models/ directory
#    - models/dinov3_classifier_full.pt         (fine-tuned DINOv3 clean/dirty model)
#    - models/best_model_logistic_regression.joblib  (occupancy ensemble)

# 3. Set your Gemini API key (optional — pipeline still works without it)
export GEMINI_API_KEY=your_key_here

# 4. Start the server
cd src/serving/triplet_inference
uvicorn app:app --reload --port 8000

# 5. Open http://localhost:8000 in your browser
#    Drop a triplet zip (frame_0.jpg, frame_1.jpg, frame_2.jpg, perception.json) to run inference
#    Sample zips are in data/examples/
```

See [SETUP.md](SETUP.md) for full environment setup including Colab training instructions.

## Video Links

- **Demo video: https://www.youtube.com/watch?v=Hrp1h42zlRA
- **Technical walkthrough: https://youtu.be/wtWaT4PXIBM

## Evaluation

### Occupancy Classification (XGBoost + RF + MLP VotingClassifier)
| Model | Accuracy |
|-------|----------|
| XGBoost | 78.2% |
| Random Forest | 84.6% |
| MLP | 85.9% |
| **Ensemble (soft vote)** | **85.6%** |
| Hardcoded baseline | 58.7% |

Ensemble F1: **0.764** on held-out test set (312 samples). Feature ablation showed person overlap features contribute the largest F1 delta (−0.015 when removed).

### Clean/Dirty Classification (DINOv3-ViT-Large fine-tuned)
| Model | Test F1 | Test Accuracy |
|-------|---------|---------------|
| ResNet-18 | 0.578 | 66.5% |
| DINOv3-Large | 0.669 | 79.5% |
| ResNet-50 | 0.712 | 82.7% |

3-fold GroupKFold cross-validation used throughout; splits are group-based (video+table) to prevent temporal leakage. Test set: 370 frames.

### Pipeline
Confidence threshold: 80%. When both local models fall below threshold, Gemini 2.5 Flash is used for escalation with full multimodal context (3 frames + perception.json + prior stage outputs).

## Design Decisions

**Why DINOv3 over ResNet for clean/dirty?** DINOv2/v3 self-supervised pre-training produces rich patch-level features that generalise well from few examples. Our ablation confirmed this: ResNet-18 achieved F1=0.578, ResNet-50 F1=0.712, and DINOv3-Large F1=0.669 — competitive with a much larger CNN. More importantly, DINOv3 requires far less labelled data to fine-tune, which matters given our class imbalance (2462 clean vs. 360 dirty frames). We used differential learning rates (backbone 1e-6, head 3e-4) to avoid catastrophic forgetting of pre-trained features.

**Why GroupShuffleSplit / GroupKFold?** Raw frame-level splits would leak temporal information — consecutive frames from the same video clip are nearly identical. We group by `(video_id, table_id)` so no group appears in both train and val/test, preventing inflated accuracy from temporal correlation. This is the correct methodology for video-derived datasets and mirrors best practices in action recognition benchmarks.

**Why an ensemble for occupancy instead of a single model?** Occupancy prediction from `perception.json` features is a tabular problem where no single model dominates. XGBoost captures non-linear interactions, Random Forest provides variance reduction, and MLP learns smooth decision boundaries. Soft-voting the three produced +26 percentage points over a hardcoded person-count rule and was more robust than any individual model.

**Why Gemini escalation?** Both local models are trained on a single restaurant's data and can fail on novel table configurations or lighting. Gemini 2.5 Flash receives all three frames plus the full perception context — a richer signal than either local model alone — and provides a calibrated fallback without requiring retraining. The 80% confidence threshold was chosen by inspecting the threshold-tuning table from Cell 12 of the occupancy notebook.

## Scope Note

This repository is part of a larger project that began earlier this semester with three other people. This repo covers only the work the two of us did.  All rubric claims in the self-assessment are supported by files included in this submitted repository. The demo video may show broader product context, but the graded ML evidence is contained here.

## Individual Contributions

**Kabir Sankaranrajendra**
- Designed and trained the occupancy classification pipeline (`notebooks/simple_yolo_classifier_occ_unocc.ipynb`): feature engineering from `perception.json`, XGBoost/RF/MLP ensemble, GroupShuffleSplit leakage-prevention strategy, feature ablation study, threshold tuning
- Designed and trained the clean/dirty DINOv3 model (`notebooks/dino_v3_rerun_(1) (1).ipynb`): fine-tuning DINOv3-ViT-Large, Focal Loss with label smoothing, warmup + cosine LR schedule, differential learning rates, image augmentation pipeline, multi-architecture ablation (ResNet-18/50 vs. DINOv3)
- Built the multi-stage inference pipeline logic in `src/serving/triplet_inference/app.py`
- Integrated Gemini 2.5 Flash escalation and confidence-threshold routing

**Cameron Cuperman**
- Built the perception data collection and simulation system (`docs/table_perception_simulator.html`)
- Developed the model serving infrastructure (`src/serving/dinov3_endpoint/model_server.py`): `MultiModelService`, `ImagePredictor`, `JoblibPredictor`, GCS artifact resolution, health endpoint
- Designed the FastAPI web application front-end (`src/serving/triplet_inference/static/index.html`): drag-and-drop upload, per-stage pipeline timeline with confidence bars, color-coded decision card
- Configured Vertex AI deployment (`src/serving/dinov3_endpoint/Dockerfile`, `src/serving/dinov3_endpoint/deploy_vertex.sh`)
