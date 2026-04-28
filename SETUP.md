# Setup Instructions

## Requirements

- Python 3.11+
- pip
- A GPU is recommended for training (Colab T4/A100 works); CPU is fine for inference

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, PyTorch, Transformers (DINOv3), scikit-learn, XGBoost, joblib, Google GenAI SDK, and supporting libraries.

## 2. Obtain Model Artifacts

The trained model weights are not checked into the repo. Place both files in the `models/` directory:

| File | Description |
|------|-------------|
| `models/dinov3_classifier_full.pt` | Fine-tuned DINOv3-ViT-Large clean/dirty classifier |
| `models/best_model_logistic_regression.joblib` | Trained occupancy VotingClassifier ensemble |

To retrain from scratch, run the notebooks in order:
1. `notebooks/simple_yolo_classifier_occ_unocc.ipynb` — trains and saves the occupancy `.joblib`
2. `notebooks/dino_v3_rerun_(1) (1).ipynb` — trains and saves the DINOv3 `.pt` (requires GPU, run on Colab)

## 3. Configure Environment Variables

```bash
# Required for Gemini escalation (optional — pipeline degrades gracefully without it)
export GEMINI_API_KEY=your_google_ai_studio_key

# Override model paths if artifacts are not in the repo root
export PT_MODEL_PATH=/path/to/dinov3_classifier_full.pt
export JOBLIB_MODEL_PATH=/path/to/best_model_logistic_regression.joblib

# Optional: override confidence threshold (default 0.80)
export TRIPLET_CONFIDENCE_THRESHOLD=0.80

# Optional: override Gemini model (default gemini-2.5-flash)
export GEMINI_MODEL=gemini-2.5-flash
```

## 4. Run the Web Application

```bash
cd src/serving/triplet_inference
uvicorn app:app --reload --port 8000
```

Navigate to `http://localhost:8000`. The UI will display model load status in the top-right corner.

## 5. Prepare a Triplet Zip

The UI accepts `.zip` files containing exactly:
- `frame_0.jpg` — first frame
- `frame_1.jpg` — middle frame (used for clean/dirty inference)
- `frame_2.jpg` — last frame
- `perception.json` — track data from the perception system

Sample zips are in the `data/examples/` directory.

## Colab Training (GPU)

Open `notebooks/dino_v3_rerun_(1) (1).ipynb` in Google Colab with a T4 or A100 runtime. The notebook installs its own dependencies via `!pip install` cells. After training, download `dinov3_classifier_full.pt` and place it in the `models/` directory.
