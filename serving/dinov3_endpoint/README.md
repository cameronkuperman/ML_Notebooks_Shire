# Multi-Model Vertex AI Endpoint

This folder wraps the notebook checkpoints as a Vertex AI custom prediction container.

It can serve:

- `.pt` DINOv3 image model: clean vs dirty
- `.joblib` sklearn/XGBoost-style model: occupied vs unoccupied from `perception.json` features

## 1. Put the model files here

From the notebooks, download or copy the files into this folder:

```bash
dinov3_classifier_full.pt
best_model_logistic_regression (1).joblib
```

If your filenames are different, set:

```bash
export PT_MODEL_FILE="dinov3_classifier_full.pt"
export JOBLIB_MODEL_FILE="your_model.joblib"
```

## Local triplet UI

This repo also includes a local-first FastAPI UI that uses the artifacts already in the workspace:

```bash
uvicorn serving.triplet_inference.app:app --reload
```

Open `http://127.0.0.1:8000`, upload a zip containing `frame_0.jpg`, `frame_1.jpg`, `frame_2.jpg`, and `perception.json`, and the app will run:

```text
perception.json -> local joblib occupancy -> local frame_1 .pt clean/dirty -> Gemini if low confidence
```

Gemini escalation is optional. Set `GEMINI_API_KEY` to enable it; without the key, the UI shows that escalation was skipped and returns the best local model result.

If your `.joblib` is not saved yet, save it from the notebook like this:

```python
import joblib

joblib.dump(
    {
        "model": best_model,
        "threshold": selected_threshold,
        "feature_names": LEAN_V2_FEATURE_NAMES,
        "labels": {0: "unoccupied", 1: "occupied"},
    },
    "occupancy_model.joblib",
)
```

## 2. Deploy to Vertex AI

Authenticate first:

```bash
gcloud auth login
gcloud auth application-default login
```

Then run:

```bash
cd serving/dinov3_endpoint
export PROJECT_ID="YOUR_GCP_PROJECT"
export BUCKET="YOUR_GCS_BUCKET"
export REGION="us-central1"
bash deploy_vertex.sh
```

For a GPU endpoint, add:

```bash
export MACHINE_TYPE="n1-standard-4"
export ACCELERATOR_TYPE="nvidia-tesla-t4"
export ACCELERATOR_COUNT="1"
```

If Hugging Face requires a token for the DINOv3 backbone in your environment, pass it during deploy. The deploy script forwards it into the Vertex container:

For the simple env-var path:

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

## 3. Preprocessing repo options

Best option: copy the preprocessing feature code into this container. This repo already includes the notebook's `engineer_features_v2()` logic, so the occupancy model can accept raw `perception.json` directly.

Alternative: deploy your preprocessing repo as its own Cloud Run service and point this model endpoint at it:

```bash
export PREPROCESSOR_URL="https://YOUR-PREPROCESSOR-SERVICE.run.app/preprocess"
bash deploy_vertex.sh
```

That service should accept:

```json
{"instance": "...anything you send to Vertex..."}
```

and return either:

```json
{"features": [0, 1, 0, 0.2, 0.8, 0.01, 0.04, 0.3]}
```

or:

```json
{"perception": {"people": []}}
```

## 4. Call the endpoint

The prediction route expects Vertex JSON with `instances` and optional `parameters`.

Occupied/unoccupied from `perception.json`:

```json
{
  "parameters": {"task": "occupancy"},
  "instances": [
    {
      "perception": {
        "people": [
          {
            "frame_index": 0,
            "track_id": "person_1",
            "overlap_frac_of_person": 0.42,
            "distance_norm": 0.15,
            "displacement_from_prev": null,
            "score": 0.91
          }
        ]
      }
    }
  ]
}
```

Or send features directly:

```json
{
  "parameters": {"task": "occupancy"},
  "instances": [
    {"features": [0, 0, 1, 0.0, 0.93, 0.0, 0.0, 0.0]}
  ]
}
```

Clean/dirty from image:

```json
{
  "parameters": {"task": "image_clean_dirty"},
  "instances": [
    {"gcs_uri": "gs://YOUR_BUCKET/path/to/frame.jpg"}
  ]
}
```

Then:

```bash
gcloud ai endpoints predict ENDPOINT_ID --region=us-central1 --json-request=sample_request.json
```

Base64 also works:

```json
{
  "instances": [
    {"image_b64": "BASE64_ENCODED_IMAGE"}
  ]
}
```

Responses look like:

```json
{
  "predictions": [
    {
      "label": "occupied",
      "class_id": 1,
      "occupied_probability": 0.91,
      "unoccupied_probability": 0.09,
      "threshold": 0.52,
      "features": {"stable_track_count": 1}
    }
  ]
}
```
