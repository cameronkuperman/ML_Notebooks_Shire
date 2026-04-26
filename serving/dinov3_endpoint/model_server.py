import base64
import io
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as tv_models
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from transformers import AutoModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_FILE = "dinov3_classifier_full.pt"
DEFAULT_JOBLIB_FILE = "best_model_logistic_regression (1).joblib"
HF_MODELS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
EMBED_DIMS = {"small": 384, "base": 768, "large": 1024, "huge": 1280}
LEAN_V2_FEATURE_NAMES = [
    "stable_track_count",
    "tracks_seen_all3",
    "one_frame_track_count",
    "score_weighted_overlap",
    "score_weighted_table_closeness",
    "stable_mean_displacement",
    "stable_max_displacement",
    "seated_score",
]


class PredictRequest(BaseModel):
    instances: list[Any]
    parameters: dict[str, Any] | None = None


class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, dropout=0.15, unfreeze_last_n=0):
        super().__init__()
        self.backbone = backbone
        self.unfreeze_last_n = unfreeze_last_n

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        if unfreeze_last_n > 0:
            if hasattr(backbone, "vit"):
                encoder_layers = backbone.vit.encoder.layer
            elif hasattr(backbone, "encoder"):
                encoder_layers = backbone.encoder.layer
            else:
                encoder_layers = getattr(backbone, "layers", [])

            for layer in encoder_layers[-unfreeze_last_n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        if self.unfreeze_last_n > 0:
            out = self.backbone(x)
        else:
            with torch.no_grad():
                out = self.backbone(x)

        if hasattr(out, "last_hidden_state"):
            cls_token = out.last_hidden_state[:, 0]
        elif isinstance(out, dict) and "x_norm_clstoken" in out:
            cls_token = out["x_norm_clstoken"]
        else:
            cls_token = out[0][:, 0] if isinstance(out, (list, tuple)) else out[:, 0]

        return self.head(cls_token)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.unfreeze_last_n > 0:
            if hasattr(self.backbone, "vit"):
                layers = self.backbone.vit.encoder.layer
            elif hasattr(self.backbone, "encoder"):
                layers = self.backbone.encoder.layer
            else:
                layers = getattr(self.backbone, "layers", [])

            for layer in layers[-self.unfreeze_last_n:]:
                layer.train(mode)
        return self


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    bucket_and_blob = uri[5:]
    bucket, _, blob = bucket_and_blob.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def download_gcs_file(uri: str, destination: Path) -> Path:
    bucket_name, blob_name = parse_gcs_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(destination)
    return destination


def find_model_artifact() -> Path:
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        if model_path.startswith("gs://"):
            return download_gcs_file(model_path, Path("/tmp/model") / Path(model_path).name)
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    model_file = os.getenv("MODEL_FILE", DEFAULT_MODEL_FILE)
    for candidate in (Path("/model") / model_file, Path.cwd() / model_file):
        if candidate.exists():
            return candidate

    storage_uri = os.getenv("AIP_STORAGE_URI", "")
    if storage_uri.startswith("gs://"):
        return download_gcs_file(f"{storage_uri.rstrip('/')}/{model_file}", Path("/tmp/model") / model_file)

    raise FileNotFoundError(
        f"Could not find {model_file}. Set MODEL_PATH, copy it to /model, or upload it as a Vertex artifact."
    )


def find_optional_artifact(env_path: str, env_file: str, default_file: str) -> Path | None:
    model_path = os.getenv(env_path)
    if model_path:
        if model_path.startswith("gs://"):
            return download_gcs_file(model_path, Path("/tmp/model") / Path(model_path).name)
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"{env_path} does not exist: {model_path}")

    model_file = os.getenv(env_file, default_file)
    for candidate in (Path("/model") / model_file, Path.cwd() / model_file):
        if candidate.exists():
            return candidate

    storage_uri = os.getenv("AIP_STORAGE_URI", "")
    if storage_uri.startswith("gs://"):
        try:
            return download_gcs_file(f"{storage_uri.rstrip('/')}/{model_file}", Path("/tmp/model") / model_file)
        except Exception:
            return None

    return None


def torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint dict with a 'model_state_dict' key.")
    return checkpoint


def normalize_label_maps(checkpoint: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    raw_id2label = checkpoint.get("id2label") or {0: "clean", 1: "dirty"}
    id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    raw_label2id = checkpoint.get("label2id") or {label: idx for idx, label in id2label.items()}
    label2id = {str(k): int(v) for k, v in raw_label2id.items()}
    return id2label, label2id


def build_resnet(model_name: str, num_classes: int):
    if model_name == "resnet18":
        model = tv_models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = tv_models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")

    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def infer_backbone_size(backbone_config: str | None, embed_dim: int | None) -> str:
    if backbone_config:
        for size, model_name in HF_MODELS.items():
            if model_name == backbone_config or size in backbone_config:
                return size
    if embed_dim:
        for size, candidate_dim in EMBED_DIMS.items():
            if embed_dim == candidate_dim:
                return size
    return os.getenv("BACKBONE_SIZE", "large")


def build_model(checkpoint: dict[str, Any]):
    id2label, _ = normalize_label_maps(checkpoint)
    model_name = checkpoint.get("model_name", "dinov3")
    num_classes = len(id2label)

    if model_name == "dinov3":
        embed_dim = checkpoint.get("embed_dim")
        backbone_config = checkpoint.get("backbone_config")
        backbone_size = infer_backbone_size(backbone_config, embed_dim)
        hf_model = backbone_config or HF_MODELS[backbone_size]
        backbone = AutoModel.from_pretrained(hf_model, trust_remote_code=True)
        model = DINOv3Classifier(
            backbone=backbone,
            embed_dim=int(embed_dim or EMBED_DIMS[backbone_size]),
            num_classes=num_classes,
            dropout=float(os.getenv("DROPOUT", "0.3")),
            unfreeze_last_n=int(checkpoint.get("unfreeze_last_n") or 0),
        )
    elif model_name in {"resnet18", "resnet50"}:
        model = build_resnet(model_name, num_classes)
    else:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_image_from_gcs(uri: str) -> Image.Image:
    bucket_name, blob_name = parse_gcs_uri(uri)
    payload = storage.Client().bucket(bucket_name).blob(blob_name).download_as_bytes()
    return Image.open(io.BytesIO(payload)).convert("RGB")


def load_image_from_instance(instance: Any) -> Image.Image:
    if isinstance(instance, str):
        if instance.startswith("gs://"):
            return load_image_from_gcs(instance)
        return Image.open(io.BytesIO(base64.b64decode(instance))).convert("RGB")

    if not isinstance(instance, dict):
        raise ValueError("Each instance must be a GCS URI, base64 string, or object.")

    gcs_uri = instance.get("gcs_uri") or instance.get("image_uri")
    if gcs_uri:
        if not str(gcs_uri).startswith("gs://"):
            raise ValueError("Only gs:// image URIs are supported for image_uri/gcs_uri.")
        return load_image_from_gcs(str(gcs_uri))

    payload = instance.get("image_b64") or instance.get("b64") or instance.get("base64")
    if payload is None and isinstance(instance.get("bytes"), dict):
        payload = instance["bytes"].get("b64")
    if payload is None:
        payload = instance.get("bytes")
    if payload is None:
        raise ValueError("Instance is missing one of: gcs_uri, image_uri, image_b64, b64, base64, bytes.")

    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _clip01(value):
    return float(np.clip(_safe_float(value), 0.0, 1.0))


def engineer_features_v2(perception: dict[str, Any], n_frames=3) -> np.ndarray:
    people = perception.get("people", []) or []
    if not isinstance(people, list):
        raise ValueError("perception.people must be a list.")

    tracks: dict[str, list[dict[str, Any]]] = {}
    weighted_overlap_numer = 0.0
    weighted_closeness_numer = 0.0
    score_denom = 0.0

    for det_idx, person in enumerate(people):
        if not isinstance(person, dict):
            continue
        track_id = person.get("track_id") or f"det_{det_idx}"
        score = _clip01(person.get("score", 0.0))
        overlap = _clip01(person.get("overlap_frac_of_person", 0.0))
        distance = _clip01(person.get("distance_norm", 1.0))
        closeness = 1.0 - distance

        tracks.setdefault(str(track_id), []).append(person)
        weighted_overlap_numer += score * overlap
        weighted_closeness_numer += score * closeness
        score_denom += score

    if not tracks:
        return np.zeros(len(LEAN_V2_FEATURE_NAMES), dtype=float)

    stable_track_count = 0
    tracks_seen_all3 = 0
    one_frame_track_count = 0
    stable_displacements = []
    track_seated_scores = []

    for detections in tracks.values():
        frames_seen = len({p.get("frame_index") for p in detections if p.get("frame_index") is not None})
        frames_seen = min(frames_seen, n_frames)

        scores = [_clip01(p.get("score", 0.0)) for p in detections]
        overlaps = [_clip01(p.get("overlap_frac_of_person", 0.0)) for p in detections]
        closenesses = [1.0 - _clip01(p.get("distance_norm", 1.0)) for p in detections]
        disps = [_safe_float(p.get("displacement_from_prev")) for p in detections if p.get("displacement_from_prev") is not None]

        if frames_seen >= 2:
            stable_track_count += 1
            stable_displacements.extend(disps)
        if frames_seen >= n_frames:
            tracks_seen_all3 += 1
        if frames_seen == 1:
            one_frame_track_count += 1

        mean_score = float(np.mean(scores)) if scores else 0.0
        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        mean_closeness = float(np.mean(closenesses)) if closenesses else 0.0
        mean_disp = float(np.mean(disps)) if disps else None
        persistence = frames_seen / float(n_frames)
        motion_stability = 1.0 - min(mean_disp / 0.08, 1.0) if frames_seen >= 2 and mean_disp is not None else 0.0
        spatial_evidence = (0.70 * mean_overlap) + (0.30 * mean_closeness)
        track_seated_scores.append(mean_score * persistence * motion_stability * spatial_evidence)

    score_weighted_overlap = weighted_overlap_numer / score_denom if score_denom > 0 else 0.0
    score_weighted_table_closeness = weighted_closeness_numer / score_denom if score_denom > 0 else 0.0
    stable_mean_displacement = float(np.mean(stable_displacements)) if stable_displacements else 0.0
    stable_max_displacement = float(np.max(stable_displacements)) if stable_displacements else 0.0
    seated_score = float(np.max(track_seated_scores)) if track_seated_scores else 0.0

    return np.array(
        [
            stable_track_count,
            tracks_seen_all3,
            one_frame_track_count,
            score_weighted_overlap,
            score_weighted_table_closeness,
            stable_mean_displacement,
            stable_max_displacement,
            seated_score,
        ],
        dtype=float,
    )


def call_preprocessor(instance: Any) -> Any:
    preprocessor_url = os.getenv("PREPROCESSOR_URL")
    if not preprocessor_url:
        return instance

    response = requests.post(preprocessor_url, json={"instance": instance}, timeout=float(os.getenv("PREPROCESSOR_TIMEOUT", "20")))
    response.raise_for_status()
    payload = response.json()
    if "features" in payload:
        return {"features": payload["features"]}
    if "perception" in payload:
        return {"perception": payload["perception"]}
    return payload


def features_from_instance(instance: Any) -> np.ndarray:
    instance = call_preprocessor(instance)

    if isinstance(instance, list):
        return np.asarray(instance, dtype=float)

    if not isinstance(instance, dict):
        raise ValueError("Occupancy instances must be feature lists, perception objects, or dicts containing features/perception.")

    if "features" in instance:
        return np.asarray(instance["features"], dtype=float)

    perception = instance.get("perception", instance)
    if isinstance(perception, dict):
        return engineer_features_v2(perception)

    raise ValueError("Could not derive features from occupancy instance.")


class ImagePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = find_model_artifact()
        checkpoint = torch_load_checkpoint(model_path, self.device)
        self.id2label, self.label2id = normalize_label_maps(checkpoint)
        self.threshold = float(checkpoint.get("best_threshold", 0.5))
        self.dirty_id = self.label2id.get("dirty", 1)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.model = build_model(checkpoint).to(self.device).eval()

    @torch.no_grad()
    def predict(self, instances: list[Any]) -> list[dict[str, Any]]:
        if not instances:
            return []

        images = [load_image_from_instance(instance) for instance in instances]
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1).cpu()

        predictions = []
        for row in probabilities:
            dirty_probability = float(row[self.dirty_id])
            predicted_id = self.dirty_id if dirty_probability >= self.threshold else self.label2id.get("clean", 0)
            predictions.append(
                {
                    "label": self.id2label[predicted_id],
                    "class_id": predicted_id,
                    "dirty_probability": dirty_probability,
                    "clean_probability": float(row[self.label2id.get("clean", 0)]),
                    "threshold": self.threshold,
                    "probabilities": {label: float(row[idx]) for idx, label in self.id2label.items()},
                }
            )
        return predictions


class JoblibPredictor:
    def __init__(self, model_path: Path):
        payload = joblib.load(model_path)
        if isinstance(payload, dict):
            self.model = payload.get("model") or payload.get("estimator") or payload.get("pipeline")
            self.threshold = float(payload.get("threshold", payload.get("best_threshold", 0.5)))
            self.feature_names = payload.get("feature_names", LEAN_V2_FEATURE_NAMES)
            self.labels = payload.get("labels", {0: "unoccupied", 1: "occupied"})
        else:
            self.model = payload
            self.threshold = float(os.getenv("JOBLIB_THRESHOLD", "0.5"))
            self.feature_names = LEAN_V2_FEATURE_NAMES
            self.labels = {0: "unoccupied", 1: "occupied"}

        if self.model is None:
            raise ValueError("Joblib payload dict must contain one of: model, estimator, pipeline.")
        self.labels = {int(k): str(v) for k, v in self.labels.items()}

    def probability(self, features: np.ndarray) -> float:
        feature_batch = features.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(feature_batch)[0][1])
        if hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(feature_batch)[0])
            return float(1.0 / (1.0 + np.exp(-score)))
        return float(self.model.predict(feature_batch)[0])

    def predict(self, instances: list[Any]) -> list[dict[str, Any]]:
        predictions = []
        for instance in instances:
            features = features_from_instance(instance)
            occupied_probability = self.probability(features)
            predicted_id = 1 if occupied_probability >= self.threshold else 0
            predictions.append(
                {
                    "label": self.labels.get(predicted_id, str(predicted_id)),
                    "class_id": predicted_id,
                    "occupied_probability": occupied_probability,
                    "unoccupied_probability": 1.0 - occupied_probability,
                    "threshold": self.threshold,
                    "features": {name: float(value) for name, value in zip(self.feature_names, features)},
                }
            )
        return predictions


class MultiModelService:
    def __init__(self):
        self.image_predictor = None
        self.occupancy_predictor = None

        pt_path = find_optional_artifact("PT_MODEL_PATH", "PT_MODEL_FILE", DEFAULT_MODEL_FILE)
        if pt_path is not None:
            os.environ.setdefault("MODEL_PATH", str(pt_path))
            self.image_predictor = ImagePredictor()

        joblib_path = find_optional_artifact("JOBLIB_MODEL_PATH", "JOBLIB_MODEL_FILE", DEFAULT_JOBLIB_FILE)
        if joblib_path is not None:
            self.occupancy_predictor = JoblibPredictor(joblib_path)

        if self.image_predictor is None and self.occupancy_predictor is None:
            raise FileNotFoundError(
                "No models found. Upload a .pt as PT_MODEL_FILE and/or a .joblib as JOBLIB_MODEL_FILE."
            )

    def loaded_models(self) -> list[str]:
        models = []
        if self.image_predictor is not None:
            models.append("image_clean_dirty")
        if self.occupancy_predictor is not None:
            models.append("occupancy")
        return models

    def predict(self, request: PredictRequest) -> list[dict[str, Any]]:
        task = (request.parameters or {}).get("task")
        if task is None and request.instances and isinstance(request.instances[0], dict):
            task = request.instances[0].get("task")
        task = task or os.getenv("DEFAULT_TASK", "occupancy")

        if task in {"image", "clean_dirty", "image_clean_dirty", "pt"}:
            if self.image_predictor is None:
                raise ValueError("The .pt image model is not loaded on this endpoint.")
            return self.image_predictor.predict(request.instances)

        if task in {"occupancy", "occupied", "joblib", "tabular"}:
            if self.occupancy_predictor is None:
                raise ValueError("The .joblib occupancy model is not loaded on this endpoint.")
            return self.occupancy_predictor.predict(request.instances)

        raise ValueError(f"Unknown task '{task}'. Use image_clean_dirty or occupancy.")


app = FastAPI(title="Multi-Model Restaurant Classifier")
service: MultiModelService | None = None


def get_service() -> MultiModelService:
    global service
    if service is None:
        service = MultiModelService()
    return service


@app.get("/health")
def health():
    model_service = get_service()
    device = str(model_service.image_predictor.device) if model_service.image_predictor is not None else "cpu"
    return {"status": "ok", "models": model_service.loaded_models(), "device": device}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        return {"predictions": get_service().predict(request)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
