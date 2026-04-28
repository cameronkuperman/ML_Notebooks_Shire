import base64
import io
import json
import mimetypes
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

MODEL_SERVER_DIR = Path(__file__).resolve().parents[1] / "dinov3_endpoint"
sys.path.insert(0, str(MODEL_SERVER_DIR))
ROOT_DIR = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIDENCE_THRESHOLD = float(os.getenv("TRIPLET_CONFIDENCE_THRESHOLD", "0.80"))
REQUIRED_FRAMES = ("frame_0.jpg", "frame_1.jpg", "frame_2.jpg")

os.environ.setdefault("PT_MODEL_PATH", str(ROOT_DIR / "models" / "dinov3_classifier_full.pt"))
os.environ.setdefault("JOBLIB_MODEL_PATH", str(ROOT_DIR / "models" / "best_model_logistic_regression (1).joblib"))

from model_server import MultiModelService  # noqa: E402

app = FastAPI(title="Triplet Inference UI")
service: MultiModelService | None = None


def get_service() -> MultiModelService:
    global service
    if service is None:
        try:
            service = MultiModelService()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model load failed: {exc}") from exc
    return service


def zip_member_basename(name: str) -> str:
    return Path(name).name.lower()


def read_triplet_zip(payload: bytes) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Upload must be a valid .zip file.") from exc

    members = {zip_member_basename(name): name for name in archive.namelist() if not name.endswith("/")}
    missing = [name for name in (*REQUIRED_FRAMES, "perception.json") if name not in members]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Triplet zip is missing required file(s): "
                + ", ".join(missing)
                + ". v1 expects preprocessed triplets, not raw video."
            ),
        )

    frames = {}
    for frame_name in REQUIRED_FRAMES:
        frame_bytes = archive.read(members[frame_name])
        frames[frame_name] = {
            "name": frame_name,
            "bytes": frame_bytes,
            "data_url": bytes_to_data_url(frame_bytes, frame_name),
        }

    try:
        perception = json.loads(archive.read(members["perception.json"]).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse perception.json: {exc}") from exc

    perception_for_features = normalize_perception(perception)
    return {"frames": frames, "perception": perception, "perception_for_features": perception_for_features}


def normalize_perception(perception: Any) -> dict[str, Any]:
    if isinstance(perception, list):
        if not perception:
            return {"people": []}
        first = perception[0]
        if isinstance(first, dict):
            return first
    if isinstance(perception, dict):
        if "people" in perception:
            return perception
        if isinstance(perception.get("perception"), dict):
            return perception["perception"]
    raise HTTPException(status_code=400, detail="perception.json must be an object or a non-empty list of objects.")


def bytes_to_data_url(payload: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def image_instance_from_bytes(payload: bytes) -> dict[str, str]:
    return {"image_b64": base64.b64encode(payload).decode("ascii")}


def confidence_for_occupancy(prediction: dict[str, Any]) -> float:
    if prediction["label"] == "occupied":
        return float(prediction.get("occupied_probability", 0.0))
    return float(prediction.get("unoccupied_probability", 0.0))


def confidence_for_clean_dirty(prediction: dict[str, Any]) -> float:
    probabilities = prediction.get("probabilities") or {}
    if probabilities:
        return float(max(probabilities.values()))
    if prediction["label"] == "dirty":
        return float(prediction.get("dirty_probability", 0.0))
    return float(prediction.get("clean_probability", 0.0))


def stage(
    name: str,
    label: str,
    confidence: float | None,
    status: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": name,
        "label": label,
        "confidence": confidence,
        "accepted": status in {"accepted", "final"},
        "status": status,
        "reason": reason,
        "metadata": metadata or {},
    }


def call_gemini(
    frames: dict[str, dict[str, Any]],
    perception: Any,
    stages: list[dict[str, Any]],
    history: str | None,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "reason": "GEMINI_API_KEY is not set, so Gemini escalation was skipped.",
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        parts: list[Any] = [
            "Classify this restaurant table triplet as occupied, clean, or dirty. "
            "Return JSON with label, confidence, and reason only.",
            f"Prior state history: {history or 'none'}",
            f"Perception JSON: {json.dumps(perception)[:6000]}",
            f"Prior model stages: {json.dumps(stages)[:6000]}",
        ]
        for frame_name in REQUIRED_FRAMES:
            frame = frames[frame_name]
            parts.append(types.Part.from_bytes(data=frame["bytes"], mime_type="image/jpeg"))

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        parsed = json.loads(response.text or "{}")
        label = str(parsed.get("label", "uncertain")).lower()
        if label not in {"occupied", "clean", "dirty"}:
            label = "uncertain"
        return {
            "label": label,
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", "Gemini returned no reason.")),
        }
    except Exception as exc:
        return {"label": "uncertain", "confidence": 0.0, "reason": f"Gemini escalation failed: {exc}"}


def choose_best_available(stages: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [item for item in stages if item.get("confidence") is not None and item["label"] != "perception.json"]
    if not candidates:
        return {"label": "uncertain", "confidence": 0.0, "source": "Triplet Parse"}
    best = max(candidates, key=lambda item: float(item.get("confidence") or 0.0))
    return {"label": best["label"], "confidence": best["confidence"], "source": best["stage"]}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/health")
def health():
    pt_path = Path(os.environ["PT_MODEL_PATH"])
    joblib_path = Path(os.environ["JOBLIB_MODEL_PATH"])
    models = []
    if pt_path.exists():
        models.append("image_clean_dirty")
    if joblib_path.exists():
        models.append("occupancy")
    return {
        "status": "ok",
        "models": models,
        "threshold": CONFIDENCE_THRESHOLD,
        "artifacts": {
            "pt_model": str(pt_path),
            "joblib_model": str(joblib_path),
        },
    }


@app.post("/api/predict-triplet")
async def predict_triplet(file: UploadFile = File(...), history: str | None = Form(default=None)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip containing frame_0/1/2.jpg and perception.json.")

    triplet = read_triplet_zip(await file.read())
    model_service = get_service()
    stages: list[dict[str, Any]] = [
        stage(
            "Triplet Parse",
            "perception.json",
            1.0,
            "accepted",
            "Found frame_0.jpg, frame_1.jpg, frame_2.jpg, and perception.json.",
            {"required_frames": list(REQUIRED_FRAMES)},
        )
    ]

    if model_service.occupancy_predictor is None:
        raise HTTPException(status_code=503, detail="The local .joblib occupancy model is not loaded.")

    occupancy = model_service.occupancy_predictor.predict([{"perception": triplet["perception_for_features"]}])[0]
    occupancy_confidence = confidence_for_occupancy(occupancy)
    occupancy_accepted = occupancy_confidence >= CONFIDENCE_THRESHOLD
    stages.append(
        stage(
            "Occupancy LR",
            occupancy["label"],
            occupancy_confidence,
            "accepted" if occupancy_accepted else "fallback",
            "Local joblib model from perception.json features.",
            occupancy,
        )
    )

    final = None
    if occupancy_accepted and occupancy["label"] == "occupied":
        stages.append(stage("Clean/Dirty PT", "skipped", None, "skipped", "Skipped because occupancy was confident."))
        final = {"label": "occupied", "confidence": occupancy_confidence, "source": "Occupancy LR"}
    elif occupancy_accepted and occupancy["label"] == "unoccupied":
        if model_service.image_predictor is None:
            raise HTTPException(status_code=503, detail="The local .pt clean/dirty model is not loaded.")
        clean_dirty = model_service.image_predictor.predict([image_instance_from_bytes(triplet["frames"]["frame_1.jpg"]["bytes"])])[0]
        clean_dirty_confidence = confidence_for_clean_dirty(clean_dirty)
        clean_dirty_accepted = clean_dirty_confidence >= CONFIDENCE_THRESHOLD
        stages.append(
            stage(
                "Clean/Dirty PT",
                clean_dirty["label"],
                clean_dirty_confidence,
                "accepted" if clean_dirty_accepted else "fallback",
                "Local .pt model on middle crop frame_1.jpg.",
                clean_dirty,
            )
        )
        if clean_dirty_accepted:
            final = {"label": clean_dirty["label"], "confidence": clean_dirty_confidence, "source": "Clean/Dirty PT"}

    if final is None:
        gemini = call_gemini(triplet["frames"], triplet["perception"], stages, history)
        gemini_usable = gemini["label"] in {"occupied", "clean", "dirty"} and gemini["confidence"] > 0
        stages.append(
            stage(
                "Gemini Escalation",
                gemini["label"],
                gemini["confidence"],
                "final" if gemini_usable else "fallback",
                gemini["reason"],
                {"model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")},
            )
        )
        if gemini_usable:
            final = {"label": gemini["label"], "confidence": gemini["confidence"], "source": "Gemini Escalation"}
        else:
            final = choose_best_available(stages)
            final["source"] = f"{final['source']} (Gemini unavailable)"

    return {
        "final": final,
        "stages": stages,
        "threshold": CONFIDENCE_THRESHOLD,
        "frames": {name: {"name": frame["name"], "data_url": frame["data_url"]} for name, frame in triplet["frames"].items()},
        "perception": triplet["perception"],
    }
