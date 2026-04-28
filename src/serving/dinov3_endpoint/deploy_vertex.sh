#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID, for example: export PROJECT_ID=my-gcp-project}"
: "${BUCKET:?Set BUCKET, for example: export BUCKET=my-model-bucket}"

REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-vertex-models}"
IMAGE_NAME="${IMAGE_NAME:-dinov3-clean-dirty}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-dinov3-clean-dirty}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-dinov3-clean-dirty-endpoint}"
PT_MODEL_FILE="${PT_MODEL_FILE:-${MODEL_FILE:-dirty_clean_classifier_full.pt}}"
JOBLIB_MODEL_FILE="${JOBLIB_MODEL_FILE:-occupancy_model.joblib}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MAX_REPLICAS="${MAX_REPLICAS:-1}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-dinov3-clean-dirty}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
ARTIFACT_URI="gs://${BUCKET}/${ARTIFACT_PREFIX}"
CONTAINER_ENV_VARS="PT_MODEL_FILE=${PT_MODEL_FILE},JOBLIB_MODEL_FILE=${JOBLIB_MODEL_FILE},DEFAULT_TASK=${DEFAULT_TASK:-occupancy}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  CONTAINER_ENV_VARS+=",HF_TOKEN=${HF_TOKEN}"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  CONTAINER_ENV_VARS+=",HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"
fi

if [[ -n "${PREPROCESSOR_URL:-}" ]]; then
  CONTAINER_ENV_VARS+=",PREPROCESSOR_URL=${PREPROCESSOR_URL}"
fi

if [[ ! -f "${PT_MODEL_FILE}" && ! -f "${JOBLIB_MODEL_FILE}" ]]; then
  echo "Missing model files. Copy ${PT_MODEL_FILE} and/or ${JOBLIB_MODEL_FILE} into this directory." >&2
  exit 1
fi

gcloud config set project "${PROJECT_ID}"
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "${REGION}" "gs://${BUCKET}"
fi

if ! gcloud artifacts repositories describe "${REPOSITORY}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Vertex AI model serving images"
fi

if [[ -f "${PT_MODEL_FILE}" ]]; then
  gsutil cp "${PT_MODEL_FILE}" "${ARTIFACT_URI}/${PT_MODEL_FILE}"
fi

if [[ -f "${JOBLIB_MODEL_FILE}" ]]; then
  gsutil cp "${JOBLIB_MODEL_FILE}" "${ARTIFACT_URI}/${JOBLIB_MODEL_FILE}"
fi

gcloud builds submit . --tag "${IMAGE_URI}"

gcloud ai models upload \
  --region="${REGION}" \
  --display-name="${MODEL_DISPLAY_NAME}" \
  --artifact-uri="${ARTIFACT_URI}" \
  --container-image-uri="${IMAGE_URI}" \
  --container-health-route="/health" \
  --container-predict-route="/predict" \
  --container-ports=8080 \
  --container-env-vars="${CONTAINER_ENV_VARS}"

MODEL_ID="$(gcloud ai models list \
  --region="${REGION}" \
  --filter="displayName=${MODEL_DISPLAY_NAME}" \
  --sort-by="~createTime" \
  --limit=1 \
  --format="value(name)")"

ENDPOINT_NAME="$(gcloud ai endpoints create \
  --region="${REGION}" \
  --display-name="${ENDPOINT_DISPLAY_NAME}" \
  --format="value(name)")"
ENDPOINT_ID="${ENDPOINT_NAME##*/}"

DEPLOY_ARGS=(
  ai endpoints deploy-model "${ENDPOINT_ID}"
  --region="${REGION}"
  --model="${MODEL_ID}"
  --display-name="${MODEL_DISPLAY_NAME}"
  --machine-type="${MACHINE_TYPE}"
  --min-replica-count="${MIN_REPLICAS}"
  --max-replica-count="${MAX_REPLICAS}"
  --traffic-split="0=100"
)

if [[ -n "${ACCELERATOR_TYPE:-}" ]]; then
  DEPLOY_ARGS+=(--accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT:-1}")
fi

gcloud "${DEPLOY_ARGS[@]}"

echo "Endpoint ID: ${ENDPOINT_ID}"
echo "Predict:"
echo "gcloud ai endpoints predict ${ENDPOINT_ID} --region=${REGION} --json-request=sample_request.json"
