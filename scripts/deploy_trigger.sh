#!/usr/bin/env bash
# Trigger rundy active learning: wrzucenie kolejka.csv do analyst/oznaczone/<partia>/
# uruchamia pipeline rundy (Eventarc -> funkcja Cloud Run -> Vertex AI PipelineJob).
#
# Uruchom w Cloud Shell (konsola GCP, ikona >_ w prawym gornym rogu), jako wlasciciel projektu:
#   bash deploy_trigger.sh
# Bez kluczy: funkcja dziala na koncie serwisowym projektu (domyslne konto Compute Engine).
# Drugie uruchomienie aktualizuje funkcje (nie tworzy duplikatu).
set -euo pipefail

PROJECT="dark-data-discovery"
REGION="us-central1"
BUCKET="dark-data-discovery-fsl-data"
FN="fsl-al-round-trigger"

gcloud config set project "$PROJECT" >/dev/null
NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
RUNTIME_SA="${NUM}-compute@developer.gserviceaccount.com"
GCS_AGENT="service-${NUM}@gs-project-accounts.iam.gserviceaccount.com"

echo "== 1/5 wlaczam API (Cloud Run functions, Eventarc, Cloud Build, Pub/Sub)"
gcloud services enable cloudfunctions.googleapis.com run.googleapis.com eventarc.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com pubsub.googleapis.com aiplatform.googleapis.com

echo "== 2/5 uprawnienia (bez kluczy, tylko role kont serwisowych projektu)"
# usluga GCS publikuje zdarzenia o nowych plikach
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:${GCS_AGENT}" \
  --role=roles/pubsub.publisher --condition=None >/dev/null
# funkcja: odbiera zdarzenia, czyta szablon i stan partii, tworzy przebieg Vertex jako to samo konto
for ROLE in roles/eventarc.eventReceiver roles/run.invoker roles/aiplatform.user roles/storage.objectViewer; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:${RUNTIME_SA}" \
    --role="$ROLE" --condition=None >/dev/null
done
# przebieg Vertex dziala na tym samym koncie: funkcja moze dzialac "jako" ono, i tylko jako ono
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/iam.serviceAccountUser >/dev/null

echo "== 3/5 szablon pipeline'u rundy w GCS"
if ! gcloud storage ls "gs://${BUCKET}/pipelines/al_round_pipeline.yaml" >/dev/null 2>&1; then
  echo "BRAK gs://${BUCKET}/pipelines/al_round_pipeline.yaml"
  echo "Uruchom komorke 4 w al_runda.ipynb (kompiluje pipeline i zapisuje go w GCS), potem ten skrypt ponownie."
  exit 1
fi

echo "== 4/5 kod funkcji"
SRC=$(mktemp -d)
cat > "$SRC/main.py" <<'PY'
"""Cloud Run function: a filled kolejka.csv in analyst/oznaczone/ starts an active learning round.

Event: google.cloud.storage.object.v1.finalized on the project bucket (Eventarc).
Only analyst/oznaczone/<batch>/kolejka.csv starts a round; everything else in the bucket
(including what the round itself writes: wynik.json, reports, the next batch) is ignored.

Duplicates: Eventarc delivers at least once, so the same upload can arrive twice. The pipeline
job id is built from the batch and the object generation, and Vertex refuses a second job with
the same id. A re-upload of the file (a corrected answer) has a new generation and runs again.
No keys: the function runs as a service account of the project.
"""
import os
import re

import functions_framework
from google.api_core import exceptions as gexc
from google.cloud import aiplatform, storage

PROJECT = os.environ["PROJECT"]
REGION = os.environ.get("REGION", "us-central1")
BUCKET = os.environ["BUCKET"]
TEMPLATE = os.environ.get("TEMPLATE", f"gs://{BUCKET}/pipelines/al_round_pipeline.yaml")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", f"gs://{BUCKET}/pipeline-root")
FILLED = re.compile(r"^analyst/oznaczone/(?P<batch>[^\s]+)/kolejka\.csv$")


def job_id_for(batch: str, generation: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", batch.lower()).strip("-")
    tail = f"-{generation}" if generation else ""
    return f"al-round-{slug}"[: 128 - len(tail)].rstrip("-") + tail


@functions_framework.cloud_event
def on_labels(event):
    data = event.data or {}
    bucket, name, generation = data.get("bucket", ""), data.get("name", ""), str(data.get("generation", ""))
    m = FILLED.match(name)
    if bucket != BUCKET or not m:
        print(f"skip {bucket}/{name}")
        return
    batch = m.group("batch")
    if not storage.Client(project=PROJECT).bucket(BUCKET).blob(f"analyst/do_oznaczenia/{batch}/stan.json").exists():
        print(f"skip {name}: no analyst/do_oznaczenia/{batch}/stan.json (batch from an older image or a wrong path)")
        return
    job_id = job_id_for(batch, generation)
    aiplatform.init(project=PROJECT, location=REGION)
    job = aiplatform.PipelineJob(
        display_name="fsl-al-round",
        template_path=TEMPLATE,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={"batch": batch},
        job_id=job_id,
        enable_caching=False,
    )
    try:
        job.submit(service_account=os.environ.get("PIPELINE_SA") or None)
    except gexc.AlreadyExists:
        print(f"duplicate event for {name} (generation {generation}): {job_id} already submitted")
        return
    print(f"round submitted: batch={batch} job={job_id}")
PY
cat > "$SRC/requirements.txt" <<'REQ'
functions-framework==3.*
google-cloud-aiplatform[pipelines]>=1.60
pyyaml>=6
google-cloud-storage>=2.10
REQ

echo "== 5/5 wdrozenie funkcji ${FN} (pierwszy raz kilka minut)"
LOC=$(gcloud storage buckets describe "gs://${BUCKET}" --format='value(location)' | tr '[:upper:]' '[:lower:]')
echo "   bucket w lokalizacji: ${LOC}"
for TRY in 1 2 3; do
  if gcloud functions deploy "$FN" --gen2 --region="$REGION" --runtime=python312 \
      --source="$SRC" --entry-point=on_labels \
      --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
      --trigger-event-filters="bucket=${BUCKET}" \
      --trigger-location="$LOC" \
      --service-account="$RUNTIME_SA" --trigger-service-account="$RUNTIME_SA" \
      --set-env-vars="PROJECT=${PROJECT},REGION=${REGION},BUCKET=${BUCKET}" \
      --memory=512Mi --timeout=120s --max-instances=3 --quiet; then
    break
  fi
  # swiezo nadane uprawnienia Eventarc potrafia sie propagowac kilka minut
  [ "$TRY" -lt 3 ] && { echo "   nie wyszlo (czesto: uprawnienia jeszcze sie propaguja), ponawiam za 90 s"; sleep 90; } \
                   || { echo "Wdrozenie nie wyszlo 3 razy. Wklej Claude'owi ostatnie linie bledu."; exit 1; }
done

echo
echo "Gotowe. Od teraz plik analyst/oznaczone/<partia>/kolejka.csv sam uruchamia runde."
echo "Logi funkcji (co zlapala, co uruchomila):"
echo "  gcloud functions logs read ${FN} --gen2 --region=${REGION} --limit=20"
