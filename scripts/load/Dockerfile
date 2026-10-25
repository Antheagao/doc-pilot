# The load generator: Locust and nothing else, in a small image of its own.
# It runs inside the doc-pilot-load compose network (docker-compose.load.yml,
# profile `loadgen`) and reaches the API at http://api:8000, so the numbers
# exclude Docker Desktop's host port forwarder.
#
# The scenario code is not copied in: compose mounts scripts/load at /load
# read-only and the corpus at /evals, so an edit to locustfile.py or
# workload.py needs no rebuild. The image holds only the pinned Locust.
FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Root, not a service user: it reads /load/out/tenants.json, which `load.py
# seed` writes mode 600 for the host user, and a bind mount keeps that owner's
# uid. The container holds no secret of its own and no key for any model.
WORKDIR /load
CMD ["sleep", "infinity"]
