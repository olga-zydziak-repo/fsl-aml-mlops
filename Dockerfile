# syntax=docker/dockerfile:1
#
# Training image for the `fsl` package.
#
# Built from a plain, well-known Python base (not a Google "prebuilt" image) —
# deliberately, after two failed attempts to guess a valid Vertex AI prebuilt
# PyTorch training URI. `container_uri` in Vertex CustomJob / KFP container
# components accepts ANY image; it does not have to come from Google's
# `vertex-ai/training/*` family. Building from `python:3.10-slim` means every
# layer is under our control and nothing depends on Google's naming scheme.
#
# Used in BOTH loops:
#   - inner loop: CustomJob.from_local_script(container_uri=<this image>, ...)
#   - outer loop: KFP container_component pointing at <this image>

# ---------------------------------------------------------------------------
# Base image: official, minimal, well-known. "slim" = Debian without the
# extras (docs, compilers we don't need) -- smaller image, faster pulls.
# Pinned to 3.10 to match the venv you've been using on Workbench
# (matching versions avoids "works on Workbench, breaks in the container").
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# ---------------------------------------------------------------------------
# Working directory inside the container. Everything from here on (COPY, RUN)
# happens relative to /app unless stated otherwise.
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# System-level build dependencies. `build-essential` provides a C compiler,
# which some Python packages need to compile native extensions during pip
# install. `--no-install-recommends` skips optional extras Debian would
# otherwise pull in, keeping the image smaller. The final `rm -rf` clears
# apt's package cache -- it's not needed after install and would otherwise
# sit in the image forever, inflating its size.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Copy ONLY the dependency manifest first, before the rest of the code.
# This is a deliberate ordering trick for Docker's layer cache: Docker caches
# each instruction as a layer, and reuses a cached layer if the preceding
# files haven't changed. If we copied all the code first, ANY code edit would
# invalidate the cache for the (slow) pip install step below, forcing a full
# reinstall of torch etc. every time. Copying pyproject.toml alone means pip
# install is only re-run when dependencies actually change.
# ---------------------------------------------------------------------------
COPY pyproject.toml .

# ---------------------------------------------------------------------------
# A minimal src/ layout stub is needed here because pyproject.toml's
# [tool.setuptools.packages.find] expects `src/` to exist when pip resolves
# the package metadata, even before the real code is copied. This keeps the
# dependency-install layer cacheable while still letting pip see the project
# structure. The REAL package code is copied and overwrites this in the next
# step.
# ---------------------------------------------------------------------------
RUN mkdir -p src/fsl && touch src/fsl/__init__.py

# ---------------------------------------------------------------------------
# Install dependencies declared in pyproject.toml (torch, torchvision,
# learn2learn, google-cloud-storage, google-cloud-aiplatform, ...).
# --no-cache-dir keeps pip from storing its download cache in the image --
# we only ever build this image once per dependency change, so the cache
# buys nothing and only adds size.
# This is the SLOW step (torch is large) -- and thanks to the layer-cache
# trick above, it only re-runs when pyproject.toml changes, not on every
# code edit.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# NOW copy the real package code. This layer is cheap to rebuild (just a file
# copy, no installation), so it's fine that it changes on every code edit --
# it sits AFTER the expensive pip install layer, so editing fsl.training does
# NOT force torch to reinstall.
# ---------------------------------------------------------------------------
COPY src/ src/
COPY scripts/ scripts/

# ---------------------------------------------------------------------------
# Re-run the editable install so the package metadata points at the real
# code that just landed (the stub from earlier is now overwritten).
# Cheap: dependencies are already satisfied, this just re-links the package.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir -e . --no-deps

# ---------------------------------------------------------------------------
# Default command when the container runs with no arguments. Vertex overrides
# this with its own `args=[...]` when submitting a job, but having a sane
# default means `docker run <image>` works standalone for local debugging too.
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "scripts/train.py"]