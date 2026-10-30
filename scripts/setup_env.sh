#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# setup_env.sh — Create a DURABLE Python environment on a Vertex AI Workbench
# instance and register it as a Jupyter kernel visible in JupyterLab.
#
# Why this layout:
#   * The venv lives on the PERSISTENT home disk (/home/jupyter/envs), NOT the
#     boot disk — so it survives Workbench environment upgrades (the boot disk
#     is wiped on upgrade).
#   * The kernel is registered with `--user`, so the kernelspec also lands on
#     the persistent home disk (~/.local/share/jupyter/kernels) and survives.
#   * Plain venv (not conda) because this project is pure-pip (pyproject.toml);
#     the kernel still shows up in JupyterLab via ipykernel.
#
# Run this in the JupyterLab Terminal:  File > New > Terminal
#
# Usage:
#   ./setup_env.sh                      # env name 'fsl', project = current dir
#   ./setup_env.sh fsl /home/jupyter/vertex-fewshot-mlops
set -euo pipefail

ENV_NAME="${1:-fsl}"
PROJECT_DIR="${2:-$PWD}"
ENV_HOME="/home/jupyter/envs/${ENV_NAME}"

# --- preconditions ---------------------------------------------------------
if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  echo "ERROR: no pyproject.toml found in ${PROJECT_DIR}" >&2
  echo "Pass the project directory explicitly:" >&2
  echo "  ./setup_env.sh ${ENV_NAME} /home/jupyter/<your-project>" >&2
  exit 1
fi

# --- create (or reuse) the venv on the persistent disk ---------------------
if [[ -d "${ENV_HOME}" ]]; then
  echo ">> Env already exists at ${ENV_HOME} — reusing and re-installing."
else
  echo ">> Creating venv at ${ENV_HOME} (persistent home disk, survives upgrades)"
  python -m venv "${ENV_HOME}"
fi

# shellcheck disable=SC1091
source "${ENV_HOME}/bin/activate"

echo ">> Upgrading pip"
python -m pip install --upgrade pip

echo ">> Installing project (editable) + dev extras from ${PROJECT_DIR}"
pip install -e "${PROJECT_DIR}[dev]"

echo ">> Installing ipykernel"
pip install ipykernel

# --- register the kernel (persistent kernelspec via --user) ----------------
echo ">> Registering Jupyter kernel '${ENV_NAME}'"
python -m ipykernel install --user \
  --name "${ENV_NAME}" \
  --display-name "Python (${ENV_NAME})"

echo
echo "Done."
echo "  1. Reload the JupyterLab browser tab (hard refresh)."
echo "  2. 'Python (${ENV_NAME})' now appears in the Launcher and kernel picker."
echo
echo "Sanity check — open a notebook with that kernel and run:"
echo "    import sys, fsl"
echo "    print(sys.executable)   # -> ${ENV_HOME}/bin/python"
echo "    print(fsl.__version__)"
