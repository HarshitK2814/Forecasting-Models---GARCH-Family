# Reproducible modelling environment for the GARCH-EVT / Realized GARCH / Quantile
# Regression project. Closes the Executive Summary's "Reproducibility & Environment"
# item ("Consider a Dockerfile or container for the final environment").
#
# This pins to Researcher B's reference environment (requirements_B.txt, captured
# 2026-08-24 on Colab) because that is the run every committed table/figure was
# validated against by Datasets/10_SCRIPTS/50_reproducibility_audit.py. Researcher
# A's data-acquisition dependencies (Datasets/requirements.txt) are installed too,
# so the full pipeline — acquisition through evaluation — runs from one image.
#
# Build:  docker build -t garch-research .
# Run all tests:                 docker run --rm garch-research pytest tests/ -v
# Run the reproducibility audit:  docker run --rm garch-research python Datasets/10_SCRIPTS/50_reproducibility_audit.py
# Interactive shell:              docker run --rm -it garch-research bash
#
# Note: the base image below tracks Python 3.13's latest patch release, not the
# exact 3.13.15 patch pinned in requirements_B.txt (no official Docker tag exists
# for a single patch version). 50_reproducibility_audit.py's environment check
# will report the patch-version delta as informational; every package below is
# pinned exactly, which is what the audit's known scipy/numpy version-sensitivity
# note (see requirements_B.txt) actually depends on.
FROM python:3.13-slim

WORKDIR /app

# scipy/pandas/statsmodels wheels cover manylinux; build-essential is a fallback
# for any platform pip has to build from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Datasets/requirements.txt ./requirements-data.txt
COPY requirements_B.txt ./requirements-model.txt

# requirements-model.txt carries a descriptive "python==..." line (not a pip
# package) for the audit script to read; strip it before installing.
RUN pip install --no-cache-dir -r requirements-data.txt \
    && grep -v '^python==' requirements-model.txt | pip install --no-cache-dir -r /dev/stdin \
    && pip install --no-cache-dir pytest

COPY . .

CMD ["bash"]
