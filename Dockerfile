# PETPrep Docker Container Image distribution
#
# MIT License
#
# Copyright (c) The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

ARG BASE_IMAGE=ghcr.io/nipreps/petprep-base:20250912

#
# Build wheel
#
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS src
RUN apk add git
COPY . /src
RUN uv build --wheel /src

# Micromamba
FROM mambaorg/micromamba:2.3.2 AS micromamba

WORKDIR /

ENV MAMBA_ROOT_PREFIX="/opt/conda"
COPY env.yml /tmp/env.yml
COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN set -eux; \
    micromamba config set remote_max_retries 10; \
    micromamba config set remote_backoff_factor 3; \
    micromamba config set remote_connect_timeout_secs 30; \
    micromamba config set remote_read_timeout_secs 120; \
    micromamba config set fetch_threads 1; \
    attempts=8; \
    for attempt in $(seq 1 "$attempts"); do \
        if micromamba create -y -f /tmp/env.yml; then \
            break; \
        fi; \
        if [ "$attempt" -eq "$attempts" ]; then \
            echo "micromamba create failed after ${attempts} attempts" >&2; \
            exit 1; \
        fi; \
        sleep_time=$((attempt * 15)); \
        echo "micromamba create failed (attempt ${attempt}/${attempts}), retrying in ${sleep_time}s" >&2; \
        sleep "$sleep_time"; \
    done; \
    micromamba clean -y -a

# UV_USE_IO_URING for apparent race-condition (https://github.com/nodejs/node/issues/48444)
# Check if this is still necessary when updating the base image.
ENV PATH="/opt/conda/envs/petprep/bin:$PATH" \
    UV_USE_IO_URING=0
RUN npm install -g svgo@^3.2.0 bids-validator@1.14.10 && \
    rm -r ~/.npm

#
# Main stage
#
FROM ${BASE_IMAGE} AS petprep

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users petprep
WORKDIR /home/petprep

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/petprep /opt/conda/envs/petprep

ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba shell init -s bash && \
    echo "micromamba activate petprep" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/petprep/bin:$PATH"

# Precaching atlases
COPY scripts/fetch_templates.py fetch_templates.py
RUN python fetch_templates.py && \
    rm fetch_templates.py && \
    find $HOME/.cache/templateflow -type d -exec chmod go=u {} + && \
    find $HOME/.cache/templateflow -type f -exec chmod go=u {} +

# FSL environment
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLDIR="/opt/conda/envs/petprep" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# fs_install_mcr requires unzip to extract the downloaded MCR archive
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Installing PETPREP
COPY --from=src /src/dist/*.whl .
RUN pip install --no-cache-dir $( ls *.whl )[container,test]

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

# For detecting the container
ENV IS_DOCKER_8395080871=1

RUN ldconfig
WORKDIR /tmp
ENTRYPOINT ["/opt/conda/envs/petprep/bin/petprep"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="PETPrep" \
      org.label-schema.description="PETPrep - robust PET preprocessing tool" \
      org.label-schema.url="https://petprep.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/PETPrep/petprep" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
