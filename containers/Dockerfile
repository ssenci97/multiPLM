ARG BASE_REF=docker.io/condaforge/miniforge3@sha256:569afc65b5338d35e276adf9a860025cd37a677659d19dcb71fc1d4c2063a94e
FROM ${BASE_REF}

LABEL org.opencontainers.image.title="multiplm"
LABEL org.opencontainers.image.description="Protein language model embeddings (fair-esm ESM2, ESM3, ESMC 300m/600m/6B, ProtT5, ProstT5), MMseqs2 and DADApy in one CUDA image. Weights are not included."
LABEL org.opencontainers.image.licenses="MIT AND Apache-2.0 AND LicenseRef-Cambrian-Open-License"
LABEL org.opencontainers.image.source="https://github.com/ssenci97/MultiPLM"

ARG ESM_EVO_REPO=https://github.com/evolutionaryscale/esm
ARG ESM_EVO_COMMIT=be185fb05e1aa3ccfda4e7bd10e1d341f81489a2
ARG FAIR_ESM_REPO=https://github.com/facebookresearch/esm
ARG FAIR_ESM_COMMIT=0b59d87ebef95948c735b1f7aad463dc6dfa991b

ENV CONDA_ALWAYS_YES=true MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen HOME=/tmp MPLCONFIGDIR=/tmp/mpl PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1 ENVS=/opt/envs

COPY envs/mp.yml envs/esm-evo.txt envs/dadapy.txt containers/mp.lock.tx[t] containers/mp.pip.lock.tx[t] containers/esm-evo.lock.tx[t] containers/dadapy.lock.tx[t] /tmp/
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [ -s /tmp/mp.lock.txt ]; then conda create -q -n mp --file /tmp/mp.lock.txt && /opt/conda/envs/mp/bin/pip install -q --no-deps -r /tmp/mp.pip.lock.txt; \
    else conda env create -q -f /tmp/mp.yml; fi \
    && grep -E '^[[:space:]]*-[[:space:]]+[a-z0-9_.-]+=' /tmp/mp.yml | sed 's/^[[:space:]]*-[[:space:]]*//' > /opt/conda/envs/mp/conda-meta/pinned \
    && conda clean -afy

RUN git config --system --add safe.directory '*' \
    && /opt/conda/envs/mp/bin/pip list --format=freeze | grep -v '@' > /tmp/base.constraints.txt \
    && /opt/conda/envs/mp/bin/python -m venv --system-site-packages "$ENVS/esm-evo" \
    && if [ -s /tmp/esm-evo.lock.txt ]; then "$ENVS/esm-evo/bin/pip" install -q --no-deps -r /tmp/esm-evo.lock.txt; \
       else "$ENVS/esm-evo/bin/pip" install -q -c /tmp/base.constraints.txt -r /tmp/esm-evo.txt; fi \
    && git clone -q "$ESM_EVO_REPO" /opt/src/esm-evo \
    && git -C /opt/src/esm-evo checkout -q -f "$ESM_EVO_COMMIT" \
    && [ "$(git -C /opt/src/esm-evo rev-parse HEAD)" = "$ESM_EVO_COMMIT" ] \
    && "$ENVS/esm-evo/bin/pip" install -q --no-deps /opt/src/esm-evo

RUN /opt/conda/envs/mp/bin/python -m venv --system-site-packages "$ENVS/esm-fair" \
    && git clone -q "$FAIR_ESM_REPO" /opt/src/fair-esm \
    && git -C /opt/src/fair-esm checkout -q -f "$FAIR_ESM_COMMIT" \
    && [ "$(git -C /opt/src/fair-esm rev-parse HEAD)" = "$FAIR_ESM_COMMIT" ] \
    && "$ENVS/esm-fair/bin/pip" install -q --no-deps /opt/src/fair-esm

RUN apt-get update -q && apt-get install -y -q --no-install-recommends gcc libc6-dev curl ca-certificates \
    && /opt/conda/envs/mp/bin/python -m venv "$ENVS/dadapy" \
    && if [ -s /tmp/dadapy.lock.txt ]; then "$ENVS/dadapy/bin/pip" install -q --no-deps -r /tmp/dadapy.lock.txt && "$ENVS/dadapy/bin/pip" install -q --no-deps "$(grep '^dadapy @' /tmp/dadapy.txt)"; \
       else "$ENVS/dadapy/bin/pip" install -q -r /tmp/dadapy.txt; fi \
    && apt-get purge -y -q gcc libc6-dev && apt-get autoremove -y -q && rm -rf /var/lib/apt/lists/*

COPY LICENSES /opt/licenses
COPY LICENSES.md /opt/licenses/README.md

RUN /opt/conda/envs/mp/bin/python -c "import torch, transformers, sentencepiece, h5py, numpy, yaml, tqdm, huggingface_hub, safetensors; print('mp', torch.__version__, torch.version.cuda, transformers.__version__)" \
    && ! /opt/conda/envs/mp/bin/python -c "import esm" 2>/dev/null \
    && "$ENVS/esm-evo/bin/python" -c "import esm.pretrained, esm.utils.constants.esm3; from esm.models.esmc import ESMC; from esm.models.esm3 import ESM3; from esm.sdk.api import ESMProtein, LogitsConfig; from esm.tokenization import get_esmc_model_tokenizers; get_esmc_model_tokenizers(); print('esm-evo ok')" \
    && "$ENVS/esm-fair/bin/python" -c "import esm, esm.pretrained; assert hasattr(esm.pretrained, 'load_model_and_alphabet_local'); print('esm-fair', esm.__version__)" \
    && /opt/conda/envs/mp/bin/mmseqs version \
    && "$ENVS/dadapy/bin/python" -c "import numpy as np; from dadapy import Data; d = Data(coordinates=np.random.default_rng(0).random((300, 3)), maxk=3, verbose=False, n_jobs=1).compute_id_2NN()[0]; assert 2 < d < 4, d; print('dadapy ok', d)" \
    && ! "$ENVS/dadapy/bin/python" -c "import torch" 2>/dev/null \
    && for b in git curl flock; do command -v "$b" > /dev/null || { echo "MISSING $b"; exit 1; }; done
