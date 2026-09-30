# The biohub-compute sandbox: a Jupyter server that can read the competition data and
# the repository's source, and nothing else on this machine.
#
# Base image pinned by digest (the same python:3.12-slim R-0014's build used).
# Every package comes from the hash lock beside this file, compiled for
# linux/cp312 against the repository's own uv.lock as constraints, so the
# sandbox sees the versions the scorer sees. tracksdata is the Biohub-X build
# (R-0014), copied in with its recorded hash; pip is never allowed to resolve.
#
# Runtime isolation is the launcher's job (tools/workstation/launch_jupyter_mcp.py):
# internal network with no outbound route, read-only root, data root mounted
# read-only, one scratch mount, non-root user, dropped capabilities.
FROM python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY biohub-compute-sandbox.lock.txt /build/lock.txt
COPY tracksdata-requirement.txt /build/tracksdata-requirement.txt
COPY wheels/ /build/wheels/

RUN pip install --require-hashes --no-deps -r /build/lock.txt \
    && pip install --no-index --no-deps --require-hashes --find-links /build/wheels \
         -r /build/tracksdata-requirement.txt \
    && python -m ipykernel install --prefix /usr/local --name python3 \
    && rm -rf /build \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox

# The kernel finds biohubx on the read-only source mount the launcher provides.
ENV PYTHONPATH=/repo/src \
    HOME=/home/sandbox \
    BIOHUB_DATA_ROOT=/data \
    MPLCONFIGDIR=/tmp/matplotlib

USER 1000:1000
WORKDIR /scratch
EXPOSE 8888
