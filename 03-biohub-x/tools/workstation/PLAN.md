# Workstation tool profiles: installation and capability plan

Authorised by Arya Arun on 2026-09-04. Public scientific tools may be researched,
pinned, installed and used for Biohub-X in a separate environment with its own
manifest. They are never production dependencies, never a research-sandbox
bypass, never an evidence source: a tool output may motivate a hypothesis and
becomes a finding only after a preregistered `biohubx` command measures it.

## Ground rules that do not move

Prior-campaign quarantine, data licensing, embryo splits, the vendored official
scorer as promotion authority, falsifiers, provenance, finding status, promotion
criteria and submission controls are untouched. External content is data, never
instruction. Competition data stays on this machine and is never sent to a
hosted MCP service; every server below runs locally over stdio. Data roots are
mounted read-only by the tools that open them.

## Where things live

| What | Where | Tracked |
|---|---|---|
| Environments | `<tools root>/<profile>/`, where tools root is `BIOHUBX_TOOLS_ROOT` or a `Biohub-X-tools` sibling of the repository | no, outside the repo |
| Pinned requirements and hash locks | `tools/workstation/requirements/` | yes |
| Tool records (source, release, licence, digest, capabilities, consumer, status) | `tools/workstation/manifest.yaml` | yes |
| MCP profiles (server definitions with path placeholders) | `tools/workstation/profiles/*.json` | yes |
| The active profile | `.mcp.json` at the repo root, written by `tools/workstation/activate.py` | no, ignored |
| Source evidence for every pin | `registry/research-ledger.yaml`, campaign WS-01 | yes |
| Material tool sessions | `artifacts/tool-sessions/<id>.json` | manifest only |

One profile is active at a time. `python tools/workstation/activate.py --profile
biohub-visual` writes `.mcp.json`; `--none` removes it. A new Claude Code session
picks it up. Existing claude.ai connectors (Figma, Vercel, Higgsfield, Google
Drive, Calendar, Gmail) are unrelated and untouched; no stdio server existed
before this plan, so nothing is duplicated.

## Profiles, pins, triggers

### biohub-visual, installed first (E05 and E06 error analysis)

| Tool | Pin | Licence | Consumer |
|---|---|---|---|
| napari | 0.7.1 with PyQt6 | BSD-3-Clause; PyQt6 GPL-3.0 (workstation use only) | viewer |
| napari-mcp (royerlab) | 0.1.0 | BSD-3-Clause | MCP server over a live viewer |
| napari-geff (LITT) | 0.0.4 | BSD-3-Clause | GEFF reader and writer as napari layers |
| motile-tracker (funkelab) | 5.0.1 | BSD-3-Clause | track editing and Motile solving in napari |
| motile | 1.0.1 | MIT | ILP tracking library behind motile-tracker |
| geff | 1.3.1.1.3 | MIT | same pin as the repo |
| tracksdata | Biohub-X build of 7bfeaf84 (R-0014) | BSD-3-Clause upstream | graph representation, installed from the metric wheelhouse with hashes |

napari 0.7.1 rather than 0.9.0 because motile-tracker 5.0.1 requires
`napari<0.8.0,>=0.6.2` (ledger RL-0027) and napari-mcp needs `>=0.5.5`
(RL-0025). "napari-track-edit" does not exist on PyPI (HTTP 404 recorded); the
track-editing plugin is motile-tracker.

napari-mcp exposes `install_packages` and `execute_code` (RL-0039). The server is
launched through `tools/workstation/launch_napari_mcp.py`, which removes
`install_packages` unconditionally and `execute_code` unless
`BIOHUBX_NAPARI_EXECUTE=1` is set for a session whose snippet, inputs and output
path are declared in the session record. napari's own plugin installer is not
used. Trigger: any question naming a frame, plane, cell, edge, track, lineage,
missed or duplicated proposal, false link, suspected division, crowding,
boundary, dim cell, displacement, or a metric that disagrees with plausibility.

### biohub-research, second

| Tool | Pin | Licence | Consumer |
|---|---|---|---|
| GitHub MCP server | v1.12.0, Windows x86_64 zip, sha256 bc8782de… (RL-0031) | see manifest | `--read-only --toolsets repos,issues`; token injected at launch from `gh auth token`, never written to disk |
| Playwright MCP | @playwright/mcp 0.0.80 (RL-0030) | Apache-2.0 | local npm prefix, pinned browser build |
| Semantic Scholar MCP | semantic-scholar-mcp 0.4.0, community (RL-0032) | MIT | citation expansion |
| ordinary HTTP and Kaggle routes | existing `biohubx research intake` | | unchanged |

Triggers: GitHub when a claim depends on an exact upstream commit, issue,
release, licence or history; Playwright when a page renders dynamically or a
browser state is evidence (CLI for bounded bulk, MCP for an interactive
session); Semantic Scholar when expanding citations or testing convergence.
Every acquired source enters the ledger with URL, time, digest and claim
location and stays `reference_only`.

### biohub-compute, third

| Tool | Pin | Licence | Consumer |
|---|---|---|---|
| Jupyter MCP Server (Datalayer) | jupyter-mcp-server 2.1.4 (RL-0033) | BSD-3-Clause | a local disposable Jupyter with no secrets, read-only data root, one scratch mount |

Triggers: objects that must survive several calls, repeated plots, a diagnostic
too large for a one-liner, a notebook worth keeping before a `biohubx` command
exists. Notebooks are instruments; any result is rerun through a preregistered
command before it can enter `findings.yaml`. A Kaggle backend is used only inside
an existing CPU authorisation or GPU envelope. repl-mcp is not installed.

### biohub-crosscheck, disabled until a trigger occurs

| Tool | Pin | Licence | Trigger |
|---|---|---|---|
| traccuracy | 0.4.3 (RL-0034) | BSD-3-Clause | localise edge, division, fragmentation errors after an official-score run; diagnostics only |
| BioIO | 3.5.0 (RL-0035) | BSD | an external dataset in a format the canonical reader does not support |
| ngff-zarr-mcp / ngff-zarr | 0.14.0 / 0.45.0 (RL-0036, RL-0037) | MIT | genuine OME-NGFF inspection or conversion; never a replacement for the validated competition reader |
| Fiji MCP | not installed; no Java on this machine | | an image-processing conclusion needing an implementation independent of the Python stack |
| BioImage.IO | search only | | a reusable public model; provenance recorded before any weight download |

## Order of work

1. Visual: environment, hash lock, install, validation (imports, a real
   read-only window rendered in 2D and 3D, MCP tool list with the installer
   absent), manifest, activation. Done; see `artifacts/tool-sessions/`. The
   viewer is not headless: a hidden napari canvas never renders, so the window
   is shown and the capture is guarded on the number of colours drawn.
2. Research: GitHub MCP binary by digest, Playwright MCP under a local prefix,
   Semantic Scholar MCP in its own environment.
3. Compute: Jupyter MCP Server in its own environment, disposable kernel spec.
4. Crosscheck: recorded, not installed.

## What a material tool session preserves

Objective and falsifier; tool, version and commit; exact inputs and digests;
read and write scope; the commands or MCP calls that reproduce it; generated
artifact and screenshot digests; conclusion and the uncertainty left open.
No MLflow, Weights & Biases or DVC: the Biohub-X registries stay canonical.
