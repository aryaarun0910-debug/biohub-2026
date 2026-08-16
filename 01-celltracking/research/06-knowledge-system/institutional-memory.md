---
id: 06-knowledge-system/institutional-memory
title: Institutional Memory
area: 06-knowledge-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- sqlite
- memory
---

# Institutional Memory

> The durable research memory: `research.sqlite`, folded into the repo from the old `_RESEARCH`
> sibling. It is **gitignored** (binary DB) but physically present at
> `06-knowledge-system/research.sqlite`.

## Contents

- **102 findings** (73 new · 24 tested · 5 falsified) in table `finding`.
- **7 agent runs** in table `agent_run`.
- `finding` columns include: `title, mechanism, pipeline_component, expected_benefit,
  cheapest_falsification, status`.

## Query

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect(r'research/06-knowledge-system/research.sqlite'); \
  [print(t, c.execute(f'select count(*) from {t}').fetchone()[0]) for t in ('finding','agent_run')]"
```

## Relationship to the machine

Findings are the raw substrate; the curated portfolio derived from them lives in
[../01-research-direction/research-bets.md](../01-research-direction/research-bets.md) +
[bets.yaml](../01-research-direction/bets.yaml). Falsified findings are summarised in
[failed-experiments.md](failed-experiments.md). Full raw agent transcripts live in the
gitignored `../../_evidence/agent_runs/`.
