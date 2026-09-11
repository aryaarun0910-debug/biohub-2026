#!/usr/bin/env python3
"""Biohub-X findings registry -> DB, CURATED not dumped.

Each finding was read and judged. KEEP carries a reason the claim still matters for the
2026-09-29 sprint; SKIP carries the reason it does not. The skips are recorded too, so the
judgement is auditable and nobody re-reads 35 findings to rediscover which 13 were noise.
"""
import os, sqlite3, datetime, yaml, textwrap, hashlib

from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB   = os.path.join(ROOT, "db", "biohub_base.db")
SRC  = os.path.join(ROOT, "repos", "Biohub-X", "registry", "findings.yaml")
NOW  = datetime.date.today().isoformat()

# id -> (keep?, topic, why)
JUDGED = {
"F-0001": (0, None, "Repository-isolation contract; the blind-isolation rule was dissolved 2026-09-07."),
"F-0002": (0, None, "CRLF/canonical digest behaviour - an implementation detail of a system we are porting wholesale."),
"F-0003": (1, "metric", "Unmatched predicted nodes are NOT edge false positives. Sets the whole economics of detection precision."),
"F-0004": (1, "metric", "Node-count adjustment is clamped only at zero, so the multiplier exceeds 1 when under-predicting. Measured 1.09."),
"F-0005": (1, "metric", "Frame-skipping edges are silently DROPPED by the scorer, not charged. Affects gap-recovery designs."),
"F-0006": (0, None, "Synthetic-fixture vertical slice; superseded by the real-data findings."),
"F-0007": (0, None, "Fixture-scale candidate-reach demo; the corpus-wide versions (F-0026..F-0032) supersede it."),
"F-0008": (0, None, "Fixture reproduction of F-0003/F-0004; the corpus-wide F-0015/F-0018 are the usable versions."),
"F-0009": (1, "metric", "An edgeless graph is UNSCORABLE - evaluate() returns early and raises. A graph with no edges is a bug, not a low score."),
"F-0010": (1, "data", "The visible test split is 4 volumes, byte-identical to train, verified by tree digest. Never select on it."),
"F-0011": (1, "data", "199 datasets are 2 embryos, not 199 subjects: 6bba has 128 fields of view, 44b6 has 71."),
"F-0012": (0, None, "Superseded by F-0013, which measures annotated fraction properly rather than by byte size."),
"F-0013": (1, "data", "133,318 annotated nodes / 128,883 edges against 4,725,117 estimated - 2.82% annotated, ranging 0.13-20.21% per dataset."),
"F-0014": (1, "data", "DIVISION SCARCITY: 151 annotated divisions in the whole corpus; 112 of 199 datasets have none. LOEO trains divisions on 26 (44b6) or 125 (6bba) examples."),
"F-0015": (1, "metric", "Predicting the ANNOTATED count rather than the estimated total yields a median multiplier of 1.0964 (range 1.0798-1.0999)."),
"F-0016": (1, "metric", "The division counter shares the unmatched-prediction exemption: an all-unmatched fork is uncharged but still raises num_pred_nodes."),
"F-0017": (0, None, "An 0.082 score from the first real-data chain - an obsolete baseline, and its lesson is restated in F-0025."),
"F-0018": (1, "strategy", "THE ECONOMICS OF THE NODE-COUNT LEVER: worth ~9% of adjusted edge Jaccard, and it stops paying once a filter destroys more than ~8% of correct edges."),
"F-0019": (0, None, "Superseded by F-0020, then F-0021."),
"F-0020": (0, None, "Superseded by F-0021, which audits every node rather than a sample."),
"F-0021": (1, "data", "NO intensity band can call any voxel background: the dimmest annotated cell sits at percentile 0.09. Audited over all 133,318 nodes, zero excluded."),
"F-0022": (1, "data", "The SCAR assumption nnPU relies on is NOT supported: annotation density correlates with annotated-cell intensity at Pearson r=+0.354. Densely annotated movies carry BRIGHTER annotations."),
"F-0023": (1, "strategy", "Minimum-track-length filtering with isolated-node pruning is near-free: 0.9963 / 0.9945 edge retention at span 6, with the division exemption."),
"F-0024": (0, None, "Process check that the nnPU loop runs on a T4; F-0022 undermines the nnPU premise anyway."),
"F-0025": (1, "method", "Multi-scale Difference-of-Gaussians dominates the classical local-max detector at a matched proposal budget (0.6154 vs 0.3846 reach)."),
"F-0026": (1, "method", "DoG reaches roughly TWICE the annotated cells of the classical detector, on both embryos, at every budget swept."),
"F-0027": (1, "method", "Reachability plateaus on 44b6 and still climbs on 6bba out to 8x the estimated cell count; one-to-one node recall tracks reach within 0.01."),
"F-0028": (1, "method", "Residual misses differ by embryo: on 44b6 it is a LOCALISATION failure, not a detection failure - a proposal already lies within 10um of 73% of missed cells."),
"F-0029": (1, "method", "The near-miss residual is in-plane, not a z artefact: median in-plane offset 7.46/7.85um against median |z| 3.25/4.88."),
"F-0030": (1, "method", "Greedy physical-radius suppression does NOT return local maxima - it packs the above-threshold region. Only 46 of 686 accepted points were strict 26-neighbourhood maxima."),
"F-0031": (1, "method", "DoG BANK TUNING: dropping the 4.5um scale RAISES reachability on 44b6 everywhere; adding 2.5um changes nothing. Use (2.0, 3.0)."),
"F-0032": (1, "method", "BEST MEASURED DETECTOR: strict 26-neighbourhood maxima on a (2.0,3.0)um DoG bank reaches 0.9239 (44b6) / 0.8984 (6bba) while emitting at or below the estimated cell count."),
"F-0033": (1, "strategy", "ORACLE ASSOCIATION CEILING: with ground-truth-chosen edges over frozen strict peaks, 0.9961 on 44b6 but only 0.9227 on 6bba. Detection is still the binding constraint on 6bba."),
"F-0034": (1, "method", "Per-scale strict peaks unioned before one shared suppression recovers what max-over-scales fusion loses; whether it helps depends entirely on the bank."),
"F-0035": (1, "strategy", "Truncating proposals to a budget chosen on the TRAINING embryo does not raise the held-out ceiling; the per-scale union without truncation does."),
}

def main():
    cx = sqlite3.connect(DB)
    cx.execute("PRAGMA foreign_keys=ON")
    digest = hashlib.sha256(Path(SRC).read_bytes()).hexdigest()
    cx.execute("INSERT OR IGNORE INTO source(url,sha256,method,fetched_at,note)"
               " VALUES('repos/Biohub-X/registry/findings.yaml',?,'git',?,?)",
               (digest, NOW, "Biohub-X findings registry, curated"))
    sid, observed = cx.execute("SELECT id,fetched_at FROM source WHERE sha256=?", (digest,)).fetchone()
    doc = yaml.safe_load(Path(SRC).read_text())
    kept = skipped = 0
    for f in doc["findings"]:
        fid = f["id"]
        keep, topic, why = JUDGED.get(fid, (0, None, "not reviewed"))
        claim = " ".join(f["claim"].split())
        ev = f.get("evidence") or {}
        quote = " ".join(str(ev.get("observed") or ev.get("command") or "")[:400].split())
        if keep:
            cx.execute("INSERT INTO fact"
                       "(topic,key,value,claim_type,confidence,source_id,quote,observed_at,status)"
                       " VALUES(?,?,?,'observation','high',?,?,?,'active')"
                       " ON CONFLICT(topic,key,observed_at) DO UPDATE SET value=excluded.value,"
                       " source_id=excluded.source_id,quote=excluded.quote",
                       (topic, f"{fid}: {why}", claim, sid,
                        f["claim"], observed[:10]))
            kept += 1
        else:
            cx.execute("INSERT INTO fact"
                       "(topic,key,value,claim_type,confidence,source_id,quote,observed_at,status)"
                       " VALUES('bx-skipped',?,?,'decision','high',?,?,?,'retracted')"
                       " ON CONFLICT(topic,key,observed_at) DO UPDATE SET value=excluded.value,"
                       " source_id=excluded.source_id,quote=excluded.quote",
                       (f"{fid}: NOT INGESTED", why, sid, f["claim"], observed[:10]))
            skipped += 1
    cx.commit()
    print(f"  findings judged: {kept} kept, {skipped} skipped as fat")
    for t, n in cx.execute("SELECT topic, COUNT(*) FROM fact WHERE source_id=? AND status='active'"
                           " GROUP BY topic ORDER BY 2 DESC", (sid,)):
        print(f"    {t:<12} {n}")
    cx.close()

if __name__ == "__main__":
    main()
