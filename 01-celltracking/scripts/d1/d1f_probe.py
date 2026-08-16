"""D1-F — frozen-feature linear probe. Full rewrite (v6). Not a patch of the v5 script.

WHY A REWRITE. The v5 `scripts/d1/d1f_probe.py` had six independent defects, any one of
which invalidates the instrument:

  1. loaded `{crop}__feat_near.npy`; the export writes `__feat_max.npy`
     (v6: `__feat_tta_mean_max.npy`). It could never have opened a real artifact.
  2. read `d1_class` and `near_dist_um`; the export emits neither.
  3. `main()` FITTED H0 instead of loading the checkpoint head, so H0 was not a parity
     reference — it was a seventh fitted arm wearing the deployed head's name.
  4. `build_arm(..., pi_crop=None)` was hardcoded and no temporal branch was ever
     written, so H1/H2/H3/H4 were BYTE-IDENTICAL. Four identical rows would have read
     as convergent evidence.
  5. `select_threshold` maximised F1 on a pool that is ~96% background rows standing in
     for a ~1e-5 population base rate. F1 there is a statistic about the sampler.
  6. `grouped_inner_split` was computed and then discarded; the threshold was selected
     on the very rows the coefficients were fitted on.

WHAT THIS INSTRUMENT ANSWERS, AND WHAT IT MAY SAY. `detect_head` is
`Conv3d(32, 1, kernel_size=1)` = 33 parameters. A low logit at a missing nucleus is
consistent with a bad operating point, with a linear map too weak to exploit an
informative feature, or with an uninformative feature. This probe separates the first
two. It CANNOT establish the third, and it is forbidden from claiming it:

    PERMITTED VERDICTS ARE EXACTLY FOUR
      CALIBRATION_GLOBAL    one corpus-wide scalar suffices. TRIVIAL: it says go run a
                            threshold sweep, which needs no instrument.
      CALIBRATION_PER_CROP  the operating point must move DIFFERENTLY per crop, so NO
                            single deployed constant can work. A finding about the
                            deployment rule, not about the head.
      LINEAR_HEAD           a ranking-changing linear refit works
      LINEAR_PROBE_NULL     this probe failed

WHY CALIBRATION SPLITS IN TWO. Acceptance is `logit == max_pool3d(logit)` AND
`sigmoid(logit) > tau`. That is a LEVEL SET: for any strictly increasing phi,
`A(phi(s), phi(tau)) == A(s, tau)` EXACTLY, and the local-max test runs on the RAW
logits, so even isotonic's flat segments cannot break ties differently. Temperature,
Platt, beta, histogram binning and isotonic are therefore ALL worth exactly one scalar on
this detector — `det_threshold`. Post-hoc calibration has ONE degree of freedom here, not
many. (The ECCV-2024 ">7 D-ECE, post-hoc beats train-time" result is real and simply does
not apply to an acceptance decision.) A bare `CALIBRATION` verdict is consequently
uninterpretable, and `RETIRED_VERDICTS` makes a caller that asks for one fail loudly.
`calibration_heterogeneity` is what separates the two: Cochran's Q on the per-crop optimal
intercept shifts, against the sandwich variance the HT weights imply.

M1 IS NOT CALIBRATION. The re-acceptance head is a re-DIRECTION with 32 degrees of
freedom; it escapes the level-set theorem entirely. Conflating a 32-DOF rotation with a
1-DOF monotone rescale is exactly the error the split above exists to prevent, and it is
why `LINEAR_HEAD` and the calibration tokens are different verdicts rather than degrees of
one.

`REPRESENTATION DEFICIT` is not in the vocabulary. A null linear probe on a sampled row
table cannot fund an encoder programme, and `_assert_permitted_verdict` raises rather
than let one be written into a report. (Decision package §0, correction C2.)

PRIOR SHIFT IS REFUSED, NOT MERELY UNIMPLEMENTED. `REFUSED_ARMS` blocks SLD/BBSE-style
base-rate correction with its reason: those estimators assume LABEL shift, the gap here is
CONDITIONAL shift, and worked from the measured per-family priors the correction
prescribes RAISING 6bba's threshold by 0.808 logits — the family with 9.7x the miss rate
and 85% of the edge mass.

DESIGN CONSTRAINTS ENFORCED IN CODE, NOT IN PROSE:

  * 2x2 CHECKPOINT x FAMILY (C1). 44b6 sits in split-0's 32-D basis and 6bba in
    split-1's; the bases are independent (rel-L2 1.41542 ~ sqrt(2)). A head fitted in
    one basis may never be applied in the other. Within ONE basis: fit and select on the
    SOURCE family (the family that checkpoint trained on), freeze, then open the TARGET
    family exactly once. `assert_same_basis` makes the cross-basis path impossible.
  * ARMS MUST BE DISTINCT. Before any fitting, each arm's FULL objective (weights,
    offsets, auxiliary terms, ridge) is evaluated at a fixed probe beta and the
    `(objective, grad_sha256)` pairs must be pairwise distinct. After fitting, the
    deployed 33-vectors must also be pairwise distinct. A distinct label or config hash
    is not evidence.
  * CAPABILITY REGISTRY. An arm whose mechanism is not implemented on the supplied data
    is BLOCKED and never reaches the results table. `pi_crop=None` raises at
    construction; a missing temporal table blocks H3/H4 rather than degrading them.
  * HORVITZ-THOMPSON. `uniform` rows are a 1/4096 subsample of the output grid, so an
    uncorrected intercept is overstated by log(4096) = 8.3178 logits. Grid dims absent
    from the manifest => raise. Never guess a rate.
  * NESTED SELECTION. Thresholds are chosen out-of-fold on grouped-by-crop inner folds
    with a REFIT per fold; the coefficients are never selected on their own fit rows.
  * BASIS STAMPS. Ranking and calibration are reported separately, and every sampled-row
    diagnostic carries `"promotes": false`. Promotion requires dense inference -> P3
    harmonic -> complete wrapper -> exact patched pooled scorer. Nothing here.
  * TWO TRANSFER DIRECTIONS, NEVER POOLED (C7). The 199 crops come from two embryos;
    crop-block bootstrap measures within-embryo variation and does not estimate
    private-embryo generalisation. The payload carries no pooled headline, by
    construction and by assertion.

ARMS
    H0        deployed head from the ROUTED fold checkpoint. Not fitted, no threshold
              selection, deployed threshold 0.96875 (logit 3.4340). Aborts unless the
              checkpoint head reproduces the exported logits from the exported features.
    H1        genuinely masked local head. Requires a real per-row
              `dist_to_nearest_gt_um` and a radius; a mask derived from `kind` alone is
              refused because it is indistinguishable from the base exposure profile.
    H2        H1 + implemented count/GE term. `pi_crop` MANDATORY; null RAISES.
    H3        H1 + real temporal pseudo-labels. Tables MANDATORY; absence BLOCKS.
    H4        H2 + H3. BLOCKED unless both prerequisites exist.
    H5        positive-exposure control. Loss functional unchanged, positives reweighted
              only. Clearly separated from H0 and never a parity reference.
CONTROLS
    CAL_ONLY  2-parameter (a, c) refit of `a*eta0 + c` with w frozen at w_ckpt.
    LIN_HEAD  full 33-parameter refit.
    SHUF_LIN  label-shuffled LIN_HEAD. Must return a null or the instrument is measuring
              leakage and no verdict is emitted at all.
    SHUF_CAL  label-shuffled CAL_ONLY. Supplies the calibration null band, so "learning
              the base rate" cannot be mistaken for a calibration repair.
    same-family evaluation basis: every arm is additionally read on held-out crops of the
              SOURCE family, giving the within-family ceiling against which the
              cross-family number is read.

No Kaggle, no GPU, no submission. numpy only for the fitter; polars and torch are
imported lazily and only for artifact/checkpoint IO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

ROOT = next(_p for _p in pathlib.Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())

# ============================================================================ constants
FEAT_DIM = 32
N_HEAD_PARAMS = FEAT_DIM + 1                      # Conv3d(32,1,k=1) = 33 parameters

DET_THRESHOLD = 0.96875
DET_THRESHOLD_LOGIT = float(np.log(DET_THRESHOLD / (1.0 - DET_THRESHOLD)))   # 3.4339872

# `uniform` rows are a 1/4096 subsample of the downsampled output grid. Without the
# Horvitz-Thompson correction the fitted intercept sits on the SAMPLED prior and is
# overstated by exactly this many logits.
UNIFORM_SUBSAMPLE_DENOM = 4096
HT_LOGIT_CORRECTION_1_4096 = math.log(UNIFORM_SUBSAMPLE_DENOM)              # 8.3177662

# split_k is the checkpoint that HELD OUT family k, and therefore also the checkpoint
# that TRAINED ON the other family. Both facts are needed for the 2x2.
FAMILY_TO_SPLIT = {"44b6": 0, "6bba": 1}
SPLIT_HELDOUT_FAMILY = {0: "44b6", 1: "6bba"}     # target: never seen by this checkpoint
SPLIT_SOURCE_FAMILY = {0: "6bba", 1: "44b6"}      # source: this checkpoint trained on it
CHECKPOINT_SHA256 = {
    0: "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f",
    1: "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de",
}

# The exposure profile of the deployed detection loss (retained training_config
# 4f29349439e133ad): det_neg_weight = 0.01, det_loss_weight = 1.0.
DEPLOYED_NEG_WEIGHT = 0.01

ROW_KINDS = ("gt_centre", "uniform", "subthr_localmax")

V6_REQUIRED_FEATURE_FILES = ("__feat_tta_mean_gt.npy", "__feat_tta_mean_max.npy")
V6_REQUIRED_MANIFEST_KEYS = ("crops", "tta_view_set", "n_encode_calls",
                            "n_distinct_views")
V6_REQUIRED_CROP_KEYS = ("grid_zyx", "n_frames", "n_uniform_per_frame",
                         "estimated_number_of_nodes", "checkpoint_sha256", "split")
# Columns the v5 script assumed and the export does not emit. Their PRESENCE is as much
# a contract violation as their absence: it would mean something else is being read.
V5_ABSENT_COLUMNS = ("d1_class", "matched", "d_stratum", "near_dist_um")

PERMITTED_VERDICTS = ("CALIBRATION_GLOBAL", "CALIBRATION_PER_CROP", "LINEAR_HEAD",
                      "LINEAR_PROBE_NULL")
# Retired token. A bare `CALIBRATION` is uninterpretable on this detector: acceptance is a
# LEVEL SET, so it cannot distinguish the trivial case from the interesting one. Kept only
# so a stale caller fails loudly instead of silently matching nothing.
RETIRED_VERDICTS = {"CALIBRATION": (
    "`CALIBRATION` is retired. Acceptance is `logit == max_pool3d(logit)` AND "
    "`sigmoid(logit) > tau`, a LEVEL SET: for any strictly increasing phi, "
    "A(phi(s), phi(tau)) == A(s, tau) exactly, and the local-max test runs on the RAW "
    "logits so even isotonic's flat segments cannot break ties differently. Temperature, "
    "Platt, beta, histogram binning and isotonic are therefore ALL worth exactly one "
    "scalar here — det_threshold. A bare CALIBRATION verdict buys a threshold sweep we "
    "can already run without this instrument. Use CALIBRATION_GLOBAL (one corpus-wide "
    "scalar: trivial, go run the sweep) or CALIBRATION_PER_CROP (the operating point must "
    "move DIFFERENTLY per crop, so no single deployed constant can work — a statement "
    "about the deployment rule, not about the head).")}

# Arms this instrument refuses to express, with the reason. Refusing in the registry is
# stronger than not implementing them: a later caller asking for one gets the argument.
REFUSED_ARMS = {
    "PRIOR_SHIFT": (
        "a prior-shift / SLD / BBSE base-rate correction is WRONG-SIGNED here. Worked "
        "from the measured per-family priors it prescribes RAISING 6bba's threshold by "
        "0.808 logits — the family with 9.7x the miss rate and 85% of the edge mass. "
        "Those estimators assume LABEL shift (p(y) moves, p(x|y) fixed). The gap here is "
        "CONDITIONAL shift: p(x|y) differs between families. Applying a label-shift "
        "correction to a conditional-shift problem moves the threshold the wrong way on "
        "the family that matters most."),
    "SLD": "alias of PRIOR_SHIFT; see REFUSED_ARMS['PRIOR_SHIFT']",
    "BBSE": "alias of PRIOR_SHIFT; see REFUSED_ARMS['PRIOR_SHIFT']",
}

# Pre-registered: how much per-crop spread in the optimal intercept counts as "one
# constant cannot serve every crop". POLICY, not a calibrated transfer law.
MIN_PER_CROP_SPREAD_LOGITS = 0.25
# Encoded as a hard constraint, per correction C2. These are the phrasings a null probe
# must never be allowed to produce.
_FORBIDDEN_VERDICT_TOKENS = ("REPRESENTATION", "DEFICIT", "ENCODER", "RETRAIN",
                             "SUBSTRATE", "BACKBONE")

# Pre-registered decision bands. POLICY, not a calibrated transfer law (C7).
MIN_RANKING_DELTA_AUC = 0.01
MIN_CALIBRATION_DELTA_NATS = 0.01


class ContractError(RuntimeError):
    """The supplied artifact does not satisfy the v6 contract."""


class ParityError(AssertionError):
    """A head failed to reproduce the logits it claims to reproduce."""


class BasisError(RuntimeError):
    """A head fitted in one checkpoint basis was about to touch another basis's rows."""


class MissingCapability(RuntimeError):
    """A required input for a term/arm does not exist. The arm is BLOCKED."""


class ControlFailure(RuntimeError):
    """A negative control did not return a null. No verdict may be emitted."""


class VerdictError(RuntimeError):
    """An impermissible verdict was constructed."""


# ============================================================================ numerics
def _softplus(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z > 0
    out[pos] = z[pos] + np.log1p(np.exp(-z[pos]))
    out[~pos] = np.log1p(np.exp(z[~pos]))
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _eta(X: np.ndarray, idx, beta: np.ndarray, offset) -> np.ndarray:
    d = X.shape[1]
    e = X[idx].astype(np.float64, copy=False) @ beta[:d] + float(beta[d])
    if offset is not None:
        e = e + offset[idx]
    return e


def _xt_sum(X: np.ndarray, idx, coef: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    g = np.zeros(d + 1, dtype=np.float64)
    g[:d] = X[idx].astype(np.float64, copy=False).T @ coef
    g[d] = float(coef.sum())
    return g


# ============================================================================ head
@dataclass(frozen=True)
class LinearHead:
    """A 33-parameter detection head. Deployed logit at voxel v is `w . f(v) + b`."""

    w: np.ndarray
    b: float
    name: str = "head"
    basis_split: int | None = None       # the checkpoint basis these features live in
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = np.asarray(self.w, dtype=np.float64).reshape(-1)
        if w.shape != (FEAT_DIM,):
            raise ValueError(f"weight must be ({FEAT_DIM},), got {np.shape(self.w)}")
        if not np.isfinite(w).all() or not np.isfinite(self.b):
            raise ValueError("non-finite head parameters")
        object.__setattr__(self, "w", w)
        object.__setattr__(self, "b", float(self.b))

    def logit(self, X: np.ndarray, dtype: str = "float64") -> np.ndarray:
        """`dtype='float32'` reproduces the deployed conv's arithmetic class: no autocast
        is used in predict_unet_transformer.py and Kaggle T4/P100 have no TF32 path."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != FEAT_DIM:
            raise ValueError(f"expected (N, {FEAT_DIM}) features, got {X.shape}")
        if dtype == "float32":
            return (X.astype(np.float32) @ self.w.astype(np.float32)
                    + np.float32(self.b)).astype(np.float32)
        return X.astype(np.float64) @ self.w + self.b

    def beta(self) -> np.ndarray:
        return np.concatenate([self.w, [self.b]])

    def param_sha256(self) -> str:
        return hashlib.sha256(
            np.round(self.beta(), 12).astype(np.float64).tobytes()).hexdigest()[:32]

    @classmethod
    def from_beta(cls, beta, name="head", basis_split=None, **prov) -> "LinearHead":
        beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        if beta.shape != (N_HEAD_PARAMS,):
            raise ValueError(f"beta must be ({N_HEAD_PARAMS},), got {beta.shape}")
        return cls(w=beta[:FEAT_DIM], b=float(beta[FEAT_DIM]), name=name,
                   basis_split=basis_split, provenance=dict(prov))

    @classmethod
    def from_checkpoint(cls, path, *, verify_sha256=None, split=None) -> "LinearHead":
        import torch  # noqa: PLC0415 — optional, checkpoint reads only

        path = pathlib.Path(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if verify_sha256 is not None and sha != verify_sha256:
            raise ParityError(f"checkpoint sha256 {sha} != expected {verify_sha256}")
        if split is not None and CHECKPOINT_SHA256.get(split) not in (None, sha):
            raise ParityError(
                f"{path.name} sha256 {sha[:16]} is not the pinned split-{split} weight")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict) or "detect_head.weight" not in sd:
            raise ParityError(f"{path} carries no detect_head.weight")
        wt, bt = sd["detect_head.weight"], sd["detect_head.bias"]
        if tuple(wt.shape) != (1, FEAT_DIM, 1, 1, 1):
            raise ParityError(f"detect_head.weight {tuple(wt.shape)} != (1,32,1,1,1)")
        if tuple(bt.shape) != (1,):
            raise ParityError(f"detect_head.bias {tuple(bt.shape)} != (1,)")
        return cls(w=wt.reshape(FEAT_DIM).double().numpy(),
                   b=float(bt.double().numpy()[0]),
                   name=f"H0_split{split}", basis_split=split,
                   provenance={"kind": "checkpoint_detect_head", "path": str(path),
                               "sha256": sha, "split": split, "fitted": False})

    @classmethod
    def for_basis(cls, split: int, weights_dir) -> "LinearHead":
        """The checkpoint that DEFINES basis `split`. Every row in that basis — source
        family and target family alike — was encoded by this checkpoint."""
        p = pathlib.Path(weights_dir) / f"edge_predictor_best_split_{split}.pth"
        return cls.from_checkpoint(p, verify_sha256=CHECKPOINT_SHA256[split], split=split)

    def save(self, path) -> pathlib.Path:
        """numpy trap: savez_compressed appends `.npz`, so the temp name must end in it
        or the later replace() fails after the work is done."""
        path = pathlib.Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez_compressed(tmp, w=self.w, b=np.array([self.b], dtype=np.float64))
        tmp.replace(path)
        path.with_suffix(".head.json").write_text(json.dumps({
            "format": "d1f-linear-head/2", "name": self.name,
            "basis_split": self.basis_split, "n_params": N_HEAD_PARAMS,
            "layout": "logit = w @ x + b ; deploy as detect_head.weight=w.reshape("
                      "1,32,1,1,1), detect_head.bias=[b]",
            "w": [float(v) for v in self.w], "b": float(self.b),
            "param_sha256": self.param_sha256(), "provenance": self.provenance,
        }, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path


def assert_logit_parity(head: LinearHead, X: np.ndarray, logits: np.ndarray, *,
                        atol: float = 1e-4, label: str = "") -> dict:
    """HARD GATE. `head` must reproduce `logits` from `X`, or the run aborts.

    Tolerance basis (D1_V6_SPEC.md §2): a 32-term float32 dot product accumulates ~1e-6
    of rounding and the value round-trips through float32 .npy storage. 1e-4 absolute is
    the specified gate. It is NOT slack for a TTA gap, which is O(0.1-1) logits and blows
    straight through it — a systematically signed residual means identity-view features
    were exported under the detector's name, which is blocker B3, not a tolerance issue.
    """
    X = np.asarray(X)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if X.shape[0] != logits.shape[0]:
        raise ParityError(f"{label}: {X.shape[0]} feature rows vs {logits.shape[0]} logits")
    if X.shape[0] == 0:
        raise ParityError(f"{label}: no rows to check parity on")
    got = head.logit(X, dtype="float32").astype(np.float64)
    err = np.abs(got - logits)
    n_bad = int((err > atol).sum())
    signed = float((got - logits).mean())
    denom = float(np.std(logits)) * float(np.std(got - logits))
    corr = (float(np.mean((logits - logits.mean()) * ((got - logits) - signed)) / denom)
            if denom > 1e-15 else 0.0)
    report = {
        "label": label, "n_rows": int(X.shape[0]),
        "max_abs_err": float(err.max()), "p999_abs_err": float(np.quantile(err, 0.999)),
        "median_abs_err": float(np.median(err)), "mean_signed_err": signed,
        "residual_logit_corr": corr, "n_over_tol": n_bad, "atol": atol,
        "operand_dtypes": {"features": str(X.dtype), "logits": str(logits.dtype),
                           "head": "float32 (deployment arithmetic)"},
        "passed": n_bad == 0,
    }
    if n_bad:
        raise ParityError(
            f"{label}: {n_bad}/{X.shape[0]} rows exceed atol={atol}; max_abs_err="
            f"{report['max_abs_err']:.6g}, mean_signed_err={signed:.6g}, "
            f"residual/logit corr={corr:.4f}. A signed or correlated residual is a real "
            f"mismatch (identity-view features, partial accumulation, stale head), not a "
            f"tolerance problem. There is no proceed-with-caveat branch.")
    return report


# ============================================================================ contract
@dataclass
class Corpus:
    """A loaded, validated corpus. One row = one sampled voxel in ONE checkpoint basis."""

    X: np.ndarray                       # (N, 32) float32 — TTA-MEAN feature
    X_max: np.ndarray                   # (N, 32) float32 — TTA-MEAN at best local max
    kind: np.ndarray
    crop: np.ndarray
    family: np.ndarray
    t: np.ndarray
    logit: np.ndarray                   # the DEPLOYED post-TTA logit
    encoder_split: np.ndarray           # (N,) int — which checkpoint encoded this row
    dist_to_nearest_gt_um: np.ndarray | None = None
    manifest: dict = field(default_factory=dict)
    feature_source: str = "tta_mean"

    def __post_init__(self):
        n = self.X.shape[0]
        for nm in ("X_max", "kind", "crop", "family", "t", "logit", "encoder_split"):
            if len(getattr(self, nm)) != n:
                raise ContractError(f"{nm} has {len(getattr(self, nm))} rows, X has {n}")
        if self.X.shape[1] != FEAT_DIM or self.X_max.shape[1] != FEAT_DIM:
            raise ContractError("feature blocks must be (N, 32)")
        bad = set(np.unique(self.kind)) - set(ROW_KINDS)
        if bad:
            raise ContractError(f"unknown row kinds {sorted(bad)}; expected {ROW_KINDS}")
        if not np.isfinite(self.X).all():
            raise ContractError("non-finite sampled-voxel features")
        # X_max is defined only for GT-centre rows with a local maximum inside the
        # search sphere.  The exporter deliberately writes an all-NaN sentinel for
        # uniform/subthreshold samples and for GT rows with no such maximum.  Partial
        # rows are corruption; all-finite and all-NaN rows are the two valid states.
        max_finite = np.isfinite(self.X_max)
        partial = max_finite.any(axis=1) & ~max_finite.all(axis=1)
        if partial.any():
            raise ContractError(
                f"partially non-finite max features on {int(partial.sum())} row(s)")

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def is_positive(self) -> np.ndarray:
        return self.kind == "gt_centre"

    @property
    def y(self) -> np.ndarray:
        return self.is_positive.astype(np.float64)

    def bases(self) -> list:
        return sorted(set(int(v) for v in np.unique(self.encoder_split)))

    def basis_mask(self, split: int) -> np.ndarray:
        return self.encoder_split == int(split)

    def family_mask(self, family: str) -> np.ndarray:
        return self.family == family

    def crops(self, mask: np.ndarray | None = None) -> list:
        c = self.crop if mask is None else self.crop[mask]
        return sorted(set(c.tolist()))

    def _crop_manifest(self, crop: str) -> dict:
        m = self.manifest.get("crops", {}).get(crop)
        if m is None:
            raise ContractError(f"manifest has no entry for crop {crop!r}")
        return m

    def sampling_rate(self) -> dict:
        """n_uniform_rows / (n_frames * Z*Y*X). RAISES if the grid dims are absent —
        the 1/4096 correction is 8.3178 logits and may never be guessed."""
        out = {}
        for c in self.crops():
            m = self._crop_manifest(c)
            if "grid_zyx" not in m or "n_frames" not in m:
                raise ContractError(
                    f"crop {c!r}: manifest lacks grid_zyx/n_frames, so the uniform "
                    f"subsampling rate is unknown. The Horvitz-Thompson intercept "
                    f"correction at 1/{UNIFORM_SUBSAMPLE_DENOM} is "
                    f"{HT_LOGIT_CORRECTION_1_4096:.4f} logits; guessing it is not an "
                    f"option. Export the output-grid dims.")
            Z, Y, X = m["grid_zyx"]
            n_uni = int(((self.crop == c) & (self.kind == "uniform")).sum())
            out[c] = n_uni / (float(m["n_frames"]) * float(Z) * float(Y) * float(X))
        return out

    def pi_crop(self) -> dict:
        """Per-crop count prior on the DOWNSAMPLED [1,4,4] output grid. Training-time
        aggregate regulariser only: never a runtime input, never a routing feature."""
        out = {}
        for c in self.crops():
            m = self._crop_manifest(c)
            Z, Y, X = m["grid_zyx"]
            denom = float(m["n_frames"]) * float(Z) * float(Y) * float(X)
            out[c] = float(m["estimated_number_of_nodes"]) / denom
        return out

    def n_est(self) -> dict:
        return {c: float(self._crop_manifest(c)["estimated_number_of_nodes"])
                for c in self.crops()}

    def subset(self, mask: np.ndarray) -> "Corpus":
        mask = np.asarray(mask, dtype=bool)
        return Corpus(
            X=self.X[mask], X_max=self.X_max[mask], kind=self.kind[mask],
            crop=self.crop[mask], family=self.family[mask], t=self.t[mask],
            logit=self.logit[mask], encoder_split=self.encoder_split[mask],
            dist_to_nearest_gt_um=(None if self.dist_to_nearest_gt_um is None
                                   else self.dist_to_nearest_gt_um[mask]),
            manifest=self.manifest, feature_source=self.feature_source)


def family_of(crop: str) -> str:
    return str(crop).split("_")[0]


def validate_manifest(manifest: dict, *,
                      expect_encode_calls: int | None = None) -> dict:
    """Hard gate on the v6 manifest. Missing keys BLOCK; they never warn.

    `n_views` is REFUSED, not merely absent. Conflating "how many times encode() was
    called" with "how many distinct spatial views that produced" is the exact error that
    made a whole lane report a four-view TTA for a block that makes 8 encode calls over 7
    distinct views. v6 records the two separately and so must every consumer.
    """
    if "n_views" in manifest:
        raise ContractError(
            "manifest carries the retired field `n_views`. It conflates the averaging "
            "divisor with the number of distinct spatial views, which differ here (8 vs "
            "7). Emit `n_encode_calls` and `n_distinct_views` instead.")
    missing = [k for k in V6_REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ContractError(f"manifest missing top-level keys {missing}")
    n_encode_calls = int(manifest["n_encode_calls"])
    n_distinct_views = int(manifest["n_distinct_views"])
    views = list(manifest["tta_view_set"])
    if len(views) != n_encode_calls:
        raise ContractError(
            f"tta_view_set lists {len(views)} entries but n_encode_calls="
            f"{n_encode_calls}; the view list is per-CALL, so they must agree.")
    # The exporter records call labels, not canonical spatial permutations.  In the
    # deployed block ``rot90(k=1) -> transpose(y,x)`` is algebraically identical to
    # ``flip_x`` (locked by tests/test_d1_v6_export.py).  Canonicalise that known alias
    # before checking the measured 8-call/7-view collision; otherwise a truthful v6
    # artifact is rejected merely because the duplicate calls have different labels.
    canonical_views = [
        "flip_x" if str(v) == "rot90_k1_then_transpose_yx" else str(v)
        for v in views
    ]
    if len(set(canonical_views)) != n_distinct_views:
        raise ContractError(
            f"tta_view_set holds {len(set(canonical_views))} canonical spatial views but "
            f"n_distinct_views={n_distinct_views}. The duplicate-view collision is the "
            f"measured property of this block; a manifest that miscounts it cannot be "
            f"used to reason about the average.")
    if n_distinct_views > n_encode_calls:
        raise ContractError(
            f"n_distinct_views={n_distinct_views} exceeds n_encode_calls="
            f"{n_encode_calls}; a view cannot appear without being encoded.")
    if expect_encode_calls is not None and n_encode_calls != expect_encode_calls:
        raise ContractError(
            f"manifest says n_encode_calls={n_encode_calls} but the kernel's TTA block "
            f"averages {expect_encode_calls}. A TTA-mean feature averaged over a "
            f"DIFFERENT view set than the logits is not the detector representation.")
    problems = {c: [k for k in V6_REQUIRED_CROP_KEYS if k not in m]
                for c, m in manifest["crops"].items()}
    problems = {c: v for c, v in problems.items() if v}
    if problems:
        raise ContractError(f"per-crop manifest keys missing: {problems}")
    return {"n_crops": len(manifest["crops"]), "n_encode_calls": n_encode_calls,
            "n_distinct_views": n_distinct_views,
            "tta_view_set": manifest["tta_view_set"]}


def load_basis(audit_dir, *, encoder_split: int | None = None,
               expect_encode_calls: int | None = None) -> Corpus:
    """Load ONE checkpoint basis: a v6 audit directory whose rows were all encoded by a
    single checkpoint. `encoder_split` overrides the manifest's own declaration."""
    import polars as pl  # noqa: PLC0415

    audit_dir = pathlib.Path(audit_dir)
    man_p = audit_dir / "d1_manifest.json"
    if not man_p.exists():
        raise ContractError(f"no d1_manifest.json in {audit_dir}")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    validate_manifest(manifest, expect_encode_calls=expect_encode_calls)

    crops = sorted(c for c, m in manifest["crops"].items()
                   if m.get("status", "complete") == "complete")
    if not crops:
        raise ContractError("no complete crops; refusing to fit on a partial export")

    frames, xs, xm, encs = [], [], [], []
    for c in crops:
        gt_p = audit_dir / f"{c}{V6_REQUIRED_FEATURE_FILES[0]}"
        mx_p = audit_dir / f"{c}{V6_REQUIRED_FEATURE_FILES[1]}"
        if not gt_p.exists() or not mx_p.exists():
            raise ContractError(
                f"{c}: v6 TTA-mean features absent ({gt_p.name}). The v5 "
                f"__feat_gt.npy is an IDENTITY-VIEW feature and cannot reproduce the "
                f"deployed post-TTA logit. Refusing to substitute it.")
        rows = pl.read_parquet(audit_dir / f"{c}__rows.parquet")
        # Checked PER CROP, before any concat: trap 21 means a column present in only
        # some crops is silently dropped rather than raising.
        for col in V5_ABSENT_COLUMNS:
            if col in rows.columns:
                raise ContractError(
                    f"{c}: column {col!r} is present but the frozen contract says it "
                    f"does not exist; whatever is being read is not what the v5 script "
                    f"believed it was reading")
        frames.append(rows)
        xs.append(np.load(gt_p))
        xm.append(np.load(mx_p))
        es = (encoder_split if encoder_split is not None
              else manifest["crops"][c].get("encoder_split",
                                            manifest["crops"][c]["split"]))
        encs.append(np.full(rows.height, int(es), dtype=np.int64))

    cols0 = set(frames[0].columns)
    for c, f in zip(crops, frames):
        if set(f.columns) != cols0:
            raise ContractError(
                f"{c}: row schema differs from the first crop (missing "
                f"{sorted(cols0 - set(f.columns))}, extra "
                f"{sorted(set(f.columns) - cols0)}). Trap 21.")
    # Parquet schemas can contain the same named fields in different physical order when
    # optional K-list keys first appear on different rows.  That is not schema drift, but
    # Polars vertical concat is positional and will otherwise fail (or, in older versions,
    # misalign fields).  Canonicalise to the first crop only after exact set equality.
    frames = [f.select(frames[0].columns) for f in frames]
    df = pl.concat(frames, how="vertical_relaxed")
    X = np.concatenate(xs).astype(np.float32, copy=False)
    Xm = np.concatenate(xm).astype(np.float32, copy=False)
    if X.shape[0] != df.height:
        raise ContractError(f"feature/row mismatch: {X.shape[0]} vs {df.height}")
    crop = df["dataset"].to_numpy().astype(str)
    return Corpus(
        X=X, X_max=Xm, kind=df["kind"].to_numpy().astype(str), crop=crop,
        family=np.array([family_of(c) for c in crop]),
        t=df["t"].to_numpy().astype(np.int64),
        logit=df["logit"].to_numpy().astype(np.float64),
        encoder_split=np.concatenate(encs),
        dist_to_nearest_gt_um=(df["dist_to_nearest_gt_um"].to_numpy().astype(np.float64)
                               if "dist_to_nearest_gt_um" in df.columns else None),
        manifest=manifest, feature_source="tta_mean")


def load_factorial(root, *, expect_encode_calls: int | None = None) -> Corpus:
    """Load the 2x2 checkpoint x family export.

    Layout: `<root>/basis_0/` and `<root>/basis_1/`, each a v6 audit dir holding BOTH
    families encoded by that split's checkpoint. A single-basis directory is accepted and
    loaded, but `run_direction` will then block: within a routed-only export each basis
    contains only its held-out family, so there is no source family to fit on.
    """
    root = pathlib.Path(root)
    basis_dirs = sorted(p for p in root.glob("basis_*") if (p / "d1_manifest.json").exists())
    if not basis_dirs:
        return load_basis(root, expect_encode_calls=expect_encode_calls)
    parts = []
    for p in basis_dirs:
        split = int(p.name.split("_")[-1])
        parts.append(load_basis(p, encoder_split=split,
                                expect_encode_calls=expect_encode_calls))
    merged_manifest = {"tta_view_set": parts[0].manifest["tta_view_set"],
                       "n_encode_calls": parts[0].manifest["n_encode_calls"],
                       "n_distinct_views": parts[0].manifest["n_distinct_views"],
                       "crops": {}}
    for p in parts:
        merged_manifest["crops"].update(p.manifest["crops"])
    return Corpus(
        X=np.concatenate([p.X for p in parts]),
        X_max=np.concatenate([p.X_max for p in parts]),
        kind=np.concatenate([p.kind for p in parts]),
        crop=np.concatenate([p.crop for p in parts]),
        family=np.concatenate([p.family for p in parts]),
        t=np.concatenate([p.t for p in parts]),
        logit=np.concatenate([p.logit for p in parts]),
        encoder_split=np.concatenate([p.encoder_split for p in parts]),
        dist_to_nearest_gt_um=(
            np.concatenate([p.dist_to_nearest_gt_um for p in parts])
            if all(p.dist_to_nearest_gt_um is not None for p in parts) else None),
        manifest=merged_manifest, feature_source="tta_mean")


def assert_same_basis(head: LinearHead, corpus: Corpus, mask: np.ndarray,
                      *, label: str = "") -> None:
    """C1. 44b6 sits in split-0's basis and 6bba in split-1's, and the bases are
    independent (rel-L2 1.41542 ~ sqrt(2)). A head fitted in one may never be applied in
    the other; this makes that path raise instead of returning a number."""
    if head.basis_split is None:
        raise BasisError(f"{label}: head carries no basis tag; refusing to apply it")
    seen = set(int(v) for v in np.unique(corpus.encoder_split[mask]))
    if seen != {int(head.basis_split)}:
        raise BasisError(
            f"{label}: head lives in checkpoint basis {head.basis_split} but the rows it "
            f"is being applied to were encoded by {sorted(seen)}. The two 32-D bases are "
            f"independent (rel-L2 ~ sqrt(2)); a coefficient vector has no shared meaning "
            f"across them.")


# ============================================================================ fitting
@dataclass
class FitSpec:
    """Everything that determines the fitted bytes. Hash this to fingerprint a run."""

    l2: float = 1.0
    anchor: np.ndarray | None = None
    max_iter: int = 80
    tol: float = 1e-12
    max_halvings: int = 30
    firth: str = "auto"                  # "auto" | "always" | "never"
    l2_mle_epsilon: float = 1e-8
    beta_max: float = 1e3
    chunk: int = 1 << 18

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "anchor"}
        d["anchor"] = (None if self.anchor is None
                       else [float(v) for v in np.ravel(self.anchor)])
        return d


@dataclass
class SeparationDiagnosis:
    kind: str                            # "none" | "quasi" | "complete"
    min_margin: float
    beta_norm: float
    diverged: bool
    n_iter: int

    @property
    def separated(self) -> bool:
        return self.kind != "none"


@dataclass
class FitResult:
    beta: np.ndarray
    objective: float
    grad_inf_norm: float
    newton_decrement: float
    n_iter: int
    converged: bool
    firth_engaged: bool
    separation: SeparationDiagnosis | None = None
    term_report: dict = field(default_factory=dict)


def _accumulate(X, y, s, offset, beta, chunk, want_hess=True, firth_hat=None):
    """One chunked pass: objective, gradient, Hessian of the weighted BCE part."""
    n, d = X.shape
    w, b = beta[:d], float(beta[d])
    obj = 0.0
    grad = np.zeros(d + 1, dtype=np.float64)
    hess = np.zeros((d + 1, d + 1), dtype=np.float64) if want_hess else None
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        Xc = X[lo:hi].astype(np.float64, copy=False)
        eta = Xc @ w + b
        if offset is not None:
            eta = eta + offset[lo:hi]
        sc, yc = s[lo:hi], y[lo:hi]
        obj += float(sc @ (_softplus(eta) - yc * eta))
        p = _sigmoid(eta)
        r = sc * (p - yc)
        if firth_hat is not None:
            r = r + sc * firth_hat[lo:hi] * (p - 0.5)
        grad[:d] += Xc.T @ r
        grad[d] += float(r.sum())
        if want_hess:
            v = sc * p * (1.0 - p)
            Xv = Xc * v[:, None]
            hess[:d, :d] += Xc.T @ Xv
            col = Xv.sum(axis=0)
            hess[:d, d] += col
            hess[d, :d] += col
            hess[d, d] += float(v.sum())
    return obj, grad, hess


def _hat_diagonal(X, s, offset, beta, info_inv, chunk):
    n, d = X.shape
    h = np.empty(n, dtype=np.float64)
    w, b = beta[:d], float(beta[d])
    A, c, dd = info_inv[:d, :d], info_inv[:d, d], float(info_inv[d, d])
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        Xc = X[lo:hi].astype(np.float64, copy=False)
        eta = Xc @ w + b
        if offset is not None:
            eta = eta + offset[lo:hi]
        p = _sigmoid(eta)
        q = np.einsum("ij,jk,ik->i", Xc, A, Xc, optimize=True) + 2.0 * (Xc @ c) + dd
        h[lo:hi] = s[lo:hi] * p * (1.0 - p) * q
    return h


def _logdet_info(hess: np.ndarray) -> float:
    k = hess.shape[0]
    jit = 1e-12 * max(1.0, float(np.trace(hess)) / k)
    sign, ld = np.linalg.slogdet(hess + jit * np.eye(k))
    return float(ld) if sign > 0 else -math.inf


def _newton(X, y, s, offset, *, l2, anchor, terms, firth, max_iter, tol, chunk,
            max_halvings=30, beta_abort=np.inf):
    """Damped Newton / IRLS: exact gradient, exact (or Gauss-Newton PSD) Hessian,
    Cholesky solve, backtracking on the objective. No RNG, no shuffling, no early stop —
    two runs on the same inputs return bit-identical betas."""
    d = X.shape[1]
    k = d + 1
    beta = np.zeros(k, dtype=np.float64)
    anchor_k = np.zeros(k, dtype=np.float64)
    if anchor is not None:
        anchor_k[:d] = np.asarray(anchor, dtype=np.float64).reshape(-1)
    pen = np.zeros(k)
    pen[:d] = 1.0                          # the bias is NEVER penalised

    def _pen(b_):
        dd = pen * (b_ - anchor_k)
        return 0.5 * l2 * float(dd @ dd), l2 * dd

    def obj_only(b_):
        if firth:
            _, _, hb = _accumulate(X, y, s, offset, b_, chunk, want_hess=True)
            o = _accumulate(X, y, s, offset, b_, chunk, want_hess=False)[0]
            o -= 0.5 * _logdet_info(hb)
        else:
            o = _accumulate(X, y, s, offset, b_, chunk, want_hess=False)[0]
        o += _pen(b_)[0]
        for t in terms:
            o += t.value_grad_hess(b_)[0]
        return o

    def full(b_):
        hat = None
        if firth:
            _, _, hb = _accumulate(X, y, s, offset, b_, chunk, want_hess=True)
            hat = _hat_diagonal(X, s, offset, b_,
                                np.linalg.inv(hb + 1e-12 * np.eye(k)), chunk)
        obj, grad, hess = _accumulate(X, y, s, offset, b_, chunk, True, hat)
        if firth:
            obj = obj - 0.5 * _logdet_info(hess)
        po, pg = _pen(b_)
        obj += po
        grad = grad + pg
        hess = hess + l2 * np.diag(pen)
        rep = {}
        for t in terms:
            tv, tg, th = t.value_grad_hess(b_)
            obj += tv
            grad = grad + tg
            hess = hess + th
            rep[t.name] = {"value": float(tv), "grad_l2": float(np.linalg.norm(tg))}
        return obj, grad, hess, rep

    obj, grad, hess, rep = full(beta)
    converged, decrement, n_iter = False, float("inf"), 0
    for n_iter in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(hess + 1e-12 * np.eye(k), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        # Newton decrement: a SCALE-INVARIANT stopping test. An absolute gradient
        # tolerance is wrong here because the count/GE term carries a gradient weighted
        # by n_c and by dKL/dq, which diverges as pi -> 0; the fit would be reported
        # non-converged forever while sitting on the optimum.
        decrement = 0.5 * float(grad @ step)
        if 0.0 <= decrement <= tol * max(1.0, abs(obj)):
            converged = True
            break
        t, ok = 1.0, False
        for _ in range(max_halvings):
            cand = beta - t * step
            c_obj = obj_only(cand)
            if np.isfinite(c_obj) and c_obj <= obj + 1e-14 * abs(obj):
                beta, (obj, grad, hess, rep) = cand, full(cand)
                ok = True
                break
            t *= 0.5
        if not ok:
            converged = decrement <= 1e-8 * max(1.0, abs(obj))
            break
        if float(np.linalg.norm(beta)) > beta_abort:
            break
        if np.abs(t * step).max() < tol:
            converged = True
            break
    return FitResult(beta=beta, objective=float(obj),
                     grad_inf_norm=float(np.abs(grad).max()),
                     newton_decrement=float(decrement), n_iter=n_iter,
                     converged=converged, firth_engaged=bool(firth), term_report=rep)


def diagnose_separation(X, y, s, offset=None, *, beta_max=1e3, chunk=1 << 18,
                        margin_eps=1e-9) -> SeparationDiagnosis:
    """Complete separation exists iff some beta gives (2y-1)*eta > 0 on every active row
    (Albert & Anderson 1984). Under separation the unpenalised Newton iterate diverges
    ALONG such a direction, so we run it, normalise the iterate, and test that unit
    direction's margins — a direct test of the definition, not a proxy on ||beta||."""
    act = s > 0
    Xa, ya, sa = X[act], np.asarray(y, float)[act], np.asarray(s, float)[act]
    oa = None if offset is None else np.asarray(offset, float)[act]
    res = _newton(Xa, ya, sa, oa, l2=0.0, anchor=None, terms=(), firth=False,
                  max_iter=60, tol=1e-14, chunk=chunk, beta_abort=beta_max * 10)
    nrm = float(np.linalg.norm(res.beta))
    diverged = (nrm > beta_max) or (not res.converged)
    if nrm < 1e-12:
        return SeparationDiagnosis("none", 0.0, nrm, diverged, res.n_iter)
    unit = res.beta / nrm
    d = Xa.shape[1]
    eta = Xa.astype(np.float64) @ unit[:d] + unit[d]
    if oa is not None:
        eta = eta + oa / nrm
    margin = (2.0 * ya - 1.0) * eta
    mn = float(margin.min()) if margin.size else 0.0
    # Deliberately NOT gated on `diverged`: under complete separation the fitted
    # probabilities saturate to exactly 0/1 in float64, the gradient underflows, and any
    # step-based test reports "converged" at a large but finite beta. That is exactly the
    # case Firth exists for.
    eta_b = Xa.astype(np.float64) @ res.beta[:d] + res.beta[d]
    if oa is not None:
        eta_b = eta_b + oa
    p = _sigmoid(eta_b)
    saturated = float(np.minimum(p, 1.0 - p).max())
    if mn > margin_eps:
        kind = "complete"
    elif mn >= -margin_eps and saturated < 1e-8:
        kind = "quasi"
    else:
        kind = "none"
    return SeparationDiagnosis(kind, mn, nrm, diverged, res.n_iter)


def fit_head(X, y, s, *, offset=None, spec: FitSpec | None = None,
             terms: tuple = ()) -> FitResult:
    """Fit a (d+1)-parameter head. Deterministic, full-batch, numpy only.

    FIRTH ENGAGEMENT RULE, and this is the only rule:
      "never"  -> never.  "always" -> always.
      "auto"   -> engage IFF l2 <= l2_mle_epsilon (no ridge, so the penalised MLE is not
                  guaranteed to exist) AND diagnose_separation returns complete/quasi.
      With l2 above that epsilon the ridge already makes the objective strictly convex and
      coercive; adding Firth would change the estimand for no numerical reason.
    """
    spec = spec or FitSpec()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D design, got {X.shape}")
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if not (X.shape[0] == y.shape[0] == s.shape[0]):
        raise ValueError(f"row mismatch: X {X.shape[0]}, y {y.shape[0]}, s {s.shape[0]}")
    if np.any(s < 0):
        raise ValueError("negative sample weights are not a supported reweighting")
    if s.sum() <= 0:
        raise ValueError("all sample weights are zero: this arm has no data")
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64).reshape(-1)
        if offset.shape[0] != X.shape[0]:
            raise ValueError("offset length must match X")

    sep, use_firth = None, spec.firth == "always"
    if spec.firth == "auto" and spec.l2 <= spec.l2_mle_epsilon:
        sep = diagnose_separation(X, y, s, offset, beta_max=spec.beta_max,
                                  chunk=spec.chunk)
        use_firth = sep.separated
    res = _newton(X, y, s, offset, l2=spec.l2, anchor=spec.anchor, terms=tuple(terms),
                  firth=use_firth, max_iter=spec.max_iter, tol=spec.tol,
                  chunk=spec.chunk, max_halvings=spec.max_halvings)
    res.separation = sep
    return res


# ============================================================================ terms
@dataclass
class CountGETerm:
    """H2 · Topaz-style generalized expectation on the per-crop UNIFORM voxel sample.

        A(beta) = gamma * SUM_c n_c * KL( pi_c || q_c(beta) )
        q_c(beta) = mean over crop c's uniform rows of sigmoid(eta_i)

    `pi_crop` is MANDATORY and a null RAISES. That is the byte-identical-arms defect:
    `build_arm(..., pi_crop=None)` silently skipped this branch and H2 became H1.
    Units (ROADMAP_DETECTION_ADDENDUM): estimated_number_of_nodes / (n_frames * Z_out *
    Y_out * X_out) on the DOWNSAMPLED [1,4,4] grid, evaluated on UNIFORM rows only, or
    the constraint measures the sampler rather than the field.
    """

    X: np.ndarray
    crop_of_row: np.ndarray
    uniform_mask: np.ndarray
    pi_crop: dict
    gamma: float = 1.0
    offset: np.ndarray | None = None
    prior_multiplier: float = 1.0
    name: str = "count_ge"

    def __post_init__(self):
        if self.pi_crop is None:
            raise MissingCapability(
                "CountGETerm requires pi_crop; a null count prior makes H2 identical to "
                "H1, which is the byte-identical-arms defect. Supply {crop: "
                "estimated_number_of_nodes / (n_frames*Z_out*Y_out*X_out)}.")
        if not isinstance(self.pi_crop, dict) or not self.pi_crop:
            raise MissingCapability("pi_crop must be a non-empty {crop: pi} mapping")
        if self.gamma <= 0:
            raise MissingCapability(f"CountGETerm gamma must be > 0, got {self.gamma}")
        um = np.asarray(self.uniform_mask, dtype=bool)
        if not um.any():
            raise MissingCapability("CountGETerm has no uniform rows to constrain")
        groups = {}
        for c in np.unique(self.crop_of_row[um]):
            key = str(c)
            if key not in self.pi_crop:
                raise MissingCapability(f"pi_crop is missing crop {key!r}")
            pi = float(self.pi_crop[key]) * float(self.prior_multiplier)
            if not (0.0 < pi < 1.0):
                raise MissingCapability(
                    f"pi_crop[{key!r}] * multiplier = {pi} is outside (0, 1); it must be "
                    f"a per-OUTPUT-GRID-VOXEL rate, not a per-movie count")
            groups[key] = (np.flatnonzero(um & (self.crop_of_row == c)), pi)
        self._groups = groups

    def value_grad_hess(self, beta):
        k = self.X.shape[1] + 1
        val, grad, hess = 0.0, np.zeros(k), np.zeros((k, k))
        for _c, (idx, pi) in self._groups.items():
            n_c = float(idx.size)
            p = _sigmoid(_eta(self.X, idx, beta, self.offset))
            q = min(max(float(p.mean()), 1e-12), 1.0 - 1e-12)
            val += self.gamma * n_c * (pi * np.log(pi / q)
                                       + (1.0 - pi) * np.log((1.0 - pi) / (1.0 - q)))
            dkl = -pi / q + (1.0 - pi) / (1.0 - q)
            d2kl = pi / (q * q) + (1.0 - pi) / ((1.0 - q) ** 2)
            dq = _xt_sum(self.X, idx, (p * (1.0 - p)) / n_c)
            grad += self.gamma * n_c * dkl * dq
            hess += self.gamma * n_c * d2kl * np.outer(dq, dq)      # Gauss-Newton, PSD
        return float(val), grad, hess

    def self_test(self, beta) -> dict:
        v, g, h = self.value_grad_hess(np.asarray(beta, dtype=np.float64))
        ev = np.linalg.eigvalsh((h + h.T) / 2.0)
        return {"term": self.name, "value": v, "grad_l2": float(np.linalg.norm(g)),
                "hess_min_eig": float(ev.min()), "n_groups": len(self._groups),
                "active": bool(abs(v) > 0 and np.linalg.norm(g) > 0)}


@dataclass
class TemporalTerm:
    """H3 · temporal pseudo-positives + temporal logit consistency. Both MANDATORY.

        A(beta) = gamma_pp   * SUM_{j in P} u_j * [ softplus(eta_j) - eta_j ]
                + gamma_pair * SUM_{(a,b)}  0.5 * v_ab * (eta_a - eta_b)^2

    Pseudo-labels are POSITIVES ONLY: no negative-label path exists in this term, so
    "unmatched voxels are not negatives" holds by construction rather than by discipline.
    Held-out-family rows are rejected AT CONSTRUCTION, so a fold-dishonest path is
    impossible rather than merely unused.
    """

    X: np.ndarray
    pseudo_pos_idx: np.ndarray
    pseudo_pos_w: np.ndarray
    pair_idx: np.ndarray
    pair_w: np.ndarray
    gamma_pp: float = 1.0
    gamma_pair: float = 1.0
    offset: np.ndarray | None = None
    allowed_rows: np.ndarray | None = None
    name: str = "temporal"

    def __post_init__(self):
        if self.pseudo_pos_idx is None or self.pair_idx is None:
            raise MissingCapability(
                "TemporalTerm requires BOTH a pseudo-positive table and a consistency "
                "pair table. H3/H4 are BLOCKED without them, not reduced to H1.")
        self.pseudo_pos_idx = np.asarray(self.pseudo_pos_idx, dtype=np.int64).reshape(-1)
        self.pair_idx = np.asarray(self.pair_idx, dtype=np.int64).reshape(-1, 2)
        self.pseudo_pos_w = np.asarray(self.pseudo_pos_w, dtype=np.float64).reshape(-1)
        self.pair_w = np.asarray(self.pair_w, dtype=np.float64).reshape(-1)
        if self.pseudo_pos_idx.size == 0 or self.pair_idx.shape[0] == 0:
            raise MissingCapability("temporal tables are empty; H3 is BLOCKED, not emitted")
        if self.pseudo_pos_w.shape != self.pseudo_pos_idx.shape:
            raise MissingCapability("pseudo-positive weights must match indices")
        if self.pair_w.shape[0] != self.pair_idx.shape[0]:
            raise MissingCapability("pair weights must match pair rows")
        if (self.pseudo_pos_w <= 0).all() and (self.pair_w <= 0).all():
            raise MissingCapability("temporal weights are all zero; the term is inert")
        if self.gamma_pp <= 0 and self.gamma_pair <= 0:
            raise MissingCapability("both temporal gammas are zero; the term is inert")
        if self.allowed_rows is not None:
            allowed = np.asarray(self.allowed_rows, dtype=bool)
            bad = ~allowed[self.pseudo_pos_idx]
            if bad.any():
                raise MissingCapability(
                    f"{int(bad.sum())} pseudo-positives fall outside the source-family "
                    f"rows: a target-family path must be structurally impossible")
            badp = ~allowed[self.pair_idx].all(axis=1)
            if badp.any():
                raise MissingCapability(
                    f"{int(badp.sum())} temporal pairs touch target-family rows")

    def value_grad_hess(self, beta):
        d = self.X.shape[1]
        k = d + 1
        val, grad, hess = 0.0, np.zeros(k), np.zeros((k, k))
        if self.gamma_pp > 0:
            idx, u = self.pseudo_pos_idx, self.pseudo_pos_w
            e = _eta(self.X, idx, beta, self.offset)
            val += self.gamma_pp * float(u @ (_softplus(e) - e))
            p = _sigmoid(e)
            grad += self.gamma_pp * _xt_sum(self.X, idx, u * (p - 1.0))
            v = u * p * (1.0 - p)
            Xc = self.X[idx].astype(np.float64, copy=False)
            Xv = Xc * v[:, None]
            hess[:d, :d] += self.gamma_pp * (Xc.T @ Xv)
            col = self.gamma_pp * Xv.sum(axis=0)
            hess[:d, d] += col
            hess[d, :d] += col
            hess[d, d] += self.gamma_pp * float(v.sum())
        if self.gamma_pair > 0:
            a, b = self.pair_idx[:, 0], self.pair_idx[:, 1]
            D = (self.X[a].astype(np.float64, copy=False)
                 - self.X[b].astype(np.float64, copy=False))     # bias cancels
            dd = D @ beta[:d]
            if self.offset is not None:
                dd = dd + (self.offset[a] - self.offset[b])
            val += self.gamma_pair * 0.5 * float(self.pair_w @ (dd * dd))
            grad[:d] += self.gamma_pair * (D.T @ (self.pair_w * dd))
            hess[:d, :d] += self.gamma_pair * (D.T @ (D * self.pair_w[:, None]))
        return float(val), grad, hess

    def self_test(self, beta) -> dict:
        v, g, h = self.value_grad_hess(np.asarray(beta, dtype=np.float64))
        ev = np.linalg.eigvalsh((h + h.T) / 2.0)
        return {"term": self.name, "value": v, "grad_l2": float(np.linalg.norm(g)),
                "hess_min_eig": float(ev.min()),
                "n_pseudo_pos": int(self.pseudo_pos_idx.size),
                "n_pairs": int(self.pair_idx.shape[0]),
                "active": bool(abs(v) > 0 and np.linalg.norm(g) > 0)}


# ---------------------------------------------------------------- weight builders
def kind_weights(kind, *, uniform_weight: float, subthr_weight: float = 0.0):
    """Base exposure profile. `subthr_localmax` defaults to weight ZERO: a sub-threshold
    local maximum is an UNLABELLED voxel, very plausibly a real unannotated nucleus, and
    calling it a negative is the exact supervision defect M2 exists to remove."""
    kind = np.asarray(kind)
    w = np.zeros(kind.shape[0], dtype=np.float64)
    w[kind == "gt_centre"] = 1.0
    w[kind == "uniform"] = float(uniform_weight)
    w[kind == "subthr_localmax"] = float(subthr_weight)
    return w


def local_mask_weights(base, dist_um, *, radius_um: float, is_positive):
    """Linajea-style local mask: OUTSIDE `radius_um` of an annotated centre, weight 0.
    Positives always keep their weight. A mask derived from `kind` alone would be
    indistinguishable from the base exposure profile and is refused by `build_arms`."""
    dist_um = np.asarray(dist_um, dtype=np.float64)
    if not np.isfinite(dist_um).any():
        raise MissingCapability("local mask needs a finite per-row distance-to-nearest-GT")
    w = np.array(base, dtype=np.float64, copy=True)
    w[(dist_um > float(radius_um)) & (~np.asarray(is_positive, dtype=bool))] = 0.0
    return w


def exposure_balance(base, y, *, groups=None, target_ratio: float = 1.0):
    """H5: reweight POSITIVES ONLY so positive exposure equals `target_ratio` x negative.
    The loss functional is unchanged; only s_i on positive rows moves. Grouping by crop
    stops a handful of dense crops supplying all the positive exposure."""
    base = np.asarray(base, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = base.copy()
    groups = np.zeros(base.shape[0], dtype=np.int64) if groups is None else np.asarray(groups)
    for g in np.unique(groups):
        m = groups == g
        pos, neg = m & (y == 1), m & (y == 0)
        wp, wn = base[pos].sum(), base[neg].sum()
        if wp > 0 and wn > 0:
            w[pos] = base[pos] * (target_ratio * wn / wp)
    return w


def design_weights(base, kind, crop, sampling_rate: dict):
    """Horvitz-Thompson correction so the fitted INTERCEPT is population-calibrated.

    Positives are exported exhaustively (every annotated centre); `uniform` rows are a
    per-frame subsample of the output grid. Uncorrected, the intercept sits on the
    SAMPLED prior: at 1/4096 that is an overstatement of
    log(4096) = 8.3178 logits, comparable to the whole span from the population base rate
    to the deployed threshold at +3.4340. Every threshold derived from it would be wrong.
    """
    base = np.asarray(base, dtype=np.float64)
    kind, crop = np.asarray(kind), np.asarray(crop)
    w = base.copy()
    um = kind == "uniform"
    for c in np.unique(crop[um]):
        key = str(c)
        if key not in sampling_rate:
            raise MissingCapability(
                f"design_weights needs sampling_rate[{key!r}] = n_uniform / "
                f"(n_frames*Z_out*Y_out*X_out). Absent grid dims => raise, never guess: "
                f"the 1/{UNIFORM_SUBSAMPLE_DENOM} correction is "
                f"{HT_LOGIT_CORRECTION_1_4096:.4f} logits.")
        r = float(sampling_rate[key])
        if not (0.0 < r <= 1.0):
            raise MissingCapability(f"sampling_rate[{key!r}]={r} outside (0, 1]")
        sel = um & (crop == c)
        w[sel] = base[sel] / r
    return w


def ht_weights(kind, crop, sampling_rate: dict):
    """Pure Horvitz-Thompson population weights for DIAGNOSTICS (not for the loss).

    gt_centre -> 1 (exhaustive census); uniform -> 1/rate (probability sample of the
    grid); subthr_localmax -> 0. Sub-threshold maxima are neither labelled nor a
    probability sample of anything, so they belong in no population estimate.
    """
    return design_weights(kind_weights(kind, uniform_weight=1.0, subthr_weight=0.0),
                          kind, crop, sampling_rate)


def intercept_correction(rate: float) -> float:
    """Logits by which an uncorrected intercept overstates the population log-odds."""
    if not (0.0 < float(rate) <= 1.0):
        raise ValueError(f"rate {rate} outside (0, 1]")
    return -math.log(float(rate))


# ============================================================================ arms
@dataclass
class ArmSpec:
    name: str
    requires: tuple
    param_space: str                     # "full33" | "cal2"
    role: str                            # "hypothesis" | "control" | "reference"
    doc: str


REGISTRY = {
    "H0": ArmSpec("H0", ("checkpoint",), "full33", "reference",
                  "Deployed 33-parameter head from the ROUTED fold checkpoint. NOT "
                  "fitted, NO threshold selection, deployed threshold 0.96875."),
    "H1": ArmSpec("H1", ("row_mask",), "full33", "hypothesis",
                  "Linajea local masked loss. Requires a real per-row distance to the "
                  "nearest annotated centre and a radius."),
    "H2": ArmSpec("H2", ("row_mask", "pi_crop"), "full33", "hypothesis",
                  "H1 + implemented training-only count/GE constraint."),
    "H3": ArmSpec("H3", ("row_mask", "temporal"), "full33", "hypothesis",
                  "H1 + real temporal pseudo-positives and logit consistency."),
    "H4": ArmSpec("H4", ("row_mask", "pi_crop", "temporal"), "full33", "hypothesis",
                  "Full M2-CTPU: masked + count + temporal."),
    "H5": ArmSpec("H5", (), "full33", "control",
                  "Positive-exposure control. Loss functional unchanged, positives "
                  "reweighted only. Separates 'wrong negative supervision' from 'simply "
                  "fewer positives'. Not H0 and never a parity reference."),
    "CAL_ONLY": ArmSpec("CAL_ONLY", ("checkpoint",), "cal2", "control",
                        "Calibration-only: refit a*eta0 + c with w FROZEN at w_ckpt. "
                        "Monotone in the deployed score, so it cannot change ranking."),
    "LIN_HEAD": ArmSpec("LIN_HEAD", (), "full33", "control",
                        "Full 33-parameter linear refit on the deployed exposure "
                        "profile. The ranking-changing comparator."),
    "SHUF_LIN": ArmSpec("SHUF_LIN", (), "full33", "negative_control",
                        "LIN_HEAD on labels shuffled within crop. Must return a null or "
                        "the whole instrument is measuring leakage."),
    "SHUF_CAL": ArmSpec("SHUF_CAL", ("checkpoint",), "cal2", "negative_control",
                        "CAL_ONLY on shuffled labels. Supplies the calibration null "
                        "band, so learning the base rate cannot pass as a repair."),
}

DEFAULT_ARMS = ("H0", "CAL_ONLY", "LIN_HEAD", "SHUF_CAL", "SHUF_LIN",
                "H5", "H1", "H2", "H3", "H4")

# A shuffled arm is replicated over seeds and named `SHUF_LIN#3`. `arm_base` recovers the
# registry key. MEASURED (FIXTURE, split-1 direction, 10 seeds): one label-shuffled refit
# returns a target AUC with sd 0.148 — the single draw is NOT a null, the DISTRIBUTION is.
# Gating one draw against 0.5 +/- 0.05 fires spuriously in the majority of runs.
SHUFFLED_ARMS = ("SHUF_LIN", "SHUF_CAL")
# 19 replicates is the smallest count at which the add-one permutation p-value can reach
# alpha = 0.05: 1/(19+1) = 0.05. Fewer, and every arm fails the null-tail test for want of
# replicates rather than for want of signal, and the run reports LINEAR_PROBE_NULL for a
# reason unrelated to the data. `decide_verdict` raises rather than let that happen.
DEFAULT_N_SHUFFLES = 19


def arm_base(name: str) -> str:
    """`SHUF_LIN#3` -> `SHUF_LIN`. Registry lookups always go through this."""
    return str(name).split("#", 1)[0]


@dataclass
class Capabilities:
    """What the supplied data actually supports. Absence BLOCKS; it never degrades."""

    checkpoint_head: LinearHead | None = None
    dist_to_nearest_gt_um: np.ndarray | None = None
    mask_radius_um: float | None = None
    pi_crop: dict | None = None
    temporal_pseudo_pos_idx: np.ndarray | None = None
    temporal_pseudo_pos_w: np.ndarray | None = None
    temporal_pair_idx: np.ndarray | None = None
    temporal_pair_w: np.ndarray | None = None
    sampling_rate: dict | None = None

    def have(self, cap: str) -> bool:
        if cap == "checkpoint":
            return self.checkpoint_head is not None
        if cap == "row_mask":
            return (self.dist_to_nearest_gt_um is not None
                    and self.mask_radius_um is not None)
        if cap == "pi_crop":
            return bool(self.pi_crop)
        if cap == "temporal":
            return (self.temporal_pseudo_pos_idx is not None
                    and self.temporal_pair_idx is not None)
        raise KeyError(f"unknown capability {cap!r}")

    def missing_for(self, arm: str) -> list:
        return [c for c in REGISTRY[arm_base(arm)].requires if not self.have(c)]


@dataclass
class Arm:
    name: str
    design: np.ndarray                   # (N, d) — 32-D features or the 1-D eta0 column
    y: np.ndarray
    s: np.ndarray
    offset: np.ndarray | None
    param_space: str
    terms: tuple = ()
    spec: FitSpec = field(default_factory=FitSpec)
    head: LinearHead | None = None       # H0 only: pre-existing, never fitted
    fitted: bool = True
    notes: dict = field(default_factory=dict)

    def fingerprint(self, probe: np.ndarray) -> dict:
        """FULL objective value + gradient at a fixed probe beta: sample weights,
        offsets, auxiliary terms and ridge. A distinct label or config hash is NOT
        evidence that two arms differ; the objective is."""
        obj, grad, _ = _accumulate(self.design, self.y, self.s, self.offset, probe,
                                   self.spec.chunk, want_hess=False)
        term_vals = {}
        for t in self.terms:
            tv, tg, _ = t.value_grad_hess(probe)
            obj += tv
            grad = grad + tg
            term_vals[t.name] = float(tv)
        d = self.design.shape[1]
        dv = np.zeros(d + 1)
        dv[:d] = probe[:d] - (0.0 if self.spec.anchor is None
                              else np.asarray(self.spec.anchor).reshape(-1))
        obj += 0.5 * self.spec.l2 * float(dv @ dv)
        grad = grad + self.spec.l2 * dv
        return {"arm": self.name, "param_space": self.param_space,
                "objective": float(obj), "grad_l2": float(np.linalg.norm(grad)),
                "grad_sha256": hashlib.sha256(
                    np.round(grad, 9).astype(np.float64).tobytes()).hexdigest()[:32],
                "sum_weights": float(self.s.sum()),
                "n_active_rows": int((self.s > 0).sum()), "terms": term_vals}


def probe_beta(n_params: int, seed: int = 20260806) -> np.ndarray:
    """A fixed, non-degenerate probe point. Deterministic; never used for fitting."""
    return np.random.default_rng(seed + n_params).normal(0.0, 0.25, size=n_params)


SHUFFLE_SCOPES = ("global", "within_crop")


def shuffle_labels(y, crop, active, *, seed: int = 20260806, scope: str = "global"):
    """Permute labels among active rows. `scope` decides what survives the permutation.

    The MECHANISM `within_crop` leaves open: it preserves each crop's positive COUNT, and
    the exported row pool makes a crop's positive rate collinear with that crop's mean
    feature -- every `gt_centre` row is exported while `uniform` rows are a fixed-size
    per-frame subsample, so a crop with more nuclei has both a higher positive rate AND a
    mean feature pulled toward the nucleus direction. A 33-parameter refit can exploit
    that between-crop channel even though the labels carry no per-row information at all.
    `global` shuffling removes the per-crop rate variation as well, so it is the default
    and the scope the verdict gate consumes.

    SIZE NOT ESTABLISHED. An inherited docstring quoted `within_crop` at target AUC 0.581
    against `global` 0.528 (se 0.034) and attributed it to FIXTURE, 20 seeds. That does
    NOT reproduce on the fixture in `tests/test_d1f_probe.py` (19 seeds, split-1: global
    mean 0.4648 se 0.0379, within_crop mean 0.4833 se 0.0356 -- the two are not separated
    and neither is above chance). The gap is configuration-dependent and the numbers are
    withdrawn pending a measurement on the real v6 export. The MECHANISM stands and is
    why `global` is the default; the MAGNITUDE is unmeasured.
    """
    if scope not in SHUFFLE_SCOPES:
        raise ValueError(f"shuffle scope {scope!r} not in {SHUFFLE_SCOPES}")
    y = np.asarray(y, dtype=np.float64).copy()
    crop, active = np.asarray(crop), np.asarray(active, dtype=bool)
    rng = np.random.default_rng(seed)
    if scope == "global":
        idx = np.flatnonzero(active)
        y[idx] = y[idx][rng.permutation(idx.size)]
        return y
    for c in sorted(set(crop[active].tolist())):
        idx = np.flatnonzero(active & (crop == c))
        y[idx] = y[idx][rng.permutation(idx.size)]
    return y


def shuffle_labels_within_crop(y, crop, active, seed: int = 20260806):
    """Back-compatible alias. NOT a null on this row pool -- see `shuffle_labels`."""
    return shuffle_labels(y, crop, active, seed=seed, scope="within_crop")


def build_arms(corpus: Corpus, caps: Capabilities, *, source_mask: np.ndarray,
               arms=DEFAULT_ARMS, l2: float = 1.0,
               uniform_weight: float = DEPLOYED_NEG_WEIGHT, gamma_count: float = 1.0,
               gamma_pp: float = 1.0, gamma_pair: float = 1.0,
               prior_multiplier: float = 1.0, fit_spec: FitSpec | None = None,
               shuffle_scope: str = "global",
               n_shuffles: int = DEFAULT_N_SHUFFLES, shuffle_seed: int = 20260806):
    """Return (built, blocked, fingerprints). `source_mask` restricts every arm to
    SOURCE-family rows inside ONE checkpoint basis; the target family is never fitted on.

    Every `SHUF_*` arm is REPLICATED over `n_shuffles` seeds and emitted as `SHUF_LIN#k`.
    A single label-shuffled refit is NOT a null. MEASURED (FIXTURE, split-1 direction,
    10 seeds): the target AUC of one shuffled refit has sd 0.148, so a single draw lands
    outside 0.5 +/- 0.05 more often than not, and the "null" that `decide_verdict`
    subtracts would be one noisy draw. The null is the DISTRIBUTION over seeds, and the
    verdict gate consumes all of it.
    """
    X = corpus.X
    n = X.shape[0]
    source_mask = np.asarray(source_mask, dtype=bool)
    y = corpus.y
    base = kind_weights(corpus.kind, uniform_weight=uniform_weight, subthr_weight=0.0)
    base = base * source_mask                     # target family gets exactly zero weight
    if caps.sampling_rate is not None:
        base = design_weights(base, corpus.kind, corpus.crop, caps.sampling_rate)
    offset = (None if prior_multiplier == 1.0
              else np.full(n, float(np.log(prior_multiplier))))

    eta0 = None
    if caps.checkpoint_head is not None:
        eta0 = caps.checkpoint_head.logit(X, dtype="float64")

    if int(n_shuffles) < 2:
        raise ValueError(
            f"n_shuffles={n_shuffles}: a single label-shuffled refit is a draw from a "
            f"distribution with sd ~0.15 AUC, not a null. At least 2 seeds are required "
            f"so the null band is measured rather than assumed.")

    expanded = []
    for name in arms:
        if arm_base(name) in SHUFFLED_ARMS and "#" not in str(name):
            expanded.extend(f"{name}#{k}" for k in range(int(n_shuffles)))
        else:
            expanded.append(name)

    built, blocked = {}, {}
    for name in expanded:
        base_name = arm_base(name)
        if base_name in REFUSED_ARMS:
            blocked[name] = f"REFUSED — {REFUSED_ARMS[base_name]}"
            continue
        if base_name not in REGISTRY:
            blocked[name] = f"unknown arm {name!r}"
            continue
        miss = caps.missing_for(name)
        if miss:
            blocked[name] = (f"BLOCKED — missing capabilities {miss}. "
                             f"{REGISTRY[base_name].doc}")
            continue
        seed = shuffle_seed + int(name.split("#", 1)[1]) if "#" in name else shuffle_seed
        try:
            built[name] = _build_one(name, corpus, caps, X, y, base, offset, source_mask,
                                     eta0, l2, gamma_count, gamma_pp, gamma_pair,
                                     prior_multiplier, fit_spec, shuffle_scope, seed)
        except MissingCapability as exc:
            blocked[name] = f"BLOCKED — {exc}"
    fps = assert_arms_distinct(built)
    return built, blocked, fps


def _build_one(name, corpus, caps, X, y, base, offset, source_mask, eta0, l2,
               gamma_count, gamma_pp, gamma_pair, prior_multiplier, fit_spec,
               shuffle_scope="global", shuffle_seed=20260806):
    spec = FitSpec(**{**(fit_spec or FitSpec()).as_dict(),
                      "anchor": (fit_spec.anchor if fit_spec else None), "l2": l2})
    name_b = arm_base(name)

    if name_b == "H0":
        return Arm(name="H0", design=X, y=y, s=base, offset=offset, param_space="full33",
                   spec=spec, head=caps.checkpoint_head, fitted=False,
                   notes={"threshold_logit": DET_THRESHOLD_LOGIT,
                          "threshold_selected": False, "fitted": False,
                          "why": "H0 is the deployed head. Fitting it, or selecting its "
                                 "threshold, destroys the only parity reference there is."})

    if name_b in ("CAL_ONLY", "SHUF_CAL"):
        # 2-parameter space: eta = a*eta0 + c, w frozen at w_ckpt.
        design = eta0.reshape(-1, 1).astype(np.float64)
        shuf = name_b == "SHUF_CAL"
        yy = (shuffle_labels(y, corpus.crop, base > 0, seed=shuffle_seed,
                             scope=shuffle_scope) if shuf else y)
        return Arm(name=name, design=design, y=yy, s=base, offset=offset,
                   param_space="cal2", spec=FitSpec(**{**spec.as_dict(),
                                                      "anchor": None, "firth": "never"}),
                   notes={"parameterisation": "a*eta0 + c, w frozen at w_ckpt",
                          "n_free_params": 2,
                          "labels": (f"shuffled ({shuffle_scope})" if shuf else "real"),
                          "shuffle_scope": (shuffle_scope if shuf else None),
                          "shuffle_seed": (shuffle_seed if shuf else None),
                          "cannot_change_ranking": True})

    if name_b in ("LIN_HEAD", "SHUF_LIN"):
        shuf = name_b == "SHUF_LIN"
        yy = (shuffle_labels(y, corpus.crop, base > 0, seed=shuffle_seed,
                             scope=shuffle_scope) if shuf else y)
        return Arm(name=name, design=X, y=yy, s=base, offset=offset,
                   param_space="full33", spec=spec,
                   notes={"parameterisation": "full 33-parameter refit",
                          "labels": (f"shuffled ({shuffle_scope})" if shuf else "real"),
                          "shuffle_scope": (shuffle_scope if shuf else None),
                          "shuffle_seed": (shuffle_seed if shuf else None)})

    if name_b == "H5":
        s = exposure_balance(base, y, groups=corpus.crop, target_ratio=1.0)
        return Arm(name="H5", design=X, y=y, s=s, offset=offset, param_space="full33",
                   spec=spec, notes={"control": "positives reweighted only; loss "
                                                "functional unchanged", "grouped_by": "crop"})

    # H1-H4 all start from the REAL local mask.
    s = local_mask_weights(base, caps.dist_to_nearest_gt_um,
                           radius_um=caps.mask_radius_um, is_positive=corpus.is_positive)
    if np.allclose(s, base):
        raise MissingCapability(
            "the local mask zeroed no rows, so H1 is byte-identical to the base exposure "
            "profile. Either mask_radius_um exceeds every distance in the corpus or the "
            "distance column is degenerate. Refusing to emit a relabelled copy.")
    tset = []
    if name_b in ("H2", "H4"):
        tset.append(CountGETerm(
            X=X, crop_of_row=corpus.crop,
            uniform_mask=(corpus.kind == "uniform") & source_mask,
            pi_crop=caps.pi_crop, gamma=gamma_count, offset=offset,
            prior_multiplier=prior_multiplier))
    if name_b in ("H3", "H4"):
        tset.append(TemporalTerm(
            X=X, pseudo_pos_idx=caps.temporal_pseudo_pos_idx,
            pseudo_pos_w=caps.temporal_pseudo_pos_w, pair_idx=caps.temporal_pair_idx,
            pair_w=caps.temporal_pair_w, gamma_pp=gamma_pp, gamma_pair=gamma_pair,
            offset=offset, allowed_rows=source_mask))
    probe = probe_beta(X.shape[1] + 1)
    for t in tset:                       # per-term self test BEFORE any fitting
        st = t.self_test(probe)
        if not st["active"]:
            raise MissingCapability(f"term {t.name} is inert at the probe point: {st}")
        if st["hess_min_eig"] < -1e-8:
            raise MissingCapability(f"term {t.name} has a non-PSD Hessian block: {st}")
    return Arm(name=name, design=X, y=y, s=s, offset=offset, param_space="full33",
               spec=spec, terms=tuple(tset),
               notes={"mask_radius_um": caps.mask_radius_um,
                      "n_rows_masked_out": int(((base > 0) & (s == 0)).sum()),
                      "terms": [t.name for t in tset]})


def assert_arms_distinct(built: dict) -> dict:
    """Pairwise objective fingerprints at a fixed probe beta, within each parameter space.
    H0 is exempt: it is not fitted and has no objective."""
    fps, seen = {}, {}
    for k, a in built.items():
        if not a.fitted:
            continue
        fp = a.fingerprint(probe_beta(a.design.shape[1] + 1))
        fps[k] = fp
        key = (a.param_space, round(fp["objective"], 9), fp["grad_sha256"])
        if key in seen:
            raise MissingCapability(
                f"arms {seen[key]!r} and {k!r} have an IDENTICAL objective at the probe "
                f"point (obj={fp['objective']:.9f}, space={a.param_space}). They are one "
                f"arm wearing two labels — the exact defect that made H1/H2/H3/H4 "
                f"byte-identical in the v5 script. Refusing to emit both.")
        seen[key] = k
    return fps


def assert_fitted_heads_distinct(heads: dict) -> None:
    """Post-fit guard: no two EMITTED arms may deploy the same 33 parameters. Two arms
    with different objectives can still converge to the same point; that is still one
    result wearing two labels."""
    seen = {}
    for name, h in heads.items():
        sig = h.param_sha256()
        if sig in seen:
            raise MissingCapability(
                f"arms {seen[sig]!r} and {name!r} converged to IDENTICAL deployed "
                f"parameters ({sig}). Distinct objectives are not enough — the results "
                f"table would carry the same head twice.")
        seen[sig] = name


def fit_arm(arm: Arm, *, basis_split: int) -> tuple[LinearHead, FitResult | None,
                                                    np.ndarray]:
    """Fit one arm and return (deployed 33-param head, fit result, per-row score).

    H0 is returned unfitted, by contract. CAL_ONLY/SHUF_CAL fit 2 parameters and are then
    EXPANDED into the equivalent deployed head, w = a*w_ckpt, b = a*b_ckpt + c: a
    calibration of a linear head is itself a linear head, which is why it deploys at zero
    extra cost and also why it can never change ranking.
    """
    if not arm.fitted:
        if arm.head is None:
            raise MissingCapability("H0 has no checkpoint head")
        return arm.head, None, arm.head.logit(arm.design, dtype="float64")

    res = fit_head(arm.design, arm.y, arm.s, offset=arm.offset, spec=arm.spec,
                   terms=arm.terms)
    if arm.param_space == "cal2":
        a, c = float(res.beta[0]), float(res.beta[1])
        h0 = arm.notes.get("_h0")
        if h0 is None:
            raise MissingCapability("cal2 arm lost its reference head")
        head = LinearHead(w=a * h0.w, b=a * h0.b + c, name=arm.name,
                          basis_split=basis_split,
                          provenance={"kind": "calibration_only", "a": a, "c": c,
                                      "base_head": h0.name, "n_free_params": 2,
                                      "converged": res.converged})
        score = a * arm.design[:, 0] + c
        return head, res, score
    head = LinearHead.from_beta(res.beta, name=arm.name, basis_split=basis_split,
                                kind="fitted", arm=arm.name, l2=arm.spec.l2,
                                firth_engaged=res.firth_engaged,
                                converged=res.converged, n_iter=res.n_iter,
                                terms=[t.name for t in arm.terms], notes=arm.notes)
    return head, res, head.logit(arm.design, dtype="float64")


# ============================================================================ selection
def grouped_folds(crops, n_folds: int = 4, seed: int = 20260806) -> list:
    """Deterministic group k-fold BY CROP. Same inputs -> same folds, always."""
    uniq = sorted(set(np.asarray(crops).tolist()))
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(uniq) < n_folds:
        raise ValueError(f"{len(uniq)} crops cannot make {n_folds} grouped folds")
    order = np.array(uniq, dtype=object)[np.random.default_rng(seed).permutation(len(uniq))]
    return [set(order[i::n_folds].tolist()) for i in range(n_folds)]


def _sweep(score, y, w, grid):
    """Weighted TP/FP/FN at every grid threshold in one pass."""
    o = np.argsort(score, kind="stable")
    s_sorted = score[o]
    wp, wn = (w * y)[o], (w * (1 - y))[o]
    cp = np.concatenate([np.cumsum(wp[::-1])[::-1], [0.0]])
    cn = np.concatenate([np.cumsum(wn[::-1])[::-1], [0.0]])
    idx = np.searchsorted(s_sorted, grid, side="left")
    tp, fp = cp[idx], cn[idx]
    return tp, fp, cp[0] - tp


def _objective_values(objective, tp, fp, fn, *, total_neg, budget, allow_f1):
    if objective == "node_budget":
        # HT-weighted predicted-positive mass = an unbiased estimate of the number of
        # voxels the head would put over threshold across the whole grid. Compare it to
        # the crop's estimated node count; among admissible thresholds, maximise recall.
        rec = tp / np.maximum(tp + fn, 1e-12)
        return np.where((tp + fp) <= budget, rec, -1.0)
    if objective == "recall_at_fpr":
        fpr = fp / max(total_neg, 1e-12)
        rec = tp / np.maximum(tp + fn, 1e-12)
        return np.where(fpr <= budget, rec, -1.0)
    if objective == "f1":
        if not allow_f1:
            raise ValueError(
                "F1 on this row pool is degenerate: it is ~96% background rows standing "
                "in for a ~1e-5 population base rate, so its optimum is a statistic "
                "about the sampler. Pass allow_f1=True and stamp the result "
                "SAMPLED-BASIS-ONLY if you really want it.")
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    raise ValueError(f"unknown selection objective {objective!r}")


def nested_select_quantile(fit_on, score_with, *, corpus: Corpus, source_mask,
                           ht_w, n_est: dict, objective: str = "node_budget",
                           budget_multiple: float = 1.5, n_folds: int = 4,
                           seed: int = 20260806, n_grid: int = 121,
                           allow_f1: bool = False) -> dict:
    """Choose an operating QUANTILE out-of-fold, with a REFIT per inner fold.

    The v5 script selected its threshold on the very rows the coefficients were fitted
    on. Here, for each grouped-by-crop inner fold: refit on the other folds, score the
    held fold, sweep. The selected quantity is a quantile of the score distribution, not
    a raw score, so it transfers across refits; the final threshold is that quantile of
    the FINAL model's source-row scores.

    `fit_on(train_crops) -> LinearHead` and `score_with(head) -> (N,) scores`.
    """
    crops_all = corpus.crops(source_mask)
    if len(crops_all) < 2:
        raise ValueError(
            f"nested selection needs >= 2 source crops to group by; got "
            f"{len(crops_all)}. Selecting a threshold on a single crop is selection on "
            f"the fit rows, which is defect 6 of the v5 script.")
    # The outer same-family hold-out already removed one grouped fold of source crops, so
    # the inner loop can be asked for more folds than there are crops left. Clamping is
    # correct; silently selecting on fewer groups than requested is not, so the effective
    # count is reported.
    n_folds_eff = int(min(n_folds, len(crops_all)))
    folds = grouped_folds(np.array(crops_all), n_folds=n_folds_eff, seed=seed)
    qgrid = np.linspace(0.50, 1.0 - 1e-6, n_grid)
    per_fold, mat = [], []
    for f in folds:
        inner_train = sorted(set(crops_all) - f)
        if not inner_train:
            continue
        held = source_mask & np.isin(corpus.crop, sorted(f))
        act = held & (ht_w > 0)
        if not act.any() or (ht_w[act] * corpus.y[act]).sum() <= 0:
            continue
        head = fit_on(inner_train)
        sc = score_with(head)[act]
        yy, ww = corpus.y[act], ht_w[act]
        grid = np.quantile(sc, qgrid)
        tp, fp, fn = _sweep(sc, yy, ww, grid)
        budget = (budget_multiple * sum(n_est[c] for c in sorted(f))
                  if objective == "node_budget"
                  else budget_multiple)
        vals = _objective_values(objective, tp, fp, fn,
                                 total_neg=float((ww * (1 - yy)).sum()),
                                 budget=budget, allow_f1=allow_f1)
        mat.append(list(vals))
        per_fold.append({"crops": sorted(f), "n_inner_train_crops": len(inner_train),
                         "best_quantile": float(qgrid[int(np.argmax(vals))]),
                         "best_value": float(np.max(vals))})
    if not mat:
        raise ValueError("no inner fold carried positive weight")
    mean = np.mean(np.array(mat), axis=0)
    return {"quantile": float(qgrid[int(np.argmax(mean))]), "objective": objective,
            "budget_multiple": budget_multiple, "grouped_by": "crop",
            "n_folds_requested": int(n_folds), "n_folds_effective": n_folds_eff,
            "n_folds": len(per_fold), "per_fold": per_fold, "refit_per_fold": True,
            "BASIS": "out-of-fold, grouped by crop, coefficients refitted per fold"}


# ============================================================================ diagnostics
def _weighted_auc(score, y, w) -> float:
    """Weighted Mann-Whitney AUC with EXACT tie handling.

    Ties are not a corner case here: a shuffled-label refit under ridge collapses toward
    a constant score, and a per-row tie-blind cumulative sum would then return whatever
    the original row order happened to imply instead of the 0.5 that a null must produce.
    Ties are resolved per distinct score VALUE, so a fully constant score scores exactly
    0.5 and the negative control cannot pass or fail on row ordering.
    """
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    o = np.argsort(score, kind="stable")
    s, yy, ww = score[o], y[o], w[o]
    pos_w, neg_w = ww * yy, ww * (1 - yy)
    tot_p, tot_n = pos_w.sum(), neg_w.sum()
    if tot_p <= 0 or tot_n <= 0:
        return float("nan")
    # group boundaries of equal score values
    starts = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1]]))
    gid = np.cumsum(np.concatenate([[0], (s[1:] != s[:-1]).astype(np.int64)]))
    n_groups = starts.size
    grp_neg = np.bincount(gid, weights=neg_w, minlength=n_groups)
    grp_pos = np.bincount(gid, weights=pos_w, minlength=n_groups)
    strictly_below_neg = np.concatenate([[0.0], np.cumsum(grp_neg)[:-1]])
    conc = float(grp_pos @ (strictly_below_neg + 0.5 * grp_neg))
    return float(conc / (tot_p * tot_n))


def _weighted_ap(score, y, w) -> float:
    o = np.argsort(-score, kind="stable")
    yy, ww = y[o], w[o]
    tp, fp = np.cumsum(ww * yy), np.cumsum(ww * (1 - yy))
    tot_p = (ww * yy).sum()
    if tot_p <= 0:
        return float("nan")
    prec, rec = tp / np.maximum(tp + fp, 1e-12), tp / tot_p
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def ranking_diagnostics(score, y, w) -> dict:
    """RANKING only. Invariant to any strictly increasing transform of the score, so a
    calibration-only arm is IDENTICAL to H0 here by construction — that is the point."""
    m = np.asarray(w) > 0
    score, y, w = (np.asarray(a, dtype=np.float64)[m] for a in (score, y, w))
    return {"auc": _weighted_auc(score, y, w), "ap": _weighted_ap(score, y, w),
            "n_rows": int(score.size), "pos_weight": float((w * y).sum()),
            "neg_weight": float((w * (1 - y)).sum()),
            "BASIS": "RANKING · Horvitz-Thompson-weighted sampled rows",
            "promotes": False}


def calibration_diagnostics(score, y, w, *, n_bins: int = 12) -> dict:
    """CALIBRATION only. Log-loss and ECE in HT-weighted population units, plus the
    implied intercept gap — how many logits the score is systematically off by."""
    m = np.asarray(w) > 0
    score, y, w = (np.asarray(a, dtype=np.float64)[m] for a in (score, y, w))
    tot = float(w.sum())
    logloss = float((w @ (_softplus(score) - y * score)) / max(tot, 1e-12))
    p = _sigmoid(score)
    mean_pred = float((w @ p) / max(tot, 1e-12))
    mean_obs = float((w @ y) / max(tot, 1e-12))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        sel = b == k
        if not sel.any():
            continue
        wk = float(w[sel].sum())
        ece += wk / max(tot, 1e-12) * abs(float(w[sel] @ p[sel]) / wk
                                          - float(w[sel] @ y[sel]) / wk)
    def _lo(v):
        v = min(max(v, 1e-15), 1 - 1e-15)
        return math.log(v / (1 - v))
    return {"logloss_nats": logloss, "ece": float(ece), "mean_pred": mean_pred,
            "mean_obs": mean_obs, "implied_intercept_gap_logits": _lo(mean_pred) - _lo(mean_obs),
            "BASIS": "CALIBRATION · Horvitz-Thompson-weighted population units",
            "promotes": False}


def operating_point(score, y, w, threshold: float, *, n_est_total: float | None) -> dict:
    """Threshold-dependent counts, in HT population units. F1 is reported for continuity
    with the v5 table and is explicitly NOT a promotion signal."""
    m = np.asarray(w) > 0
    score, y, w = (np.asarray(a, dtype=np.float64)[m] for a in (score, y, w))
    pred = score >= threshold
    tp = float((w * pred * y).sum())
    fp = float((w * pred * (1 - y)).sum())
    fn = float((w * (~pred) * y).sum())
    prec, rec = tp / max(tp + fp, 1e-12), tp / max(tp + fn, 1e-12)
    out = {"threshold_logit": float(threshold), "ht_tp": tp, "ht_fp": fp, "ht_fn": fn,
           "precision": prec, "recall": rec,
           "f1_SAMPLED_BASIS_ONLY": 2 * prec * rec / max(prec + rec, 1e-12),
           "ht_predicted_positive_mass": tp + fp,
           "BASIS": "OPERATING POINT · HT-weighted. Sampled GT/background F1 cannot "
                    "promote an arm: acceptance is `logit == max_pool3d(logit)` AND "
                    "`sigmoid(logit) > det_threshold`, and the local-max structure of the "
                    "dense field is not expressible in any sampled row table.",
           "promotes": False}
    if n_est_total:
        out["node_ratio_vs_n_est"] = (tp + fp) / float(n_est_total)
    return out


def crop_block_bootstrap(score, y, w, crops, *, n_boot: int = 200,
                         seed: int = 20260806) -> dict:
    """Resample CROPS with replacement and recompute AUC.

    C7: the 199 crops come from only TWO embryos. This interval measures WITHIN-EMBRYO
    crop-to-crop variation. It does NOT estimate private-embryo generalisation and must
    never be quoted as if it did.
    """
    m = np.asarray(w) > 0
    score, y, w, crops = (np.asarray(a)[m] for a in (score, y, w, crops))
    uniq = np.array(sorted(set(crops.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    vals = []
    idx_by_crop = {c: np.flatnonzero(crops == c) for c in uniq}
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), size=len(uniq))]
        idx = np.concatenate([idx_by_crop[c] for c in pick])
        v = _weighted_auc(score[idx], y[idx].astype(np.float64), w[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"n_boot": 0, "CAVEAT": "no finite resample"}
    vals = np.array(vals)
    return {"auc_lo95": float(np.quantile(vals, 0.025)),
            "auc_hi95": float(np.quantile(vals, 0.975)),
            "n_boot": len(vals), "n_crops": int(len(uniq)),
            "CAVEAT": "WITHIN-EMBRYO crop-block variation only. The crops come from two "
                      "embryos; this is not an estimate of private-embryo "
                      "generalisation."}


def _best_intercept_shift(score, y, w, *, max_iter: int = 80, tol: float = 1e-12) -> float:
    """The single scalar `delta` minimising the weighted log-loss of `sigmoid(score+delta)`.

    Strictly convex in one variable, so Newton with backtracking lands on the optimum and
    two runs return the same float.
    """
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    d = 0.0
    for _ in range(max_iter):
        p = _sigmoid(score + d)
        g = float(w @ (p - y))
        h = float(w @ (p * (1.0 - p)))
        if h <= 1e-300:
            break
        step = g / h
        d -= step
        if abs(step) < tol:
            break
    return float(d)


def calibration_heterogeneity(score, y, w, crops, *, alpha: float = 0.05,
                              min_spread_logits: float = MIN_PER_CROP_SPREAD_LOGITS,
                              min_crops: int = 3) -> dict:
    """Does ONE scalar threshold shift serve every crop, or does each crop need its own?

    THIS IS THE WHOLE POINT OF SPLITTING THE CALIBRATION VERDICT. Acceptance is a level
    set, so every post-hoc calibrator — temperature, Platt, beta, histogram binning,
    isotonic — is worth exactly one scalar on this detector: `det_threshold`. A verdict
    that says only "calibration helps" therefore buys a threshold sweep that needs no
    instrument. The question worth an instrument is whether ONE constant can do it.

    METHOD. The test statistic is WHERE EACH CROP'S ANNOTATED CENTRES SIT ON THE SCORE
    AXIS: `mu_c`, the mean score over crop c's positive rows, with `se_c = sd_c /
    sqrt(n_c)`. Cochran's Q on `mu_c` about the precision-weighted pooled mean is
    chi-square on C-1 df under "one constant is enough"; the p-value uses the
    Wilson-Hilferty cube-root transform so nothing outside numpy is needed.

    WHY NOT THE HT-WEIGHTED PER-CROP INTERCEPT, which is the obvious first choice: it is
    unusable. Under Horvitz-Thompson weights the sandwich variance of a crop's intercept
    is `sum w^2 v / (sum w v)^2`, and with `uniform` rows carrying weight 4096 at a ~1e-5
    fitted rate that came out at se ~15 LOGITS per crop on the fixture. Q was ~0.3 against
    3 df even where the injected per-crop spread was 1.96 logits, so the verdict would
    have been unreachable by construction. The sandwich is not wrong — 400 sampled rows
    standing in for 1.6M grid voxels really do not locate an intercept — it is the wrong
    QUESTION. Annotated centres are an EXHAUSTIVE CENSUS, not a probability sample, so no
    HT weight belongs on them, and their location is what a threshold has to clear. The
    global and per-crop optimal shifts are still computed and reported, because they are
    the quantity a threshold sweep would move; they are simply not what the test is on.

    NOT A ROUTER. A positive result says no single deployed constant works. It does NOT
    license keying the threshold on crop or family identity at inference — that is
    forbidden — it says the deployment RULE needs a per-crop quantity the pipeline can
    compute for an unseen crop (density, estimated node count), which is a different
    mechanism and a separate piece of work.
    """
    m = np.asarray(w) > 0
    score, y, w, crops = (np.asarray(a)[m] for a in (score, y, w, crops))
    score = score.astype(np.float64)
    y = y.astype(np.float64)
    w = w.astype(np.float64)
    uniq = sorted(set(crops.tolist()))
    global_shift = _best_intercept_shift(score, y, w)

    per_crop, mus, prec = {}, [], []
    for c in uniq:
        sel = crops == c
        entry = {}
        if (w[sel] * y[sel]).sum() > 0 and (w[sel] * (1 - y[sel])).sum() > 0:
            entry["optimal_shift_logits"] = _best_intercept_shift(
                score[sel], y[sel], w[sel])
            entry["optimal_shift_residual_logits"] = (entry["optimal_shift_logits"]
                                                      - global_shift)
        pos = sel & (y == 1)
        n_pos = int(pos.sum())
        entry["n_positive_rows"] = n_pos
        if n_pos >= 2:
            mu = float(score[pos].mean())
            sd = float(score[pos].std(ddof=1))
            se = sd / math.sqrt(n_pos)
            entry.update({"mean_positive_score": mu, "se_logits": se})
            if se > 0:
                mus.append(mu)
                prec.append(1.0 / (se * se))
        else:
            entry["reason"] = "fewer than two annotated centres"
        per_crop[str(c)] = entry

    used = len(mus)
    df = max(used - 1, 0)
    q_stat, spread, pooled = 0.0, 0.0, None
    if used >= 2:
        mus_a, prec_a = np.array(mus), np.array(prec)
        pooled = float((prec_a @ mus_a) / prec_a.sum())
        q_stat = float(prec_a @ (mus_a - pooled) ** 2)
        spread = float(np.std(mus_a, ddof=1))
        for c, e in per_crop.items():
            if "mean_positive_score" in e:
                e["residual_logits"] = e["mean_positive_score"] - pooled
    p_value = _chi2_sf(q_stat, df) if df >= 1 else 1.0
    enough = used >= min_crops
    heterogeneous = bool(enough and df >= 1 and p_value <= alpha
                         and spread >= min_spread_logits)
    return {
        "global_shift_logits": global_shift, "per_crop": per_crop,
        "statistic": "per-crop mean score over ANNOTATED CENTRES (exhaustive census, no "
                     "HT weight), Cochran Q about the precision-weighted pooled mean",
        "pooled_mean_positive_score": pooled,
        "n_crops_used": used, "df": df, "cochran_q": float(q_stat),
        "p_value": float(p_value), "per_crop_spread_logits": spread,
        "min_spread_logits": float(min_spread_logits), "alpha": float(alpha),
        "enough_crops": bool(enough), "heterogeneous": heterogeneous,
        "BASIS": "CALIBRATION DEGREES OF FREEDOM · Cochran Q on per-crop optimal "
                 "intercept shifts, sandwich variance under HT weights",
        "NOT_A_ROUTER": "a positive result says no single deployed constant works. It "
                        "does NOT license crop or family identity as a deployment "
                        "router; that remains forbidden.",
        "promotes": False,
    }


def _chi2_sf(q: float, df: int) -> float:
    """Upper tail of chi-square via Wilson-Hilferty. numpy only, accurate for df >= 1."""
    if df < 1:
        return 1.0
    if q <= 0:
        return 1.0
    k = float(df)
    z = ((q / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def assert_calibration_preserves_ranking(h0_score, cal_score, y, w, *,
                                         atol: float = 1e-9) -> dict:
    """A calibration-only arm is `a*eta0 + c`, a MONOTONE map of the deployed score. It
    therefore invents no ranking information, and its AUC must be H0's (a > 0) or exactly
    `1 - H0`'s (a < 0). Anything else is a genuinely different ordering, which means the
    arm is not a calibration of the deployed head and the CALIBRATION-vs-LINEAR_HEAD
    distinction has collapsed.

    An inverted arm is REPORTED, not raised. Under a true null `a` is a coin flip, and an
    earlier version raised on `a < 0` — which made `LINEAR_PROBE_NULL`, the verdict that
    exists precisely for that case, unreachable: the null fixture regime crashed instead
    of returning its own null. `inverted` is carried to the verdict, which refuses to
    call an inverted arm a CALIBRATION.
    """
    m = np.asarray(w) > 0
    a0 = _weighted_auc(np.asarray(h0_score)[m], np.asarray(y)[m], np.asarray(w)[m])
    a1 = _weighted_auc(np.asarray(cal_score)[m], np.asarray(y)[m], np.asarray(w)[m])
    if not np.isfinite(a0) or not np.isfinite(a1):
        raise ControlFailure("calibration ranking check has no finite AUC")
    same, flipped = abs(a0 - a1), abs((1.0 - a0) - a1)
    if min(same, flipped) > atol:
        raise ControlFailure(
            f"calibration-only arm changed the ranking (AUC {a0:.12f} -> {a1:.12f}; "
            f"neither preserved nor exactly inverted, |d|={same:.3e} / {flipped:.3e}). A "
            f"monotone map of the deployed score can only preserve or reverse the order, "
            f"so this arm is not a calibration of the deployed head and CALIBRATION vs "
            f"LINEAR_HEAD is no longer separable.")
    return {"h0_auc": a0, "cal_auc": a1, "max_abs_diff": min(same, flipped),
            "inverted": bool(flipped < same),
            "claim": "calibration is monotone in the deployed score and cannot rerank; "
                     "an inverted slope is reported and blocks the CALIBRATION verdict"}


# ============================================================================ verdict
def _assert_permitted_verdict(verdict: str) -> str:
    """C2, encoded. The instrument may not emit a representation/encoder claim."""
    up = str(verdict).upper()
    for tok in _FORBIDDEN_VERDICT_TOKENS:
        if tok in up:
            raise VerdictError(
                f"verdict {verdict!r} contains the forbidden token {tok!r}. A null "
                f"linear probe on a sampled row table cannot fund an encoder programme. "
                f"Permitted verdicts are exactly {PERMITTED_VERDICTS}.")
    if verdict in RETIRED_VERDICTS:
        raise VerdictError(RETIRED_VERDICTS[verdict])
    if verdict not in PERMITTED_VERDICTS:
        raise VerdictError(
            f"verdict {verdict!r} is not one of {PERMITTED_VERDICTS}")
    return verdict


def decide_verdict(*, h0_rank: dict, lin_rank: dict, shuf_lin_ranks: list,
                   h0_cal: dict, cal_only_cal: dict, shuf_cal_cals: list,
                   min_ranking_delta: float = MIN_RANKING_DELTA_AUC,
                   min_calibration_delta: float = MIN_CALIBRATION_DELTA_NATS,
                   shuffle_null_band: float = 0.05,
                   null_se_multiple: float = 2.0,
                   alpha: float = 0.05,
                   cal_slope_positive: bool = True,
                   cal_heterogeneity: dict | None = None) -> dict:
    """The whole decision, pre-registered.

    Gate first: the label-shuffled refits must return a target-family AUC whose MEAN is
    indistinguishable from 0.5. If it is not, the pipeline is measuring leakage and NO
    verdict is emitted — the run raises rather than reporting a number nobody should act
    on.

    The null is a DISTRIBUTION, not a draw. MEASURED (FIXTURE, split-1 direction, 10
    seeds): one shuffled refit returns target AUC with sd 0.148, mean 0.562. Gating a
    single draw against 0.5 +/- 0.05 — which is what the first version of this function
    did — fires spuriously in the majority of runs, because 0.05 is the STANDARD ERROR OF
    A 20-SEED MEAN and it was being applied to one observation. The band is therefore
    applied to the mean, and a real gain must additionally clear the whole upper tail:
    `lin_auc > null_hi95` and `lin_auc > null_mean + 3*sd`. Otherwise the "gain" is
    inside the range that permuted labels reproduce.

    Then, with both gains read NET of their own shuffled null:
      ranking gain material   -> LINEAR_HEAD  (a reranking refit subsumes calibration)
      else calibration gain material -> CALIBRATION
      else                    -> LINEAR_PROBE_NULL
    """
    null = summarise_null(list(shuf_lin_ranks), list(shuf_cal_cals))
    shuf_auc = null["auc_mean"]
    # LEAKAGE IS ONE-SIDED. Leakage means the pipeline scores ABOVE chance on permuted
    # labels; a shuffled null that lands BELOW 0.5 is the refit failing to find anything,
    # which is the outcome the control exists to confirm.
    #
    # It has to be one-sided, because permuting labels on a FIXED corpus does not
    # integrate over the corpus. The shuffled null carries an irreducible corpus-level
    # offset that no number of label permutations averages away: MEASURED (FIXTURE, 19
    # seeds) mean 0.4183, se 0.0211 — 3.9 se below 0.5 and stable across seeds. A
    # two-sided gate reads that as leakage and refuses to emit any verdict at all.
    #
    # The tolerance is the wider of the pre-registered band and a multiple of the null's
    # own standard error, so the gate never asserts a precision the estimator lacks.
    null_tolerance = max(float(shuffle_null_band), null_se_multiple * null["auc_se"])
    if not np.isfinite(shuf_auc):
        raise ControlFailure("label-shuffled control returned no finite mean AUC")
    if shuf_auc - 0.5 > null_tolerance:
        raise ControlFailure(
            f"label-shuffled control returned MEAN target AUC {shuf_auc:.4f} over "
            f"{null['n_replicates']} seeds (sd {null['auc_sd']:.4f}, se "
            f"{null['auc_se']:.4f}), ABOVE 0.5 + {null_tolerance:.4f}. With labels "
            f"permuted there is no signal to find, so scoring above chance means the "
            f"pipeline is measuring leakage (group structure, threshold reuse, or a "
            f"basis mix-up). No verdict is emitted from an instrument in this state.")

    if null["min_attainable_p"] > alpha:
        raise ControlFailure(
            f"{null['n_replicates']} shuffled replicates can attain a smallest "
            f"permutation p of {null['min_attainable_p']:.4f}, which never reaches "
            f"alpha={alpha}. Every arm would fail the null-tail test for want of "
            f"replicates and the run would report LINEAR_PROBE_NULL for a reason that "
            f"has nothing to do with the data. Raise n_shuffles to at least "
            f"{int(math.ceil(1.0 / alpha)) - 1} or state a weaker alpha explicitly.")

    # The gated quantity is the gain OVER THE DEPLOYED HEAD, full stop. The earlier
    # `net = (lin - h0) - (null - h0)` collapses algebraically to `lin - null`, which is
    # large whenever the features carry ANY signal — it clears +0.01 even when the refit
    # ranks strictly WORSE than H0, and it returned LINEAR_HEAD on the fixture regime
    # built to return CALIBRATION (rank gain -0.000146, "net" +0.5875). The shuffled null
    # belongs in the NOISE test, not in the effect size.
    lin_auc = float(lin_rank["auc"])
    rank_gain = lin_auc - float(h0_rank["auc"])
    rank_null = shuf_auc - float(h0_rank["auc"])
    rank_p = permutation_p(lin_auc, null["auc_values"], greater_is_extreme=True)
    clears_null = rank_p <= alpha
    cal_ll = float(cal_only_cal["logloss_nats"])
    calib_gain = float(h0_cal["logloss_nats"]) - cal_ll
    calib_null = float(h0_cal["logloss_nats"]) - null["logloss_mean"]
    calib_p = permutation_p(cal_ll, null["logloss_values"], greater_is_extreme=False)
    calib_clears_null = calib_p <= alpha
    # PRECONDITION for CALIBRATION: the deployed head must rank above the shuffled null
    # in the first place. Log-loss falls whenever the intercept is refitted — the sampled
    # prior is nowhere near the population base rate — so a calibration "gain" is
    # available even when the score is pure noise. On the null fixture regime that
    # returned CALIBRATION from a corpus with no feature/label association at all.
    # Perfectly calibrating a coin flip recovers no nuclei; there must be an ordering to
    # move the operating point along.
    h0_auc = float(h0_rank["auc"])
    h0_p = permutation_p(h0_auc, null["auc_values"], greater_is_extreme=True)
    h0_ranks_above_null = h0_p <= alpha
    # A calibration with a <= 0 inverts the deployed score. It is monotone, so it invents
    # no ranking information, but it is not a repair of anything and may never be sold as
    # CALIBRATION.
    cal_deployable = bool(cal_slope_positive)

    # CALIBRATION SPLITS IN TWO, and the split is the whole value of the token.
    # Acceptance is a level set, so every monotone calibrator is worth exactly one scalar
    # here. GLOBAL means that scalar exists and the answer is "run a threshold sweep" —
    # something we can already do without this instrument. PER_CROP means no single
    # deployed constant can serve every crop, which is a finding about the deployment
    # rule rather than about the head.
    het = dict(cal_heterogeneity or {})
    per_crop_needed = bool(het.get("heterogeneous", False))
    if rank_gain >= min_ranking_delta and clears_null:
        verdict = "LINEAR_HEAD"
    elif (calib_gain >= min_calibration_delta and calib_clears_null and cal_deployable
            and h0_ranks_above_null):
        verdict = "CALIBRATION_PER_CROP" if per_crop_needed else "CALIBRATION_GLOBAL"
    elif per_crop_needed and h0_ranks_above_null and cal_deployable:
        # Reachable on its own: one constant may be no better than the deployed one
        # corpus-wide and STILL be unable to serve every crop. That is a finding about
        # the deployment rule that a pooled log-loss gain cannot express.
        verdict = "CALIBRATION_PER_CROP"
    else:
        verdict = "LINEAR_PROBE_NULL"
    _assert_permitted_verdict(verdict)
    return {
        "verdict": verdict,
        "ranking_gain_auc": rank_gain, "ranking_null_auc": rank_null,
        "min_ranking_delta": min_ranking_delta,
        "GATED_ON": "ranking_gain_auc / calibration_gain_nats — the gain over the "
                    "DEPLOYED head. The shuffled null enters only as the permutation "
                    "p-value; it is not subtracted from the effect size.",
        "ranking_clears_null_tail": bool(clears_null),
        "ranking_permutation_p": rank_p,
        "calibration_gain_nats": calib_gain, "calibration_null_nats": calib_null,
        "calibration_slope_positive": cal_deployable,
        "calibration_degrees_of_freedom": het or None,
        "calibration_per_crop_needed": per_crop_needed,
        "LEVEL_SET_NOTE": "acceptance is `logit == max_pool3d(logit)` AND "
                          "`sigmoid(logit) > tau`. For any strictly increasing phi, "
                          "A(phi(s), phi(tau)) == A(s, tau) EXACTLY, and the local-max "
                          "test runs on the RAW logits. Temperature, Platt, beta, "
                          "histogram binning and isotonic are therefore all worth one "
                          "scalar here: det_threshold. That is why the calibration "
                          "verdict is split rather than reported bare.",
        "h0_ranks_above_null": bool(h0_ranks_above_null),
        "h0_permutation_p": h0_p,
        "calibration_clears_null_tail": bool(calib_clears_null),
        "calibration_permutation_p": calib_p,
        "min_calibration_delta": min_calibration_delta,
        "shuffled_control_auc": shuf_auc, "shuffle_null_band": shuffle_null_band,
        "shuffle_null_tolerance_applied": float(null_tolerance),
        "shuffle_null_gate": "ONE-SIDED. Only an ABOVE-chance shuffled null is leakage; "
                             "a below-chance null is the refit finding nothing, which is "
                             "what the control is for. Permuting labels on a fixed corpus "
                             "leaves a corpus-level offset no permutation count removes.",
        "shuffle_null_below_chance": bool(shuf_auc < 0.5),
        "shuffled_null": null, "null_se_multiple": null_se_multiple, "alpha": alpha,
        "permitted_verdicts": list(PERMITTED_VERDICTS),
        "POLICY_NOTE": "the two deltas are POLICY thresholds, not calibrated transfer "
                       "laws. They gate what this instrument is allowed to CLAIM; they "
                       "do not promote anything. Promotion requires dense inference -> "
                       "P3 harmonic -> complete wrapper -> exact patched pooled scorer.",
        "NOT_PERMITTED": "REPRESENTATION DEFICIT is not a verdict this instrument can "
                         "return. LINEAR_PROBE_NULL means this probe failed, nothing "
                         "more.",
    }


def assert_payload_clean(payload: dict) -> None:
    """Final guard before anything is written: no forbidden verdict anywhere, and no
    pooled cross-direction headline (C7)."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "verdict" and isinstance(v, str):
                    _assert_permitted_verdict(v)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(payload)
    for bad in ("pooled_verdict", "combined_verdict", "pooled_auc", "overall_auc"):
        if payload.get(bad) is not None:
            raise VerdictError(
                f"{bad!r} is populated. The two transfer directions come from two "
                f"embryos and must be reported separately (C7); pooling them "
                f"manufactures a generalisation claim the design cannot support.")


# ============================================================================ direction
def source_fit_pool(corpus: Corpus, *, basis_split: int, n_folds: int = 4,
                    seed: int = 20260806) -> dict:
    """The exact row masks `run_direction` will use, computed the same way.

    Exposed because auxiliary tables (temporal pseudo-positives, consistency pairs) must
    be built against the FIT POOL, not against the whole source family. `TemporalTerm`
    rejects any index outside `allowed_rows`, so a table built against `source` silently
    BLOCKS H3/H4 — the arms whose distinctness this instrument exists to demonstrate.
    Callers that build tables must consume `fit_pool` from here.
    """
    in_basis = corpus.basis_mask(basis_split)
    src_fam, tgt_fam = SPLIT_SOURCE_FAMILY[basis_split], SPLIT_HELDOUT_FAMILY[basis_split]
    source = in_basis & corpus.family_mask(src_fam)
    target = in_basis & corpus.family_mask(tgt_fam)
    src_crops = corpus.crops(source)
    held_crops = (sorted(grouped_folds(np.array(src_crops), n_folds=n_folds, seed=seed)[0])
                  if len(src_crops) >= n_folds else [])
    same_family_held = source & np.isin(corpus.crop, held_crops)
    return {"in_basis": in_basis, "source": source, "target": target,
            "same_family_held": same_family_held, "fit_pool": source & ~same_family_held,
            "source_family": src_fam, "target_family": tgt_fam,
            "source_crops": src_crops, "same_family_held_crops": held_crops}


def summarise_null(ranks: list, calibs: list) -> dict:
    """Collapse the replicated shuffled arms into an empirical null distribution.

    A single shuffled refit is a draw, not a null: FIXTURE sd 0.148 AUC over 10 seeds.
    What is null is the MEAN; what a real gain must clear is the observed UPPER TAIL,
    tested as a permutation p-value rather than as a Gaussian multiple of sd. AUC is
    bounded in [0, 1] and at sd ~0.19 the bound `mean + 3 sd` lands at 0.99, which would
    make the ranking verdict unreachable by construction.
    """
    a = np.array([float(r["auc"]) for r in ranks], dtype=np.float64)
    a = a[np.isfinite(a)]
    ll = np.array([float(c["logloss_nats"]) for c in calibs], dtype=np.float64)
    ll = ll[np.isfinite(ll)]
    if a.size < 2 or ll.size < 2:
        raise ControlFailure(
            f"the shuffled null needs >= 2 finite replicates, got {a.size} AUC / "
            f"{ll.size} logloss. One draw cannot measure a band with sd ~0.15.")
    return {"n_replicates": int(a.size),
            "auc_mean": float(a.mean()), "auc_sd": float(a.std(ddof=1)),
            "auc_se": float(a.std(ddof=1) / math.sqrt(a.size)),
            "auc_hi95": float(np.quantile(a, 0.95)), "auc_max": float(a.max()),
            "auc_values": [float(v) for v in a],
            "logloss_mean": float(ll.mean()), "logloss_sd": float(ll.std(ddof=1)),
            "logloss_lo05": float(np.quantile(ll, 0.05)), "logloss_min": float(ll.min()),
            "logloss_values": [float(v) for v in ll],
            "min_attainable_p": 1.0 / (float(a.size) + 1.0),
            "BASIS": "empirical null over label-shuffled refits, one refit per seed"}


def permutation_p(observed: float, null_values, *, greater_is_extreme: bool) -> float:
    """Standard add-one permutation p-value: (1 + #{null at least as extreme}) / (R + 1).

    Add-one because the observed statistic is itself one of the achievable arrangements;
    without it a p of exactly 0 would be reported from a finite set of draws.
    """
    v = np.asarray(list(null_values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ControlFailure("permutation p-value needs at least one finite null draw")
    at_least = (v >= observed) if greater_is_extreme else (v <= observed)
    return float((1.0 + float(at_least.sum())) / (v.size + 1.0))


def run_direction(corpus: Corpus, *, basis_split: int, h0: LinearHead,
                  mask_radius_um: float | None = None, temporal: dict | None = None,
                  l2: float = 1.0, arms=DEFAULT_ARMS, budget_multiple: float = 1.5,
                  n_folds: int = 4, seed: int = 20260806,
                  parity_atol: float = 1e-4, n_boot: int = 0,
                  n_shuffles: int = DEFAULT_N_SHUFFLES,
                  shuffle_scope: str = "global") -> dict:
    """One cell-pair of the 2x2: fit/select on the SOURCE family inside checkpoint basis
    `basis_split`, freeze, then open the TARGET family exactly once."""
    if int(h0.basis_split) != int(basis_split):
        raise BasisError(f"H0 head is split {h0.basis_split}, basis is {basis_split}")
    in_basis = corpus.basis_mask(basis_split)
    if not in_basis.any():
        raise BasisError(f"no rows encoded by checkpoint split {basis_split}")
    src_fam, tgt_fam = SPLIT_SOURCE_FAMILY[basis_split], SPLIT_HELDOUT_FAMILY[basis_split]
    source = in_basis & corpus.family_mask(src_fam)
    target = in_basis & corpus.family_mask(tgt_fam)
    if not source.any() or not target.any():
        raise BasisError(
            f"checkpoint basis {basis_split} carries "
            f"{int(source.sum())} source ({src_fam}) and {int(target.sum())} target "
            f"({tgt_fam}) rows. The 2x2 design needs BOTH families encoded by the SAME "
            f"checkpoint. A routed-only export puts each family in its own basis and "
            f"cannot support this: the v6 export must emit CROSS-ENCODED features "
            f"(every crop encoded by both split checkpoints).")

    # H0 parity, or abort. Checked on every row in the basis, both families.
    assert_same_basis(h0, corpus, in_basis, label=f"H0/basis{basis_split}")
    parity = assert_logit_parity(h0, corpus.X[in_basis], corpus.logit[in_basis],
                                 atol=parity_atol, label=f"H0/basis{basis_split}")

    sampling_rate = corpus.sampling_rate()
    n_est = corpus.n_est()
    ht_w = ht_weights(corpus.kind, corpus.crop, sampling_rate)
    caps = Capabilities(
        checkpoint_head=h0,
        dist_to_nearest_gt_um=corpus.dist_to_nearest_gt_um,
        mask_radius_um=mask_radius_um, pi_crop=corpus.pi_crop(),
        temporal_pseudo_pos_idx=(temporal or {}).get("pseudo_pos_idx"),
        temporal_pseudo_pos_w=(temporal or {}).get("pseudo_pos_w"),
        temporal_pair_idx=(temporal or {}).get("pair_idx"),
        temporal_pair_w=(temporal or {}).get("pair_w"),
        sampling_rate=sampling_rate)

    # Same-family control: hold out a quarter of the SOURCE crops entirely, so every arm
    # also gets a within-family read. That is the ceiling the cross-family number is read
    # against — a cross-family drop is only interpretable next to it.
    pools = source_fit_pool(corpus, basis_split=basis_split, n_folds=n_folds, seed=seed)
    src_crops = pools["source_crops"]
    sf_held_crops = pools["same_family_held_crops"]
    same_family_held = pools["same_family_held"]
    fit_pool = pools["fit_pool"]

    built, blocked, fps = build_arms(corpus, caps, source_mask=fit_pool, arms=arms, l2=l2,
                                     n_shuffles=n_shuffles, shuffle_seed=seed,
                                     shuffle_scope=shuffle_scope)
    for nm, a in built.items():
        if a.param_space == "cal2":
            a.notes["_h0"] = h0

    results, heads, scores = {}, {}, {}
    for name in built:
        arm = built[name]
        name_b = arm_base(name)
        head, res, score = fit_arm(arm, basis_split=basis_split)
        heads[name], scores[name] = head, score

        if name_b == "H0":
            thr = DET_THRESHOLD_LOGIT
            sel = {"quantile": None, "objective": None,
                   "BASIS": "DEPLOYED det_threshold 0.96875; NOT selected"}
        elif name_b in SHUFFLED_ARMS:
            # A null arm has no operating point worth selecting, and selecting one would
            # multiply the fit cost by n_shuffles * n_folds for a number nobody reads.
            thr = DET_THRESHOLD_LOGIT
            sel = {"quantile": None, "objective": None,
                   "BASIS": "NULL ARM — no threshold selected; the deployed threshold is "
                            "reused only so the operating-point row is populated"}
        else:
            def _fit_on(train_crops, _arm=arm, _split=basis_split):
                sub = Arm(name=_arm.name, design=_arm.design, y=_arm.y,
                          s=_arm.s * np.isin(corpus.crop, train_crops),
                          offset=_arm.offset, param_space=_arm.param_space,
                          terms=_arm.terms, spec=_arm.spec, head=_arm.head,
                          fitted=_arm.fitted, notes=_arm.notes)
                return fit_arm(sub, basis_split=_split)[0]

            def _score_with(h, _arm=arm):
                if _arm.param_space == "cal2":
                    a = float(h.w @ h0.w) / float(h0.w @ h0.w)
                    return a * _arm.design[:, 0] + (h.b - a * h0.b)
                return h.logit(corpus.X, dtype="float64")

            sel = nested_select_quantile(
                _fit_on, _score_with, corpus=corpus, source_mask=fit_pool, ht_w=ht_w,
                n_est=n_est, budget_multiple=budget_multiple, n_folds=n_folds, seed=seed)
            act = fit_pool & (ht_w > 0)
            thr = float(np.quantile(score[act], sel["quantile"]))

        row = {
            "arm": name, "role": REGISTRY[name_b].role,
            "param_space": arm.param_space, "fitted": arm.fitted,
            # A probe that reports a promotable head but does not preserve the 33 numbers
            # cannot produce a candidate.  Keep the exact float64 deployment parameters in
            # the result artifact; downstream kernels may cast once to the checkpoint dtype.
            "deployed_head": {
                "w": [float(v) for v in head.w],
                "b": float(head.b),
                "basis_split": head.basis_split,
                "param_sha256": head.param_sha256(),
                "provenance": head.provenance,
            },
            "threshold_logit": thr, "selection": sel,
            "fingerprint": fps.get(name),
            "fit": None if res is None else {
                "n_iter": res.n_iter, "converged": res.converged,
                "newton_decrement": res.newton_decrement,
                "firth_engaged": res.firth_engaged,
                "separation": None if res.separation is None else res.separation.__dict__,
                "terms": res.term_report},
            "notes": {k: v for k, v in arm.notes.items() if not k.startswith("_")},
            "target_family": {
                "family": tgt_fam,
                "ranking": ranking_diagnostics(score[target], corpus.y[target], ht_w[target]),
                "calibration": calibration_diagnostics(score[target], corpus.y[target],
                                                       ht_w[target]),
                "operating_point": operating_point(
                    score[target], corpus.y[target], ht_w[target], thr,
                    n_est_total=sum(n_est[c] for c in corpus.crops(target)))},
            "same_family_heldout_crops": {
                "family": src_fam, "crops": sf_held_crops,
                "ranking": ranking_diagnostics(score[same_family_held],
                                               corpus.y[same_family_held],
                                               ht_w[same_family_held]),
                "calibration": calibration_diagnostics(score[same_family_held],
                                                       corpus.y[same_family_held],
                                                       ht_w[same_family_held]),
                "BASIS": "IN-FAMILY control — the within-family ceiling. NOT a transfer "
                         "result."},
        }
        if n_boot:
            row["target_family"]["crop_block_bootstrap"] = crop_block_bootstrap(
                score[target], corpus.y[target], ht_w[target], corpus.crop[target],
                n_boot=n_boot, seed=seed)
        results[name] = row

    assert_fitted_heads_distinct(heads)

    cal_check = None
    if "CAL_ONLY" in scores and "H0" in scores:
        cal_check = assert_calibration_preserves_ranking(
            scores["H0"][target], scores["CAL_ONLY"][target],
            corpus.y[target], ht_w[target])

    def _null_of(base_name, field):
        return [r["target_family"][field] for nm, r in results.items()
                if arm_base(nm) == base_name]

    shuf_lin_ranks = _null_of("SHUF_LIN", "ranking")
    shuf_cal_cals = _null_of("SHUF_CAL", "calibration")

    # How many degrees of freedom does the calibration actually need? Read on the
    # CALIBRATION-ONLY arm's target-family scores, because that is the arm whose whole
    # content is an operating point.
    het = None
    if "CAL_ONLY" in scores:
        het = calibration_heterogeneity(scores["CAL_ONLY"][target], corpus.y[target],
                                        ht_w[target], corpus.crop[target])

    verdict = None
    if ("H0" in results and "CAL_ONLY" in results and "LIN_HEAD" in results
            and shuf_lin_ranks and shuf_cal_cals):
        verdict = decide_verdict(
            h0_rank=results["H0"]["target_family"]["ranking"],
            lin_rank=results["LIN_HEAD"]["target_family"]["ranking"],
            shuf_lin_ranks=shuf_lin_ranks,
            h0_cal=results["H0"]["target_family"]["calibration"],
            cal_only_cal=results["CAL_ONLY"]["target_family"]["calibration"],
            shuf_cal_cals=shuf_cal_cals,
            cal_slope_positive=not (cal_check or {}).get("inverted", False),
            cal_heterogeneity=het)

    uniq_rates = sorted(set(round(v, 12) for v in sampling_rate.values()))
    return {
        "direction": f"split{basis_split}_{src_fam}_to_{tgt_fam}",
        "checkpoint_basis_split": basis_split,
        "source_family": src_fam, "target_family": tgt_fam,
        "n_source_rows": int(source.sum()), "n_target_rows": int(target.sum()),
        "n_source_crops": len(src_crops), "n_target_crops": len(corpus.crops(target)),
        "h0_parity": parity,
        "horvitz_thompson": {
            "uniform_sampling_rates": uniq_rates,
            "intercept_correction_logits": [intercept_correction(r) for r in uniq_rates],
            "reference_1_over_4096_logits": HT_LOGIT_CORRECTION_1_4096,
            "NOTE": "uncorrected, the intercept sits on the sampled prior. At 1/4096 the "
                    "overstatement is 8.3178 logits, against a deployed threshold of "
                    "+3.4340."},
        "calibration_monotonicity_check": cal_check,
        "calibration_degrees_of_freedom": het,
        "arms": results, "blocked": blocked,
        "verdict_detail": verdict,
        "verdict": None if verdict is None else verdict["verdict"],
    }


def run_factorial(corpus: Corpus, heads_by_basis: dict, **kw) -> dict:
    """Both directions of the 2x2, reported SEPARATELY and never pooled.

    Split 1 (44b6 -> held-out 6bba) is staged first: 6bba is ~85% of the objective, so
    the direction that matters most is the one that runs at full scale first.
    """
    directions = []
    for split in (1, 0):
        if split not in heads_by_basis:
            continue
        if not corpus.basis_mask(split).any():
            continue
        directions.append(run_direction(corpus, basis_split=split,
                                        h0=heads_by_basis[split], **kw))
    payload = {
        "instrument": "D1-F frozen-feature linear probe (v6 rewrite)",
        "permitted_verdicts": list(PERMITTED_VERDICTS),
        "design": "2x2 checkpoint x family. Fit and select ONLY on the source family "
                  "inside one checkpoint basis; freeze; open the target family once. The "
                  "two 32-D bases are independent (rel-L2 ~ sqrt(2)) and a head is never "
                  "carried across them.",
        "directions": directions,
        "verdict_by_direction": {d["direction"]: d["verdict"] for d in directions},
        "pooled_verdict": None,
        "pooled_auc": None,
        "C7_NOTE": "the two directions are NOT pooled. The 199 crops come from two "
                   "embryos; crop-block bootstrap measures within-embryo variation and "
                   "does not estimate private-embryo generalisation. There is no "
                   "headline number here by design.",
        "PROMOTION": {"promotable_from_this_instrument": False,
                      "requires": "dense inference -> P3 harmonic -> complete wrapper -> "
                                  "exact patched pooled scorer, both embryo families, "
                                  "min-fold delta >= +0.005"},
    }
    assert_payload_clean(payload)
    return payload


# ============================================================================ fixtures
@dataclass
class FixtureSpec:
    """Synthetic v6 corpus. The v6 export does not exist yet and the 3-crop v5 smoke is
    forbidden as training data, so everything this module says about itself comes from
    here. BASIS TAG: FIXTURE.

    Grid and uniform counts are chosen so the sampling rate is EXACTLY 1/4096, which is
    the rate the 8.3178-logit intercept correction is quoted at.
    """

    n_crops_per_family: int = 4
    grid_zyx: tuple = (16, 64, 64)
    n_frames: int = 100
    n_uniform_per_frame: int = 16              # 16 / (16*64*64) = 1/4096 exactly
    n_subthr_per_crop: int = 600
    gt_min: int = 60
    gt_max: int = 400
    n_views: int = 8
    sigma_bg: float = 1.0                      # HYPOTHESIS: unet.head has no activation
    view_jitter: float = 0.15
    signal_along_w: float = 2.6
    signal_orthogonal: float = 1.9
    subthr_latent_pos_frac: float = 0.35
    # Per-crop displacement ALONG w_ckpt, in feature units. It moves a crop's positives
    # and negatives together, so the crop's separation is untouched and only its optimal
    # operating point moves. That is exactly the regime CALIBRATION_PER_CROP names: one
    # corpus-wide scalar cannot serve every crop.
    per_crop_shift: float = 0.0
    seed: int = 20260806


def _unit(v):
    return v / np.linalg.norm(v)


def _make_basis_rows(head: LinearHead, families, spec: FixtureSpec, seed: int):
    """Rows for BOTH families under ONE checkpoint basis — the cross-encoded export the
    2x2 requires. Feature scales/directions are drawn per basis, so the two bases are
    mutually unrelated, exactly as the measured rel-L2 ~ sqrt(2) says they are."""
    rng = np.random.default_rng(seed)
    w_hat = _unit(head.w)
    g = rng.normal(size=FEAT_DIM)
    ortho = _unit(g - (g @ w_hat) * w_hat)

    crops, kinds, crop_col, t_col, dist, latent, blocks = [], [], [], [], [], [], []
    n_uniform = spec.n_uniform_per_frame * spec.n_frames
    gt_counts = {}
    for fam in families:
        for i in range(spec.n_crops_per_family):
            c = f"{fam}_{seed:04x}{i:04x}"
            crops.append(c)
            n_gt = int(rng.integers(spec.gt_min, spec.gt_max + 1))
            gt_counts[c] = n_gt
            n = n_gt + n_uniform + spec.n_subthr_per_crop
            kinds.extend(["gt_centre"] * n_gt + ["uniform"] * n_uniform
                         + ["subthr_localmax"] * spec.n_subthr_per_crop)
            crop_col.extend([c] * n)
            t_col.extend(rng.integers(0, spec.n_frames, size=n).tolist())
            pos = np.zeros(n, dtype=bool)
            pos[:n_gt] = True
            sub = np.zeros(n, dtype=bool)
            sub[n_gt + n_uniform:] = True
            lat = pos.copy()
            sub_idx = np.flatnonzero(sub)
            lat[rng.choice(sub_idx, size=int(spec.subthr_latent_pos_frac * sub_idx.size),
                           replace=False)] = True
            latent.append(lat)
            f = rng.normal(0.0, spec.sigma_bg, size=(n, FEAT_DIM))
            amp = np.where(pos, 1.0, np.where(lat, 0.55, 0.0))[:, None]
            f = f + amp * (spec.signal_along_w * w_hat + spec.signal_orthogonal * ortho)
            if spec.per_crop_shift:
                f = f + float(rng.normal(0.0, spec.per_crop_shift)) * w_hat
            blocks.append(f)
            d = np.where(pos, 0.0, rng.gamma(2.0, 6.0, size=n))
            d[lat & ~pos] = rng.gamma(2.0, 2.0, size=int((lat & ~pos).sum()))
            dist.append(d)
    return (crops, np.array(kinds), np.array(crop_col),
            np.array(t_col, dtype=np.int64), np.concatenate(dist),
            np.concatenate(latent), np.concatenate(blocks), gt_counts, rng)


def make_factorial_fixture(heads: dict, spec: FixtureSpec | None = None):
    """A cross-encoded 2x2 corpus: both families under both checkpoint bases.

    The parity identity is made true by the same route the real kernel must make it true:
    per-view features are generated, the TTA-mean feature is their mean, and the exported
    logit is the mean of the per-view logits. If head(mean_v f_v) != mean_v head(f_v) in
    this fixture, the head is not linear.
    """
    spec = spec or FixtureSpec()
    fams = ("44b6", "6bba")
    parts, latents, manifest_crops = [], [], {}
    for split in sorted(heads):
        head = heads[split]
        (crops, kinds, crop_col, t_col, dist, latent, base, gt_counts,
         rng) = _make_basis_rows(head, fams, spec, spec.seed + 101 * split)
        N = base.shape[0]
        wf, bf = head.w.astype(np.float32), np.float32(head.b)
        tta_sum = np.zeros((N, FEAT_DIM), dtype=np.float32)
        logit_sum = np.zeros(N, dtype=np.float32)
        for _v in range(spec.n_views):
            fv = (base + rng.normal(0.0, spec.view_jitter,
                                    size=base.shape)).astype(np.float32)
            tta_sum += fv
            logit_sum += fv @ wf + bf
        del base
        tta_mean = (tta_sum / np.float32(spec.n_views)).astype(np.float32)
        logit = (logit_sum / np.float32(spec.n_views)).astype(np.float64)
        X_max = tta_mean.copy()
        pos_idx = np.flatnonzero(kinds == "gt_centre")
        X_max[pos_idx] = tta_mean[pos_idx] + rng.normal(
            0.0, 0.25, size=(pos_idx.size, FEAT_DIM)).astype(np.float32)
        for c in crops:
            manifest_crops[c] = {
                "status": "complete", "grid_zyx": list(spec.grid_zyx),
                "n_frames": spec.n_frames,
                "n_uniform_per_frame": spec.n_uniform_per_frame,
                "estimated_number_of_nodes": int(gt_counts[c] * 12),
                "checkpoint_sha256": f"fixture_split{split}", "split": split,
                "encoder_split": split}
        parts.append({"X": tta_mean, "X_max": X_max, "kind": kinds, "crop": crop_col,
                      "t": t_col, "logit": logit, "dist": dist,
                      "enc": np.full(N, split, dtype=np.int64)})
        latents.append(latent)
    corpus = Corpus(
        X=np.concatenate([p["X"] for p in parts]),
        X_max=np.concatenate([p["X_max"] for p in parts]),
        kind=np.concatenate([p["kind"] for p in parts]),
        crop=np.concatenate([p["crop"] for p in parts]),
        family=np.array([family_of(c) for c in
                         np.concatenate([p["crop"] for p in parts])]),
        t=np.concatenate([p["t"] for p in parts]),
        logit=np.concatenate([p["logit"] for p in parts]),
        encoder_split=np.concatenate([p["enc"] for p in parts]),
        dist_to_nearest_gt_um=np.concatenate([p["dist"] for p in parts]),
        manifest={"tta_view_set": [f"fixture_planar{spec.n_views}_{_i}"
                                  for _i in range(spec.n_views)],
                  "n_encode_calls": spec.n_views,
                  "n_distinct_views": spec.n_views, "crops": manifest_crops},
        feature_source="tta_mean")
    return corpus, np.concatenate(latents)


def fixture_heads(seed: int = 4242) -> dict:
    """Two independent 32-D bases, one per checkpoint split."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in (0, 1):
        w = rng.normal(0.0, 0.2, size=FEAT_DIM)
        out[split] = LinearHead(w=w, b=-3.0 + 0.5 * split, name=f"H0_split{split}",
                                basis_split=split,
                                provenance={"kind": "FIXTURE", "fitted": False})
    return out


FIXTURE_REGIMES = {
    # A large orthogonal component: only a refitted head can see it => LINEAR_HEAD.
    "linear_head": dict(signal_along_w=2.0, signal_orthogonal=3.0),
    # All signal along w_ckpt, but placed so the deployed threshold is badly located and
    # the sampled-prior intercept is wrong. Ranking unchanged, and ONE scalar fixes every
    # crop => CALIBRATION_GLOBAL.
    "calibration_global": dict(signal_along_w=4.5, signal_orthogonal=0.0),
    # As above, plus a per-crop displacement along w_ckpt: each crop's optimal operating
    # point sits somewhere different, so no single deployed constant serves them all
    # => CALIBRATION_PER_CROP.
    "calibration_per_crop": dict(signal_along_w=4.5, signal_orthogonal=0.0,
                                 per_crop_shift=1.5),
    # No association between features and labels at all => LINEAR_PROBE_NULL.
    "null": dict(signal_along_w=0.0, signal_orthogonal=0.0, subthr_latent_pos_frac=0.0),
}


def make_temporal_tables(corpus: Corpus, latent, *, source_mask, n_pseudo: int = 300,
                         n_pairs: int = 400, seed: int = 7) -> dict:
    """Temporal pseudo-positives + consistency pairs, SOURCE-FAMILY ROWS ONLY."""
    rng = np.random.default_rng(seed)
    elig = np.flatnonzero(source_mask & (corpus.kind == "subthr_localmax") & latent)
    n_pseudo = min(n_pseudo, elig.size)
    pp = rng.choice(elig, size=n_pseudo, replace=False)
    tr = np.flatnonzero(source_mask)
    a = rng.choice(tr, size=n_pairs, replace=False)
    b = rng.choice(tr, size=n_pairs, replace=False)
    keep = a != b
    return {"pseudo_pos_idx": pp, "pseudo_pos_w": np.full(pp.size, 0.5),
            "pair_idx": np.stack([a[keep], b[keep]], axis=1),
            "pair_w": np.full(int(keep.sum()), 0.1)}


def make_separation_fixture(n: int = 240, gap: float = 0.8, seed: int = 11):
    """A corpus on which the unpenalised MLE genuinely does not exist: two clusters,
    completely separated along feature 0 with a strictly positive margin."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.3, size=(n, FEAT_DIM))
    y = np.zeros(n)
    y[: n // 2] = 1.0
    X[: n // 2, 0] += gap
    X[n // 2:, 0] -= gap
    return X, y, np.ones(n)


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit-root", default=None,
                    help="v6 export root containing basis_0/ and basis_1/")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--fixture", choices=sorted(FIXTURE_REGIMES),
                    help="run the whole instrument on a synthetic corpus (the v6 export "
                         "does not exist yet and the 3-crop v5 smoke is forbidden as "
                         "training data)")
    ap.add_argument("--expect-encode-calls", "--expect-views", dest="expect_encode_calls",
                    type=int, default=8,
                    help="expected TTA encode-call count (legacy --expect-views alias retained)")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--mask-radius-um", type=float, default=None)
    ap.add_argument("--budget-multiple", type=float, default=1.5)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES,
                    help="label-shuffled replicates per negative control. One draw is "
                         "not a null: its target AUC has sd ~0.15.")
    ap.add_argument("--temporal-table", default=None,
                    help="npz with pseudo_pos_idx/pseudo_pos_w/pair_idx/pair_w")
    ap.add_argument("--out", default=str(ROOT / "research/06-knowledge-system/inventory/d1f_probe.json"))
    a = ap.parse_args()

    temporal = None
    if a.temporal_table:
        with np.load(a.temporal_table) as z:
            temporal = {k: z[k] for k in ("pseudo_pos_idx", "pseudo_pos_w",
                                          "pair_idx", "pair_w")}

    if a.fixture:
        heads = fixture_heads()
        spec = FixtureSpec(**FIXTURE_REGIMES[a.fixture])
        corpus, _latent = make_factorial_fixture(heads, spec)
        payload = run_factorial(corpus, heads, mask_radius_um=a.mask_radius_um,
                                temporal=temporal, l2=a.l2,
                                budget_multiple=a.budget_multiple, n_folds=a.n_folds,
                                n_boot=a.n_boot, n_shuffles=a.n_shuffles)
        payload["BASIS"] = f"FIXTURE · synthetic regime {a.fixture!r}. Says nothing about "
        payload["BASIS"] += "the real corpus; it demonstrates that the instrument works."
    else:
        if not a.audit_root or not a.weights_dir:
            raise SystemExit("--audit-root and --weights-dir are required without "
                             "--fixture")
        corpus = load_factorial(a.audit_root, expect_encode_calls=a.expect_encode_calls)
        heads = {s: LinearHead.for_basis(s, a.weights_dir) for s in corpus.bases()}
        payload = run_factorial(corpus, heads, mask_radius_um=a.mask_radius_um,
                                temporal=temporal, l2=a.l2,
                                budget_multiple=a.budget_multiple, n_folds=a.n_folds,
                                n_boot=a.n_boot, n_shuffles=a.n_shuffles)
        payload["BASIS"] = "EXACT-OOF row-sampled diagnostics on the v6 export."

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    for d in payload["directions"]:
        print(f"{d['direction']}: verdict={d['verdict']}  "
              f"arms={sorted(d['arms'])}  blocked={sorted(d['blocked'])}")
    print("NO POOLED HEADLINE: the two directions are reported separately (C7).")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
