"""Multi-scale Difference-of-Gaussians proposals, on a grid made physically isotropic.

[[F-0017]] found the binding constraint: the classical local-maximum detector
reached node recall 0.269 on real bytes, and every annotated edge it could not
reach was unreachable because an endpoint was never proposed. Association had
nothing to fix. A better proposal source is therefore the first thing worth
changing, and a blob detector matched to the physical size of a nucleus is the
obvious candidate.

Difference-of-Gaussians is scale-space blob detection: the response of a DoG
filter peaks for a blob whose radius is about sigma * sqrt(ndim), so a small bank
of sigmas covers a range of nucleus sizes. This is textbook (Lindeberg's
scale-space theory, and the LoG/DoG detectors built on it), not a competition
technique, and it is implemented here from that description rather than adapted
from anyone's pipeline.

Two properties come from Biohub-X's own official metadata rather than from any
tuning. The voxel scale is (1.625, 0.40625, 0.40625) micrometres, so z is exactly
four times y and x, and downsampling the plane by four makes the sampling grid
isotropic at 1.625 micrometres. On an isotropic grid a single scalar sigma is
meaningful in every direction, which is what lets one DoG bank describe a
spherical nucleus. Suppression then happens in micrometres, reusing the same
greedy physical-radius pass the classical detector and the heatmap reader already
share, so the three sources cannot disagree about what a duplicate is.

No constant here is inherited. The scale bank is derived from a stated nucleus
radius, the normalisation is a per-frame percentile because intensity drifts
across a movie, and the response threshold is a quantile of the response itself
rather than an absolute number, because an absolute one would mean something
different on every movie.

Consumer: ``biohubx evaluate proposals``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet, ProposalSource
from biohubx.contracts.lineage import DatasetIdentity
from biohubx.proposals.classical import DEFAULT_SUPPRESSION_RADIUS_UM, suppressed_maxima

DEFAULT_NUCLEUS_RADII_UM: tuple[float, ...] = (2.0, 3.0, 4.5)
"""Radii the bank targets, in micrometres.

Three scales spanning small to large nuclei. Stated as physical radii rather than
as sigmas so the choice is legible: a sigma is an implementation detail of the
filter, a radius is a claim about cells.
"""

DEFAULT_RESPONSE_QUANTILE = 0.999
"""Keep responses above this quantile of the frame's own DoG response.

A quantile rather than an absolute threshold, because the response scale depends
on the movie's intensity distribution and an absolute number would silently mean
something different on each. The value is a starting point to be selected on a
training embryo, never inherited as tuned.
"""

DEFAULT_NORMALISATION_PERCENTILES: tuple[float, float] = (1.0, 99.9)


def isotropic_plane_stride(scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE) -> int:
    """How much to stride the plane so the grid is isotropic.

    Derived from the official voxel scale rather than assumed: z divided by y is
    exactly 4 on this corpus, and a non-integral ratio would make this refuse
    rather than round, because a grid that is nearly isotropic is not isotropic
    and a single sigma would then mean different distances along different axes.
    """
    ratio = scale.z_um / scale.y_um
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > 1e-9:
        raise ValueError(
            f"plane stride {ratio} is not integral, so no single stride makes this grid isotropic"
        )
    if abs(scale.y_um - scale.x_um) > 1e-9:
        raise ValueError("y and x spacing differ, so one plane stride cannot make the grid isotropic")
    return nearest


def normalise_frame(
    frame: np.ndarray, percentiles: tuple[float, float] = DEFAULT_NORMALISATION_PERCENTILES
) -> np.ndarray:
    """Map a frame onto [0, 1] by its own percentiles.

    Per frame, not per movie, because illumination and bleaching drift across a
    time course and a movie-wide scaling would make late frames dimmer than early
    ones for reasons that have nothing to do with cells.
    """
    low, high = np.percentile(frame, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(frame, dtype=np.float32)
    scaled = (frame.astype(np.float32) - float(low)) / float(high - low)
    clipped: np.ndarray = np.clip(scaled, 0.0, 1.0)
    return clipped


def dog_response(
    frame: np.ndarray,
    *,
    radii_um: tuple[float, ...],
    voxel_um: float,
    ratio: float = 1.6,
) -> np.ndarray:
    """Maximum normalised DoG response across the scale bank.

    For each target radius the inner sigma is radius / sqrt(3), because a 3D blob
    of radius r produces its strongest Laplacian response at sigma = r / sqrt(3).
    The outer sigma is the conventional 1.6 times the inner, which approximates a
    Laplacian of Gaussian. Responses are scale-normalised by multiplying by sigma
    squared, without which the largest scale always wins and the bank collapses to
    one filter.
    """
    if frame.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) frame, got shape {frame.shape}")
    if not radii_um:
        raise ValueError("the scale bank is empty, so nothing would be detected")
    best = np.full(frame.shape, -np.inf, dtype=np.float32)
    for radius_um in radii_um:
        sigma = (radius_um / np.sqrt(3.0)) / voxel_um
        if sigma <= 0:
            raise ValueError(f"radius {radius_um} is not positive in voxels")
        inner = gaussian_filter(frame, sigma=sigma, mode="nearest")
        outer = gaussian_filter(frame, sigma=sigma * ratio, mode="nearest")
        best = np.maximum(best, (inner - outer) * float(sigma**2))
    return best


def refine_centroid(
    response: np.ndarray, centre: tuple[float, float, float], radius: int = 1
) -> tuple[float, float, float]:
    """Response-weighted centroid in a small cube around a peak, on the isotropic grid.

    A DoG maximum sits on a grid voxel; the cell it marks does not. On the strided
    plane one voxel is 1.625 micrometres in every direction, so a peak can be off
    by most of a micrometre in each axis before any other error is counted, and
    [[F-0028]] measured a residual whose nearest proposal lies just outside the
    matching radius. Weighting the neighbourhood by response moves the centre
    toward the mass of the blob. Negative responses are clipped to zero so a
    neighbouring trough cannot pull the centroid away from the blob.
    """
    z, y, x = (round(value) for value in centre)
    z0, z1 = max(0, z - radius), min(response.shape[0], z + radius + 1)
    y0, y1 = max(0, y - radius), min(response.shape[1], y + radius + 1)
    x0, x1 = max(0, x - radius), min(response.shape[2], x + radius + 1)
    block = np.clip(response[z0:z1, y0:y1, x0:x1], 0.0, None).astype(np.float64)
    total = float(block.sum())
    if total <= 0.0:
        return centre
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return (
        float((block * zz).sum() / total),
        float((block * yy).sum() / total),
        float((block * xx).sum() / total),
    )


def detect_instances(
    volume: np.ndarray,
    *,
    dataset: DatasetIdentity,
    radii_um: tuple[float, ...] = DEFAULT_NUCLEUS_RADII_UM,
    response_quantile: float = DEFAULT_RESPONSE_QUANTILE,
    suppression_radius_um: float = DEFAULT_SUPPRESSION_RADIUS_UM,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
    refine_centroids: bool = False,
    local_maxima_only: bool = False,
    per_scale_union: bool = False,
) -> InstanceSet:
    """Propose one instance per suppressed DoG maximum, frame by frame.

    The volume arrives at full resolution and is strided in the plane so the DoG
    runs on an isotropic grid; coordinates are scaled back to full-resolution
    voxels before an instance is built, so every downstream consumer sees the same
    coordinate convention the other proposal sources use.
    """
    if volume.ndim != 4:
        raise ValueError(f"expected a (T, Z, Y, X) volume, got shape {volume.shape}")
    if not 0.0 < response_quantile < 1.0:
        raise ValueError(f"response quantile must lie in (0, 1), got {response_quantile}")

    stride = isotropic_plane_stride(scale)
    isotropic_um = float(scale.z_um)
    instances: list[CandidateInstance] = []
    next_id = 0

    for frame_index in range(volume.shape[0]):
        plane = volume[frame_index][:, ::stride, ::stride]
        normalised = normalise_frame(plane)
        response = dog_response(normalised, radii_um=radii_um, voxel_um=isotropic_um)
        union_mask = None
        if per_scale_union:
            # H-11c. The max-over-scales response lets one scale's peak swallow
            # another's: a large scale fuses two neighbours into one maximum and
            # that maximum wins the pointwise max, so the small scale's two peaks
            # never become candidates. Taking strict peaks at each scale on its
            # own and unioning them keeps both, and the shared physical
            # suppression pass afterwards is what removes the true duplicates.
            union_mask = np.zeros(response.shape, dtype=bool)
            footprint = np.ones((3, 3, 3), dtype=bool)
            footprint[1, 1, 1] = False
            for radius_um in radii_um:
                single = dog_response(normalised, radii_um=(radius_um,), voxel_um=isotropic_um)
                neighbours = maximum_filter(single, footprint=footprint, mode="constant", cval=-np.inf)
                union_mask |= (single > neighbours) & (single > 0.0)
        finite = response[np.isfinite(response)]
        if finite.size == 0:
            continue
        cutoff = float(np.quantile(finite, response_quantile))
        span = float(finite.max() - finite.min())
        if span <= 0:
            continue
        # suppressed_maxima wants a map in [0, 1] and an absolute threshold, so
        # the response is rescaled once and the quantile carried across with it.
        floor = float(finite.min())
        scaled = np.clip((response - floor) / span, 0.0, 1.0)
        threshold = float(np.clip((cutoff - floor) / span, 1e-6, 1.0 - 1e-6))
        if union_mask is not None:
            scaled = np.where(union_mask, scaled, 0.0)
        elif local_maxima_only:
            # Greedy physical-radius suppression accepts any above-threshold voxel
            # that is not within the radius of one already accepted. On a wide
            # blob that admits its shoulders as separate proposals, and the count
            # it returns is set by how much of the frame clears the quantile, not
            # by how many blobs there are: three different scale banks returned
            # the same 1,398 candidates on one frame. Requiring a voxel to be the
            # maximum of its 26-neighbourhood first makes a candidate a peak.
            # Strictly greater than every one of the 26 neighbours, with the
            # centre excluded from the footprint. A plateau ties with itself under
            # a plain maximum filter, so a flat background would pass as a field
            # of peaks; and a DoG peak that marks a blob is positive, so a
            # non-positive maximum is a ripple in the background, not a cell.
            footprint = np.ones((3, 3, 3), dtype=bool)
            footprint[1, 1, 1] = False
            # Padded with -inf, not by replication. Under "nearest" padding a
            # voxel on a volume face is compared with a copy of itself, the strict
            # inequality fails, and every peak on the first or last slice is lost;
            # the first corpus probe found the missed cells clustered near the
            # faces for exactly that reason.
            neighbours = maximum_filter(response, footprint=footprint, mode="constant", cval=-np.inf)
            peaks = (response > neighbours) & (response > 0.0)
            scaled = np.where(peaks, scaled, 0.0)
        for raw_centre, peak in suppressed_maxima(
            scaled,
            threshold=threshold,
            radius_um=suppression_radius_um,
            # The strided plane is isotropic at the z spacing, which is the whole
            # reason for striding it. Suppression is told that spacing rather than
            # the official anisotropic one, because it suppresses on the grid it
            # is handed.
            spacing_um=(isotropic_um, isotropic_um, isotropic_um),
        ):
            centre = refine_centroid(response, raw_centre) if refine_centroids else raw_centre
            instances.append(
                CandidateInstance.from_voxel(
                    dataset=dataset,
                    instance_id=next_id,
                    frame=frame_index,
                    # Back to full-resolution voxels. The plane was strided, z was
                    # not, so only y and x are scaled back.
                    voxel=VoxelCoordinateZYX(
                        z=float(centre[0]),
                        y=float(centre[1]) * stride,
                        x=float(centre[2]) * stride,
                    ),
                    source=ProposalSource.DOG_MULTISCALE,
                    confidence=float(min(1.0, max(0.0, peak))),
                    scale=scale,
                )
            )
            next_id += 1
    if not instances:
        raise ValueError("the detector proposed nothing; an empty proposal set is a failure, not a result")
    return InstanceSet(dataset=dataset, instances=tuple(instances))
