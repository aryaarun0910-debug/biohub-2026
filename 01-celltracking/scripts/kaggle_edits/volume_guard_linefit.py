# =====================================================================================
# IN-KERNEL VOLUME GUARD for linefit smoothing  (slot-3 candidate P0-C-R)
#
# P0-C emitted node 15274 of 44b6_0b24845f at t=43 with z=64 where z_max=63, and failed
# gate A3 of scripts/audit_submission_structure.py. The cause is in the smoother: at a
# track endpoint the +/-window neighbourhood is one-sided, so the line fit EXTRAPOLATES
# and can push a node outside the acquisition volume.
#
# scripts/repair_linefit_volume.py fixes an ALREADY-WRITTEN csv by reconstructing the
# original detector coordinate. That repair cannot be submitted: this competition
# accepts submissions from NOTEBOOKS only, so a locally repaired CSV is unusable however
# well audited. The fix therefore has to run inside the kernel, which is what this does.
#
# The repair is deliberately NOT a clamp. Clamping invents a boundary coordinate the
# detector never proposed and piles mass onto the boundary plane, and the population is
# already boundary-heavy. Instead the smoothed value is discarded PER AXIS -- a fit may
# extrapolate out of bounds in z while y/x stay good -- and the original unsmoothed
# detector coordinate is restored. Smoothing then becomes provably incapable of evicting
# a node from the volume.
#
# If the ORIGINAL coordinate is itself out of bounds that is an upstream detector defect;
# it is counted separately and left alone, so the structural audit still fails and says
# so rather than being masked.
# =====================================================================================
_VG_VOLUME_ZYX = tuple(
    int(v) for v in os.environ.get("BIOHUB_OUTPUT_VOLUME_ZYX", "64,256,256").split(",")
)
_VG_AXES = ("z", "y", "x")


def _vg_in_volume(value: float, axis: int) -> bool:
    return 0.0 <= float(value) <= float(_VG_VOLUME_ZYX[axis] - 1)


def _vg_restore_axes(blended, original, stats):
    out = blended.copy()
    for _axis in range(3):
        if _vg_in_volume(out[_axis], _axis):
            continue
        _name = _VG_AXES[_axis]
        stats[f"linefit_volume_fallback_{_name}"] = (
            stats.get(f"linefit_volume_fallback_{_name}", 0) + 1
        )
        if not _vg_in_volume(original[_axis], _axis):
            _key = f"linefit_volume_original_invalid_{_name}"
            stats[_key] = stats.get(_key, 0) + 1
        out[_axis] = original[_axis]
    return out
