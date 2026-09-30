"""The divergence gate silently never fired until 2026-09-11 (it indexed a frame-t map with
frame-t+1 nodes). This asserts it fires in both directions -- the check that would have caught it."""
import sys, unittest
import numpy as np
sys.path.insert(0, "src")
from biohub.contracts import Graph, Config, SCALE
from biohub.resolve import resolve


def _fork(sep_t2_um):
    """parent at t0, two daughters at t1 (10um apart), their successors at t2 sep_t2_um apart."""
    um = lambda z, y, x: np.array([z, y, x]) / SCALE
    zyx = np.array([um(0, 0, 0), um(0, -5, 0), um(0, 5, 0),
                    um(0, -sep_t2_um / 2, 0), um(0, sep_t2_um / 2, 0)])
    e = np.array([[0, 1], [0, 2], [1, 3], [2, 4]])
    return Graph(t=np.array([0, 1, 1, 2, 2]), zyx=zyx, edges=e,
                 edge_prob=np.ones(len(e)), dataset="synthetic")


class ForkDivergence(unittest.TestCase):
    """These pin fork_accept_p=-1.0 on purpose: EXP-15's learned model now supersedes the
    divergence threshold by default, and this class tests the FALLBACK rule itself."""

    def test_diverging_sisters_keep_the_fork(self):
        g = resolve(_fork(14.0), Config(fork_divergence_min_um=0.0, fork_accept_p=-1.0))
        self.assertEqual(g.forks(), 1, "sisters separating 10->14um must stay a division")

    def test_converging_sisters_lose_the_fork(self):
        g = resolve(_fork(6.0), Config(fork_divergence_min_um=0.0, fork_accept_p=-1.0))
        self.assertEqual(g.forks(), 0, "sisters closing 10->6um are not a division")

    def test_gate_disabled_keeps_both(self):
        g = resolve(_fork(6.0), Config(fork_divergence_min_um=-99.0, fork_accept_p=-1.0))
        self.assertEqual(g.forks(), 1, "at -99 the gate must be inert")


if __name__ == "__main__":
    unittest.main()


class OutDegreeCap(unittest.TestCase):
    """A cell divides once. The scorer silently DROPS edges past two children, so a third
    child is invisible corruption -- it only surfaced as a UserWarning at motion_gate 14."""

    def test_parent_never_gets_a_third_child(self):
        um = lambda z, y, x: np.array([z, y, x]) / SCALE
        # one parent at t0; four plausible daughters at t1 arranged symmetrically
        zyx = np.array([um(0, 0, 0), um(0, -4, 0), um(0, 4, 0), um(0, 0, -4), um(0, 0, 4)])
        e = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        g = Graph(t=np.array([0, 1, 1, 1, 1]), zyx=zyx, edges=e,
                  edge_prob=np.ones(len(e)), dataset="synthetic")
        out = resolve(g, Config())
        deg = np.bincount(out.edges[:, 0], minlength=5) if len(out.edges) else np.zeros(5)
        self.assertLessEqual(int(deg.max()), 2, f"out-degree {deg.max()} exceeds 2: {out.edges}")


class LearnedAcceptance(unittest.TestCase):
    """EXP-15 replaced the hand-tuned divergence rule with a 7-feature logistic model. The
    features must be computed IDENTICALLY at training and inference or the shipped coefficients
    mean nothing, so training imports the same fork_features() the pipeline calls."""

    def test_features_are_finite_and_ordered(self):
        from biohub.resolve import fork_features
        P = np.array([[0., 0., 0.], [0., -5., 0.], [0., 5., 0.], [0., -7., 0.], [0., 7., 0.]])
        f = fork_features(P, 0, 1, 2, {1: 3, 2: 4})
        self.assertEqual(len(f), 7)
        self.assertTrue(all(np.isfinite(f)), f)
        self.assertAlmostEqual(f[0], -1.0, places=6)          # perfectly opposed
        self.assertAlmostEqual(f[1], 10.0, places=6)          # sisters 10um apart
        self.assertGreater(f[5], 0.0)                          # and separating

    def test_model_scores_a_real_division_above_a_degenerate_one(self):
        from biohub.resolve import fork_probability
        P = np.array([[0., 0., 0.], [0., -5., 0.], [0., 5., 0.], [0., -7., 0.], [0., 7., 0.]])
        good = fork_probability(P, 0, 1, 2, {1: 3, 2: 4})
        Q = np.array([[0., 0., 0.], [0., 9., 0.], [0., 10., 0.], [0., 9., 0.], [0., 10., 0.]])
        bad = fork_probability(Q, 0, 1, 2, {1: 3, 2: 4})       # same side, no divergence
        self.assertGreater(good, bad, f"opposed+diverging {good:.3f} !> same-side {bad:.3f}")
