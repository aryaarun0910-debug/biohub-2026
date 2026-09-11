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
    def test_diverging_sisters_keep_the_fork(self):
        g = resolve(_fork(14.0), Config(fork_divergence_min_um=0.0))
        self.assertEqual(g.forks(), 1, "sisters separating 10->14um must stay a division")

    def test_converging_sisters_lose_the_fork(self):
        g = resolve(_fork(6.0), Config(fork_divergence_min_um=0.0))
        self.assertEqual(g.forks(), 0, "sisters closing 10->6um are not a division")

    def test_gate_disabled_keeps_both(self):
        g = resolve(_fork(6.0), Config(fork_divergence_min_um=-99.0))
        self.assertEqual(g.forks(), 1, "at -99 the gate must be inert")


if __name__ == "__main__":
    unittest.main()
