"""Mathematical contract and execution invariance for the installable library."""

import errno
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sol_metric import SOL, Features, SOLConfig
from sol_metric._math import directions, gp_directions
from sol_metric._runtime import available_memory, ram_backed


def reference(tx, ox, ty, oy, cfg):
    """Independent float64 reference using empirical atom/bin overlaps."""
    u = directions(cfg.n_directions, tx.shape[1], cfg.seed)
    gp = np.random.default_rng(cfg.gp_seed).standard_normal((cfg.n_projections, cfg.n_quantiles))
    if cfg.lengthscale is not None:
        grid = (np.arange(cfg.n_quantiles) + 0.5) / cfg.n_quantiles
        covariance = np.exp(-((grid[:, None] - grid) ** 2) / (2 * cfg.lengthscale**2))
        gp = gp @ np.linalg.cholesky(covariance + 1e-6 * np.eye(cfg.n_quantiles)).T
    features = []
    for tokens, offsets in ((tx, ox), (ty, oy)):
        documents = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            ordered = np.sort(tokens[a:b].astype(np.float64) @ u.T, axis=0)
            t, m = b - a, cfg.n_quantiles
            bins = np.zeros((cfg.n_directions, m))
            for j in range(m):
                for r in range(t):
                    mass = max(0, min((j + 1) / m, (r + 1) / t) - max(j / m, r / t))
                    bins[:, j] += m * mass * ordered[r]
            documents.append(bins)
        q = np.stack(documents)
        features.append(np.einsum("nlm,km->nlk", q, gp) / cfg.n_quantiles)
    x, y = features
    nx, ny = len(x), len(y)
    x, y = np.sort(x, axis=0), np.sort(y, axis=0)
    squared = 0.0
    for i in range(nx):
        for j in range(ny):
            mass = max(0, min((i + 1) / nx, (j + 1) / ny) - max(i / nx, j / ny))
            squared += mass * np.mean((x[i] - y[j]) ** 2)
    return float(np.sqrt((tx.shape[1] if cfg.scale is None else cfg.scale) * squared))


class RawGPTests(unittest.TestCase):
    def test_white_noise_is_unmodified_gaussian_innovations(self):
        expected = np.random.default_rng(17).standard_normal((64, 7))
        np.testing.assert_array_equal(gp_directions(64, 7, None, 17), expected)

    def test_rbf_draws_keep_random_norms_and_unit_prior_covariance(self):
        samples = gp_directions(20000, 5, 0.2, 19)
        grid = (np.arange(5) + 0.5) / 5
        expected = np.exp(-((grid[:, None] - grid) ** 2) / (2 * 0.2**2))
        np.testing.assert_allclose(samples.mean(0), 0, atol=0.025)
        np.testing.assert_allclose(samples.T @ samples / len(samples), expected, atol=0.035)
        self.assertGreater(np.sqrt(np.mean(samples**2, axis=1)).std(), 0.2)
        self.assertGreater(np.std(samples.mean(1)), 0.3)
        np.testing.assert_allclose(np.linalg.norm(directions(32, 7, 19), axis=1), 1, atol=1e-14)


class MetricTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        rng = np.random.default_rng(13)
        self.ox = np.array([0, 1, 6, 8, 25])
        self.oy = np.array([0, 3, 5, 16])
        self.x = rng.normal(size=(25, 7)).astype(np.float32)
        self.y = (rng.normal(size=(16, 7)) + 0.7).astype(np.float32)
        self.cfg = SOLConfig(
            n_directions=7, n_projections=9, n_quantiles=8, seed=3, gp_seed=7, scale=1
        )

    def metric(self, **kwargs):
        config = kwargs.pop("config", self.cfg)
        return SOL(**asdict(config), device=self.device, **kwargs)

    def test_config_compatibility_and_keyword_overrides(self):
        expected = self.metric().from_tokens(self.x, self.ox, self.y, self.oy)
        for metric in (SOL(self.cfg, device=self.device), SOL(config=self.cfg, device=self.device)):
            self.assertEqual(metric.from_tokens(self.x, self.ox, self.y, self.oy), expected)

        # Explicit None resets meaningful options; omitted fields retain the config.
        override = SOL(
            self.cfg, seed=0, lengthscale=None, scale=None, device=self.device, dtype="float64"
        )
        expected_cfg = replace(self.cfg, seed=0, lengthscale=None, scale=None)
        self.assertAlmostEqual(
            override.from_tokens(self.x, self.ox, self.y, self.oy),
            reference(self.x, self.ox, self.y, self.oy, expected_cfg),
            places=11,
        )
        self.assertEqual(self.cfg.seed, 3)
        self.assertEqual(self.cfg.lengthscale, 0.1)
        self.assertEqual(self.cfg.scale, 1)

    def test_matches_independent_reference(self):
        actual = self.metric(dtype="float64").from_tokens(self.x, self.ox, self.y, self.oy)
        self.assertAlmostEqual(
            actual, reference(self.x, self.ox, self.y, self.oy, self.cfg), places=11
        )

    def test_dimension_scale(self):
        expected = self.metric().from_tokens(self.x, self.ox, self.y, self.oy) * np.sqrt(7)
        metric = self.metric(config=replace(self.cfg, scale=None))
        self.assertAlmostEqual(
            metric.from_tokens(self.x, self.ox, self.y, self.oy), expected, places=6
        )
        fx, fy = metric.featurize(self.x, self.ox), metric.featurize(self.y, self.oy)
        self.assertAlmostEqual(metric.from_features(fx, fy), expected, places=6)
        with self.assertRaisesRegex(ValueError, "explicit scale"):
            metric.from_features(fx.values, fy.values)

    def test_missing_dimension_fails_before_scoring(self):
        metric = self.metric(config=replace(self.cfg, scale=None))
        fx, fy = metric.featurize(self.x, self.ox), metric.featurize(self.y, self.oy)
        metadata = {key: value for key, value in fx.metadata.items() if key != "dimension"}
        unscaled = (
            (fx.values, fy.values),
            (Features(fx.values, metadata), Features(fy.values, metadata)),
        )
        gp = np.zeros((self.cfg.n_projections, self.cfg.n_quantiles))
        for inputs in unscaled:
            with (
                patch("sol_metric.metric.gp_directions", return_value=gp) as sample,
                patch.object(metric.backend, "slice_sum", return_value=0.0) as score,
            ):
                with self.assertRaisesRegex(ValueError, "explicit scale"):
                    metric.from_features(*inputs)
                sample.assert_not_called()
                score.assert_not_called()
            self.assertIsNone(metric.last_run)

    def test_batch_and_block_invariance(self):
        expected = reference(self.x, self.ox, self.y, self.oy, self.cfg)
        # Token budgets of 4 and 16 bin the 17-token document on its own.
        for batch, slices, tokens in ((1, 1, 4), (2, 5, 16), (3, 12, 40), (16, 1024, 1000)):
            with self.subTest(batch=batch, slices=slices, tokens=tokens):
                m = self.metric(
                    batch_size=batch, slice_batch_size=slices, max_tokens_per_batch=tokens
                )
                self.assertAlmostEqual(
                    m.from_tokens(self.x, self.ox, self.y, self.oy), expected, delta=2e-6 * expected
                )

    def test_identity_symmetry_and_scale(self):
        m = self.metric()
        self.assertEqual(m.from_tokens(self.x, self.ox, self.x, self.ox), 0)
        score = m.from_tokens(self.x, self.ox, self.y, self.oy)
        self.assertAlmostEqual(score, m.from_tokens(self.y, self.oy, self.x, self.ox), places=6)
        scaled = self.metric(config=replace(self.cfg, scale=9)).from_tokens(
            self.x, self.ox, self.y, self.oy
        )
        self.assertAlmostEqual(scaled, 3 * score, places=6)

    def test_singleton_is_distance_not_squared_distance(self):
        cfg = SOLConfig(n_directions=1, n_projections=1, n_quantiles=1)
        score = self.metric(config=cfg).from_tokens(
            np.array([[0.0]]), [0, 1], np.array([[3.0]]), [0, 1]
        )
        # One raw GP draw retains its Gaussian magnitude, including for M=1.
        g = np.random.default_rng(cfg.gp_seed).standard_normal() * np.sqrt(1 + 1e-6)
        self.assertAlmostEqual(score, 3 * abs(g))

    def test_results_record_raw_gp_and_cached_features_match_token_scores(self):
        metric = self.metric()
        expected = metric.from_tokens(self.x, self.ox, self.y, self.oy)
        self.assertEqual(metric.last_run["gp_normalization"], "none")
        qx, qy = metric.featurize(self.x, self.ox), metric.featurize(self.y, self.oy)
        self.assertNotIn("gp_normalization", qx.metadata)
        self.assertAlmostEqual(metric.from_features(qx, qy), expected, places=6)
        self.assertEqual(metric.last_run["gp_normalization"], "none")

    def test_bin_averages_recover_projected_means_for_any_grid(self):
        u = directions(self.cfg.n_directions, self.x.shape[1], self.cfg.seed)
        expected = np.stack(
            [
                self.x[a:b].astype(np.float64).mean(0) @ u.T
                for a, b in zip(self.ox[:-1], self.ox[1:])
            ]
        )
        for count in (1, 3, 8, 64):
            cfg = replace(self.cfg, n_quantiles=count)
            q = self.metric(config=cfg, dtype="float64").featurize(self.x, self.ox)
            self.assertEqual(q.metadata["quantile_convention"], "empirical_bin_average_v1")
            np.testing.assert_allclose(q.values.mean(-1), expected, rtol=1e-12, atol=1e-12)

    def test_replicating_atoms_preserves_quantile_features(self):
        metric = self.metric(dtype="float64")
        original = metric.featurize(self.x, self.ox).values
        repeated = metric.featurize(np.repeat(self.x, 3, axis=0), self.ox * 3).values
        np.testing.assert_allclose(original, repeated, rtol=1e-12, atol=1e-12)

    def test_rejects_two_matching_legacy_or_unlabelled_caches(self):
        metric = self.metric()
        x, y = metric.featurize(self.x, self.ox), metric.featurize(self.y, self.oy)
        for convention in ("linear_midpoints", None):
            for q in (x, y):
                q.metadata["quantile_convention"] = convention
            with self.assertRaisesRegex(ValueError, "quantile convention"):
                metric.from_features(x, y)

    def test_document_order_invariance(self):
        docs = [self.x[a:b] for a, b in zip(self.ox[:-1], self.ox[1:])][::-1]
        x = np.concatenate(docs)
        ox = np.r_[0, np.cumsum([len(d) for d in docs])]
        m = self.metric()
        self.assertAlmostEqual(
            m.from_tokens(self.x, self.ox, self.y, self.oy),
            m.from_tokens(x, ox, self.y, self.oy),
            places=6,
        )

    def test_features_cache_and_reuse(self):
        m = self.metric()
        expected = m.from_tokens(self.x, self.ox, self.y, self.oy)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as caches:
            qx = caches.enter_context(m.featurize(self.x, self.ox, path=Path(tmp) / "reference"))
            qy = caches.enter_context(m.featurize(self.y, self.oy))
            # Disk caches are read-only mappings stored in direction blocks.
            stored = np.load(Path(tmp) / "reference" / "values.npy", mmap_mode="r")
            self.assertEqual(stored.shape[1:], (4, 128, 8))  # 128 x 8 float32 = one page
            self.assertFalse(qx._data.flags.writeable)
            del stored
            self.assertEqual(qx.shape, (4, 7, 8))
            self.assertAlmostEqual(m.from_features(qx, qy), expected, places=6)
            qy = caches.enter_context(qy.save(Path(tmp) / "generated"))
            loaded = caches.enter_context(Features.load(Path(tmp) / "reference"))
            self.assertAlmostEqual(m.from_features(loaded, qy), expected, places=6)
            with self.assertRaises(FileExistsError):
                qy.save(Path(tmp) / "generated")
            json.dumps(m.last_run, allow_nan=False)

    def test_direction_blocks_read_any_range_and_round_trip(self):
        # Width 4 (4 x 256 float32 bins = one page) with 10 directions pads the last block.
        cfg = replace(self.cfg, n_directions=10, n_quantiles=256)
        metric = self.metric(config=cfg)
        reference = metric.featurize(self.x, self.ox)
        logical = reference.values
        self.assertEqual(reference._data.shape, (3, 4, 4, 256))
        self.assertEqual(logical.shape, (4, 10, 256))
        u = directions(cfg.n_directions, self.x.shape[1], cfg.seed)
        direct = metric.backend.quantiles(self.x, np.asarray(self.ox), u, cfg.n_quantiles)
        np.testing.assert_array_equal(logical, metric.backend.host(direct))
        for start, stop in ((0, 10), (0, 4), (3, 9), (4, 8), (9, 10), (5, 5)):
            with self.subTest(start=start, stop=stop):
                np.testing.assert_array_equal(reference.block(start, stop), logical[:, start:stop])
        with self.assertRaisesRegex(ValueError, "outside"):
            reference.block(2, 11)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Features(logical, reference.metadata)
            with raw.save(Path(tmp) / "a") as saved, Features.load(Path(tmp) / "a") as loaded:
                np.testing.assert_array_equal(saved.values, logical)
                np.testing.assert_array_equal(loaded.block(3, 9), logical[:, 3:9])
                self.assertEqual(
                    metric.from_features(loaded, reference),
                    metric.from_features(reference, reference),
                )
            manifest = Path(tmp) / "a" / "metadata.json"
            payload = json.loads(manifest.read_text())
            payload["schema"] = 1
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "schema"):
                Features.load(Path(tmp) / "a")

    def test_cache_closes_on_exception_before_directory_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with self.metric().featurize(self.x, self.ox, path=path) as cache:
                    self.assertFalse(cache.closed)
                    raise RuntimeError("interrupted")
            self.assertTrue(cache.closed)
            cache.close()  # Closing twice is safe.
            with self.assertRaisesRegex(ValueError, "closed"):
                _ = cache.values
            with self.assertRaisesRegex(ValueError, "closed"):
                with cache:
                    pass
            shutil.rmtree(path)  # Windows rejects this if a mapping is still open.
            self.assertFalse(path.exists())

    def test_invalid_cache_releases_file_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache"
            with self.metric().featurize(self.x, self.ox, path=path):
                pass
            manifest = path / "metadata.json"
            payload = json.loads(manifest.read_text())
            payload["shape"][0] += 1
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                Features.load(path)
            shutil.rmtree(path)
            self.assertFalse(path.exists())

    def test_closing_features_preserves_caller_owned_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = np.lib.format.open_memmap(
                Path(tmp) / "external.npy", mode="w+", dtype=np.float32, shape=(1, 1, 1)
            )
            try:
                values[:] = 3
                with Features(values, {}):
                    pass
                self.assertEqual(values[0, 0, 0], 3)
            finally:
                values._mmap.close()

    def test_rejects_mismatched_feature_provenance(self):
        m = self.metric()
        qx = m.featurize(self.x, self.ox, provenance={"encoder": "one"})
        qy = m.featurize(self.y, self.oy, provenance={"encoder": "two"})
        with self.assertRaisesRegex(ValueError, "provenance"):
            m.from_features(qx, qy)
        qy.metadata = qx.metadata.copy()
        with self.assertRaisesRegex(ValueError, "seed"):
            self.metric(config=replace(self.cfg, seed=99)).from_features(qx, qy)

    def test_batch_sizes_are_recorded_and_slices_fill_each_step(self):
        m = self.metric(batch_size=3, max_tokens_per_batch=40, slice_batch_size=6)
        m.from_tokens(self.x, self.ox, self.y, self.oy)
        execution = m.last_run["execution"]
        self.assertEqual(
            {k: execution[k] for k in ("batch_size", "max_tokens_per_batch", "slice_batch_size")},
            {"batch_size": 3, "max_tokens_per_batch": 40, "slice_batch_size": 6},
        )
        # Six slices per step: six of the seven directions are read, with one GP draw each.
        self.assertEqual(execution["score_directions"], 6)
        self.assertEqual(execution["score_projections"], 1)

    def test_validation(self):
        m = self.metric()
        for offsets in ([1, 25], [0, 0, 25], [0, 24], [0.0, 25.0]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                m.from_tokens(self.x, offsets, self.y, self.oy)
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            m.from_tokens(bad, self.ox, self.y, self.oy)
        for kwargs in ({"batch_size": 0}, {"slice_batch_size": -1}, {"dtype": "bad"}):
            with self.assertRaises(ValueError):
                self.metric(**kwargs)
        for kwargs in (
            {"n_directions": 0},
            {"n_projections": None},
            {"n_quantiles": True},
            {"lengthscale": -1},
            {"seed": -1},
            {"gp_seed": 0.5},
            {"scale": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SOL(device=self.device, **kwargs)
        with self.assertRaisesRegex(TypeError, "config must be a SOLConfig"):
            SOL(config={}, device=self.device)
        with self.assertRaises(TypeError):
            SOL(transport="midpoint")


class WorkspaceTests(unittest.TestCase):
    def test_failed_featurization_does_not_leave_partial_cache(self):
        x = np.ones((2, 3), dtype=np.float32)
        metric = SOL(n_directions=2, n_projections=2, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features"
            with patch.object(metric.backend, "quantiles", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    metric.featurize(x, [0, 1, 2], path=path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_out_of_memory_errors_name_the_batch_size_to_lower(self):
        x = np.random.default_rng(0).normal(size=(6, 3)).astype(np.float32)
        metric = SOL(n_directions=2, n_projections=2, device="cpu")
        cases = (
            ("quantiles", MemoryError(), "lower batch_size or max_tokens_per_batch"),
            ("slice_sum", MemoryError(), "lower slice_batch_size"),
            ("slice_sum", type("OutOfMemoryError", (RuntimeError,), {})(), "GPU memory"),
        )
        for method, error, advice in cases:
            with self.subTest(method=method, error=type(error).__name__):
                with patch.object(metric.backend, method, side_effect=error):
                    with self.assertRaisesRegex(MemoryError, advice):
                        metric.from_tokens(x, [0, 2, 6], x, [0, 3, 6])
        with patch.object(metric.backend, "slice_sum", side_effect=RuntimeError("other")):
            with self.assertRaisesRegex(RuntimeError, "other"):
                metric.from_tokens(x, [0, 2, 6], x, [0, 3, 6])

    def test_disk_features_fit_when_in_memory_output_does_not(self):
        x = np.arange(16, dtype=np.float32)[:, None]
        offsets = np.arange(17)
        # 16 documents x 128 directions x 64 bins x 4 bytes = 0.5 MiB of features.
        metric = SOL(n_directions=128, n_projections=2, device="cpu", feature_memory_mb=0.25)
        with patch("sol_metric.metric.directions") as sample:
            with self.assertRaisesRegex(MemoryError, "feature_memory_mb=0.25"):
                metric.featurize(x, offsets)
            sample.assert_not_called()
        expected = SOL(metric.config, device="cpu").featurize(x, offsets)
        with tempfile.TemporaryDirectory() as tmp:
            with metric.featurize(x, offsets, path=Path(tmp) / "features") as actual:
                np.testing.assert_array_equal(actual.values, expected.values)

    def test_auto_feature_memory_is_half_the_available_memory(self):
        x = np.arange(16, dtype=np.float32)[:, None]
        offsets = np.arange(17)
        # 0.5 MiB of features, as above.
        metric = SOL(n_directions=128, n_projections=2, device="cpu")
        self.assertEqual(metric.feature_memory_mb, "auto")
        available = "sol_metric.metric.available_memory"
        with patch(available, return_value=2**20 - 1):
            with self.assertRaisesRegex(MemoryError, "half the available memory"):
                metric.featurize(x, offsets)
        with patch(available, return_value=2**20):
            self.assertEqual(len(metric.featurize(x, offsets)), 16)
        with patch(available, return_value=None):
            self.assertEqual(metric._feature_limit()[0], 4096)

    def test_feature_memory_must_be_auto_or_a_nonnegative_number(self):
        for value in ("max", "4096", -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SOL(device="cpu", feature_memory_mb=value)

    def test_disk_cache_checks_free_space_before_writing(self):
        x = np.ones((4, 3), dtype=np.float32)
        metric = SOL(n_directions=2, n_projections=2, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            full = SimpleNamespace(free=1024)
            with patch("sol_metric.features.shutil.disk_usage", return_value=full):
                with patch.object(metric.backend, "quantiles") as quantiles:
                    with self.assertRaises(OSError) as raised:
                        metric.featurize(x, [0, 2, 4], path=Path(directory) / "features")
                    quantiles.assert_not_called()
            self.assertEqual(raised.exception.errno, errno.ENOSPC)
            self.assertEqual(list(Path(directory).iterdir()), [])


class HostMemoryTests(unittest.TestCase):
    GIB = 1024**3

    def available(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            files = {"proc/meminfo": "MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n", **files}
            for name, text in files.items():
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            return available_memory(Path(tmp) / "proc", Path(tmp) / "cgroup")

    def test_available_memory_is_capped_by_cgroup_limits(self):
        gib = self.GIB
        cases = {
            "no cgroup file": ({}, 32 * gib),
            "unlimited v2": (
                {"proc/self/cgroup": "0::/job\n", "cgroup/job/memory.max": "max\n"},
                32 * gib,
            ),
            # SLURM sets the job limit on a parent of the step's cgroup.
            "v2 limit on a parent, inactive file cache reclaimable": (
                {
                    "proc/self/cgroup": "0::/job/step\n",
                    "cgroup/job/memory.max": f"{16 * gib}\n",
                    "cgroup/job/memory.current": f"{6 * gib}\n",
                    "cgroup/job/memory.stat": f"anon 1\ninactive_file {2 * gib}\n",
                    "cgroup/job/step/memory.max": "max\n",
                },
                12 * gib,
            ),
            "v1 limit": (
                {
                    "proc/self/cgroup": "5:cpu,cpuacct:/slurm/job\n4:memory:/slurm/job\n",
                    "cgroup/memory/slurm/job/memory.limit_in_bytes": f"{8 * gib}\n",
                    "cgroup/memory/slurm/job/memory.usage_in_bytes": f"{3 * gib}\n",
                    "cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
                    "cgroup/memory/memory.usage_in_bytes": f"{40 * gib}\n",
                },
                5 * gib,
            ),
        }
        for name, (files, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.available(files), expected)

    def test_available_memory_is_unknown_without_meminfo(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(available_memory(Path(tmp), Path(tmp)))

    @unittest.skipUnless(sys.platform.startswith("linux"), "mount table is Linux-specific")
    def test_ram_backed_uses_the_innermost_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            mounts = Path(tmp) / "mounts"
            mounts.write_text(
                "/dev/sda1 / ext4 rw 0 0\n"
                "tmpfs /sol-ram tmpfs rw 0 0\n"
                "/dev/sdb1 /sol-ram/disk ext4 rw 0 0\n"
                "tmpfs /sol\\040space ramfs rw 0 0\n"
            )
            for path, expected in (
                ("/sol-ram/x", True),
                ("/sol-ram/disk/x", False),
                ("/sol-ramdisk", False),
                ("/sol space/x", True),
                ("/home", False),
            ):
                with self.subTest(path=path):
                    self.assertEqual(ram_backed(path, mounts), expected)


class CpuTensorTests(unittest.TestCase):
    def test_bfloat16_inputs_convert_before_numpy_and_detach(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch dependency unavailable")
        x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.bfloat16)
        x = x.T.contiguous().T.requires_grad_()
        y = (x.detach() + 1).requires_grad_()
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                metric = SOL(
                    n_directions=3, n_projections=4, n_quantiles=4, device="cpu", dtype=dtype
                )
                expected = metric.from_tokens(
                    x.detach().float().numpy(), [0, 1, 3], y.detach().float().numpy(), [0, 3]
                )
                self.assertEqual(metric.from_tokens(x, [0, 1, 3], y, [0, 3]), expected)
                self.assertIsNone(x.grad)
                self.assertIsNone(y.grad)


def cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@unittest.skipUnless(cuda_available(), "CUDA unavailable")
class CudaTests(MetricTests):
    device = "cuda:0"

    def test_resident_tensor_and_noncontiguous_input(self):
        import torch

        m = self.metric()
        expected = m.from_tokens(self.x, self.ox, self.y, self.oy)
        x = torch.tensor(self.x.T, device=self.device).T.requires_grad_()
        y = torch.tensor(self.y, device=self.device)
        self.assertAlmostEqual(m.from_tokens(x, self.ox, y, self.oy), expected, places=6)
        self.assertIsNone(x.grad)

    def test_reduced_precision_accuracy(self):
        import torch

        expected = reference(self.x, self.ox, self.y, self.oy, self.cfg)
        modes = ["float16"]
        if torch.cuda.is_bf16_supported():
            modes.append("bfloat16")
        for dtype in modes:
            with self.subTest(dtype=dtype):
                score = self.metric(dtype=dtype).from_tokens(self.x, self.ox, self.y, self.oy)
                self.assertAlmostEqual(score, expected, delta=0.02 * expected)


if __name__ == "__main__":
    unittest.main()
