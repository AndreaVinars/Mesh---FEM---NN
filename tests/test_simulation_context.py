import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "Pipeline" / "Mesh-FEM" / "simulation_context.py"
SPEC = importlib.util.spec_from_file_location("simulation_context", MODULE_PATH)
simulation_context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simulation_context)


def make_config(working_dir: Path) -> dict:
    calculix_stub = working_dir / "ccx_test"
    calculix_stub.touch()

    return {
        "plate": {"width": 12.0, "height": 12.0, "thickness": 1.0},
        "loading": {"eps0": 0.005, "gamma0": 0.005},
        "material": {
            "name": "Steel",
            "youngs_modulus": 210000.0,
            "poisson_ratio": 0.3,
        },
        "lhs_bounds": {
            "lower": [3.1, 3.1, 0.3, 0.3, 0.0, 3.1, 3.1, 0.3, 0.3, 0.0],
            "upper": [8.9, 8.9, 3.0, 2.0, 359.9, 8.9, 8.9, 3.0, 2.0, 359.9],
        },
        "simulation": {
            "calculation_mode": "direct_contraction",
            "num_simulations": 20,
            "n_jobs": 2,
            "seed": 30,
            "timeout_seconds": 60,
        },
        "mesh": {"size_min": 0.1, "size_max": 0.6},
        "mesh_quality": {
            "poor_quality_threshold": 0.6,
            "max_poor_elements": 20,
        },
        "paths": {"calculix": str(calculix_stub), "working_dir": str(working_dir)},
        "output": {
            "input_files": "input_files",
            "output_files": "output_files",
            "data_dir": "data",
        },
    }


class SimulationContextTests(unittest.TestCase):
    def test_valid_configuration_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = make_config(root)

            context = simulation_context.build_ctx(config, root / "config.yaml")

            self.assertEqual(context.calculation_mode, "direct_contraction")
            self.assertEqual(context.calculix_path, (root / "ccx_test").resolve())
            self.assertEqual(context.input_dir, root / "input_files")

    def test_zero_parallel_jobs_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = make_config(root)
            config["simulation"]["n_jobs"] = 0

            with self.assertRaisesRegex(ValueError, "n_jobs"):
                simulation_context.build_ctx(config, root / "config.yaml")

    def test_invalid_lhs_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = make_config(root)
            config["lhs_bounds"]["lower"][2] = config["lhs_bounds"]["upper"][2]

            with self.assertRaisesRegex(ValueError, "LHS lower bound"):
                simulation_context.build_ctx(config, root / "config.yaml")

    def test_missing_calculix_executable_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = make_config(root)
            config["paths"]["calculix"] = "definitely-not-a-calculix-command"

            with self.assertRaisesRegex(FileNotFoundError, "CalculiX executable"):
                simulation_context.build_ctx(config, root / "config.yaml")


if __name__ == "__main__":
    unittest.main()
