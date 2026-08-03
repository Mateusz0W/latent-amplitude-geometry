import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from amplitudes.breit_wigner import generate_breit_wigner
from amplitudes.kinematics import get_grid, get_phase_space, t_to_vector
from amplitudes.kmatrix import generate_kmatrix
from amplitudes.mix import generate_mix
from amplitudes.unitarity import unitarity_violation


class Generator:
    def __init__(self, yaml_path: str | Path):
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = _REPO_ROOT / yaml_path
        with open(yaml_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        np.random.seed(self.config["seed"])
        self.classes_names = ["K-matrix", "Breit-Wigner", "Mix"]
        self.X: list[np.ndarray] = []
        self.y_class: list[int] = []
        self.y_lambda: list[float] = []
        self.y_n_res: list[int] = []
        self.y_unitarity: list[float] = []
        self.s: np.ndarray | None = None
        self.rho: np.ndarray | None = None

    def _sample_poles(self, n_res: int) -> list[tuple[float, float]]:
        cfg = self.config
        return [
            (
                np.random.uniform(cfg["mass_range"]["min"], cfg["mass_range"]["max"]),
                np.random.uniform(cfg["coupling_range"]["min"], cfg["coupling_range"]["max"]),
            )
            for _ in range(n_res)
        ]

    def _resonances_from_poles(
        self, poles: list[tuple[float, float]]
    ) -> list[tuple[float, float, float]]:
        """Same masses as K-matrix poles; independent width/coupling for BW."""
        cfg = self.config
        return [
            (
                mass,
                np.random.uniform(cfg["width_range"]["min"], cfg["width_range"]["max"]),
                np.random.uniform(cfg["coupling_range"]["min"], cfg["coupling_range"]["max"]),
            )
            for mass, _ in poles
        ]

    def generate(self) -> None:
        s_th, self.s = get_grid(
            self.config["m_pi"], self.config["s_max"], self.config["n_points"]
        )
        self.rho = get_phase_space(s_th, self.s)

        for idx, _name in enumerate(self.classes_names):
            for _ in range(self.config["n_samples_per_class"]):
                n_res = int(np.random.choice([1, 2]))
                poles = self._sample_poles(n_res)
                # Shared masses between K and BW so Mix is a controlled λ trajectory
                resonances = self._resonances_from_poles(poles)

                t_k = generate_kmatrix(self.s, self.rho, poles)
                t_bw = generate_breit_wigner(self.s, resonances)

                if idx == 0:
                    lam = 0.0
                    t_final = t_k
                elif idx == 1:
                    lam = 1.0
                    t_final = t_bw
                else:
                    lam = float(np.random.choice([0.25, 0.5, 0.75]))
                    t_final = generate_mix(t_k, t_bw, lam)

                self.X.append(t_to_vector(t_final))
                self.y_class.append(idx)
                self.y_lambda.append(lam)
                self.y_n_res.append(n_res)
                self.y_unitarity.append(unitarity_violation(t_final, self.rho))

        self._shuffle()

    def _shuffle(self) -> None:
        """Shuffle samples so classes are not stored in contiguous blocks."""
        n = len(self.X)
        order = np.random.permutation(n)
        self.X = [self.X[i] for i in order]
        self.y_class = [self.y_class[i] for i in order]
        self.y_lambda = [self.y_lambda[i] for i in order]
        self.y_n_res = [self.y_n_res[i] for i in order]
        self.y_unitarity = [self.y_unitarity[i] for i in order]

    def save(self, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            output_path = _REPO_ROOT / "data" / "amplitudes.npz"
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = _REPO_ROOT / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            X=np.asarray(self.X),
            y_class=np.asarray(self.y_class),
            y_lambda=np.asarray(self.y_lambda),
            y_n_res=np.asarray(self.y_n_res),
            y_unitarity=np.asarray(self.y_unitarity),
            seed=np.asarray([self.config["seed"]]),
            s_grid=self.s,
            rho=self.rho,
            class_names=np.asarray(self.classes_names),
        )
        print(f"Zapisano dataset: {output_path}")
        return output_path

    def draw(self, output_dir: str | Path | None = None) -> Path:
        if output_dir is None:
            output_dir = _REPO_ROOT / "results" / "sanity"
        else:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = _REPO_ROOT / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        y_class = np.asarray(self.y_class)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for c_idx, c_name in enumerate(self.classes_names):
            candidates = np.where(y_class == c_idx)[0]
            sample_idx = int(np.random.choice(candidates))
            vec = self.X[sample_idx]
            n = len(vec) // 2
            t_comp = vec[:n] + 1j * vec[n:]
            label = f"{c_name} (λ={self.y_lambda[sample_idx]:.2f})"

            axes[0].plot(self.s, np.real(t_comp), label=label)
            axes[1].plot(self.s, np.imag(t_comp), label=label)
            axes[2].plot(self.s, np.abs(t_comp) ** 2, label=label)

        axes[0].set_title("Re(T)")
        axes[1].set_title("Im(T)")
        axes[2].set_title("|T|^2")
        for ax in axes:
            ax.set_xlabel("s")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_file = output_dir / "dataset_sample.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Zapisano figurę sanity: {out_file}")
        return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MVP amplitude dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/amplitudes.npz",
        help="Output .npz path (relative to repo root or absolute)",
    )
    args = parser.parse_args()

    generator = Generator(args.config)
    generator.generate()
    generator.save(args.output)
    generator.draw()
