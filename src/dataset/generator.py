import argparse
import json
import sys
from dataclasses import dataclass
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
from amplitudes.unitarity import unitarity_violation

CLASS_KMATRIX = 0
CLASS_BW = 1

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class PhysicsConfig:
    physics_id: int
    n_res: int
    resonances: list[tuple[float, float]]  # (M, Gamma) shared by BW and K
    poles: list[tuple[float, float]]  # (M, g) with g = sqrt(M * Gamma)


def coupling_from_width(mass: float, gamma: float) -> float:
    """Promotor: g = sqrt(M * Gamma)."""
    return float(np.sqrt(mass * gamma))


class Generator:
    def __init__(self, yaml_path: str | Path):
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = _REPO_ROOT / yaml_path
        with open(yaml_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        np.random.seed(int(self.config["seed"]))
        self.classes_names = ["K-matrix", "Breit-Wigner"]

        self.X: list[np.ndarray] = []
        self.y_class: list[int] = []
        self.y_lambda: list[float] = []
        self.y_n_res: list[int] = []
        self.y_unitarity: list[float] = []
        self.physics_id: list[int] = []
        self.split: list[int] = []
        self.masses: list[np.ndarray] = []
        self.couplings: list[np.ndarray] = []
        self.widths: list[np.ndarray] = []

        self.s: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.s_th: float | None = None
        self.norm_scale: float = 1.0
        self.sanity_report: dict = {}
        self._physics_split: dict[int, int] = {}

    def _mass_bounds(self) -> tuple[float, float]:
        cfg = self.config
        m_min = float(cfg["mass_range"]["min"])
        m_max = float(cfg["mass_range"]["max"])
        if self.s is not None:
            margin = 0.02
            m_min = max(m_min, float(np.sqrt(max(self.s[0] + margin, 0.0))))
            m_max = min(m_max, float(np.sqrt(max(self.s[-1] - margin, 0.0))))
        if m_min >= m_max:
            m_min, m_max = float(cfg["mass_range"]["min"]), float(cfg["mass_range"]["max"])
        return m_min, m_max

    def _width_bounds(self) -> tuple[float, float]:
        cfg = self.config
        return float(cfg["width_range"]["min"]), float(cfg["width_range"]["max"])

    def _clip(self, value: float, lo: float, hi: float) -> float:
        return float(np.clip(value, lo, hi))

    def _sample_resonances(self, n_res: int) -> list[tuple[float, float]]:
        """Sample (M, Γ); for n_res=2 keep the second near the first."""
        m_min, m_max = self._mass_bounds()
        g_min, g_max = self._width_bounds()
        mass_delta = float(self.config.get("mass_similarity_delta", 0.25))
        width_delta = float(self.config.get("width_similarity_delta", 0.08))

        m1 = float(np.random.uniform(m_min, m_max))
        gamma1 = float(np.random.uniform(g_min, g_max))
        resonances = [(m1, gamma1)]

        if n_res >= 2:
            m2 = self._clip(
                float(np.random.uniform(m1 - mass_delta, m1 + mass_delta)),
                m_min,
                m_max,
            )
            # Avoid identical poles (numerical issues)
            if abs(m2 - m1) < 1e-3:
                m2 = self._clip(m1 + (mass_delta if m1 + mass_delta <= m_max else -mass_delta), m_min, m_max)
            gamma2 = self._clip(
                float(np.random.uniform(gamma1 - width_delta, gamma1 + width_delta)),
                g_min,
                g_max,
            )
            resonances.append((m2, gamma2))

        resonances.sort(key=lambda r: r[0])
        return resonances

    def _poles_from_resonances(
        self, resonances: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """K-matrix poles: (M, g) with g = sqrt(M * Γ)."""
        return [(mass, coupling_from_width(mass, gamma)) for mass, gamma in resonances]

    def _sample_physics(self, physics_id: int) -> PhysicsConfig:
        cfg = self.config
        options = list(cfg.get("n_res_options", [1, 2]))
        probs = cfg.get("n_res_probs", None)
        if probs is not None:
            probs = np.asarray(probs, dtype=float)
            probs = probs / probs.sum()
            n_res = int(np.random.choice(options, p=probs))
        else:
            n_res = int(np.random.choice(options))

        resonances = self._sample_resonances(n_res)
        poles = self._poles_from_resonances(resonances)
        return PhysicsConfig(
            physics_id=physics_id,
            n_res=n_res,
            resonances=resonances,
            poles=poles,
        )

    def _is_valid_amplitude(self, t: np.ndarray) -> bool:
        if not np.all(np.isfinite(t)):
            return False
        max_abs = float(self.config.get("max_abs_t", 50.0))
        return float(np.max(np.abs(t))) <= max_abs

    def _append_sample(
        self,
        t: np.ndarray,
        class_id: int,
        lam: float,
        phys: PhysicsConfig,
        split_id: int,
    ) -> bool:
        if not self._is_valid_amplitude(t):
            return False

        masses = np.full(2, np.nan, dtype=np.float64)
        couplings = np.full(2, np.nan, dtype=np.float64)
        widths = np.full(2, np.nan, dtype=np.float64)
        for i, (m, g) in enumerate(phys.poles):
            masses[i] = m
            couplings[i] = g
        for i, (_m, gamma) in enumerate(phys.resonances):
            widths[i] = gamma

        self.X.append(t_to_vector(t))
        self.y_class.append(class_id)
        self.y_lambda.append(float(lam))
        self.y_n_res.append(int(phys.n_res))
        self.y_unitarity.append(float(unitarity_violation(t, rho=None)))
        self.physics_id.append(int(phys.physics_id))
        self.split.append(int(split_id))
        self.masses.append(masses)
        self.couplings.append(couplings)
        self.widths.append(widths)
        return True

    def _assign_physics_splits(self, n_physics: int) -> dict[int, int]:
        split_cfg = self.config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
        train_f = float(split_cfg.get("train", 0.8))
        val_f = float(split_cfg.get("val", 0.1))
        test_f = float(split_cfg.get("test", 0.1))
        total = train_f + val_f + test_f
        if not np.isclose(total, 1.0):
            raise ValueError(f"split fractions must sum to 1, got {total}")

        ids = np.arange(n_physics)
        rng = np.random.default_rng(int(self.config["seed"]))
        rng.shuffle(ids)

        n_train = int(n_physics * train_f)
        n_val = int(n_physics * val_f)
        mapping: dict[int, int] = {}
        for i, pid in enumerate(ids):
            if i < n_train:
                mapping[int(pid)] = SPLIT_TRAIN
            elif i < n_train + n_val:
                mapping[int(pid)] = SPLIT_VAL
            else:
                mapping[int(pid)] = SPLIT_TEST
        return mapping

    def _rollback_to(self, size: int) -> None:
        self.X = self.X[:size]
        self.y_class = self.y_class[:size]
        self.y_lambda = self.y_lambda[:size]
        self.y_n_res = self.y_n_res[:size]
        self.y_unitarity = self.y_unitarity[:size]
        self.physics_id = self.physics_id[:size]
        self.split = self.split[:size]
        self.masses = self.masses[:size]
        self.couplings = self.couplings[:size]
        self.widths = self.widths[:size]

    def _emit_physics(self, phys: PhysicsConfig, split_id: int) -> bool:
        """Emit K-matrix + Breit-Wigner for one physics config."""
        expected = 2
        start = len(self.X)

        t_k = generate_kmatrix(self.s, phys.poles)
        t_bw = generate_breit_wigner(self.s, phys.resonances)

        ok = self._append_sample(t_k, CLASS_KMATRIX, 0.0, phys, split_id)
        ok = self._append_sample(t_bw, CLASS_BW, 1.0, phys, split_id) and ok

        if not ok or (len(self.X) - start) != expected:
            self._rollback_to(start)
            return False
        return True

    def _apply_normalization(self) -> None:
        mode = str(self.config.get("normalization", "none")).lower()
        if mode in ("none", "", "null") or len(self.X) == 0:
            self.norm_scale = 1.0
            return

        X = np.asarray(self.X, dtype=np.float64)
        split = np.asarray(self.split)
        train_mask = split == SPLIT_TRAIN
        if not np.any(train_mask):
            train_mask = np.ones(len(X), dtype=bool)

        if mode == "rms":
            scale = float(np.sqrt(np.mean(X[train_mask] ** 2)))
        elif mode == "max_abs":
            scale = float(np.max(np.abs(X[train_mask])))
        else:
            raise ValueError(f"Unknown normalization: {mode}")

        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1.0

        self.norm_scale = scale
        self.X = [row / scale for row in self.X]

    def _build_sanity_report(self) -> dict:
        y_class = np.asarray(self.y_class)
        y_unitarity = np.asarray(self.y_unitarity)
        y_n_res = np.asarray(self.y_n_res)
        split = np.asarray(self.split)
        physics_id = np.asarray(self.physics_id)

        report: dict = {
            "n_samples": int(len(self.X)),
            "n_physics": int(len(np.unique(physics_id))) if len(physics_id) else 0,
            "feature_dim": int(len(self.X[0])) if self.X else 0,
            "n_points": int(self.config["n_points"]),
            "norm_scale": float(self.norm_scale),
            "normalization": str(self.config.get("normalization", "none")),
            "class_counts": {},
            "split_counts": {},
            "n_res_counts": {},
            "unitarity_by_class": {},
            "checks": {},
        }

        for c_idx, name in enumerate(self.classes_names):
            mask = y_class == c_idx
            report["class_counts"][name] = int(np.sum(mask))
            if np.any(mask):
                report["unitarity_by_class"][name] = {
                    "mean": float(np.mean(y_unitarity[mask])),
                    "median": float(np.median(y_unitarity[mask])),
                }

        for s_id, s_name in enumerate(SPLIT_NAMES):
            report["split_counts"][s_name] = int(np.sum(split == s_id))

        for n in np.unique(y_n_res):
            report["n_res_counts"][str(int(n))] = int(np.sum(y_n_res == n))

        k_mean = report["unitarity_by_class"].get("K-matrix", {}).get("mean", np.nan)
        bw_mean = report["unitarity_by_class"].get("Breit-Wigner", {}).get("mean", np.nan)

        report["checks"] = {
            "kmatrix_near_unitary": bool(np.isfinite(k_mean) and k_mean < 1e-2),
            "bw_violates_unitarity": bool(np.isfinite(bw_mean) and bw_mean > 1e-2),
            "kmatrix_unitarity_mean": float(k_mean) if np.isfinite(k_mean) else None,
            "bw_unitarity_mean": float(bw_mean) if np.isfinite(bw_mean) else None,
        }
        return report

    def generate(self) -> None:
        self.s_th, self.s = get_grid(
            self.config["m_pi"], self.config["s_max"], self.config["n_points"]
        )
        self.rho = get_phase_space(self.s_th, self.s)

        n_physics = int(self.config.get("n_physics", self.config.get("n_samples_per_class", 400)))
        self._physics_split = self._assign_physics_splits(n_physics)

        max_retries = int(self.config.get("max_physics_retries", 20))
        emitted_physics = 0
        attempt = 0
        while emitted_physics < n_physics and attempt < n_physics * max_retries:
            phys = self._sample_physics(emitted_physics)
            split_id = self._physics_split[emitted_physics]
            if self._emit_physics(phys, split_id):
                emitted_physics += 1
            attempt += 1

        if emitted_physics < n_physics:
            print(
                f"Uwaga: wygenerowano tylko {emitted_physics}/{n_physics} konfiguracji fizyki "
                f"(odrzucenia numeryczne)."
            )

        self._apply_normalization()
        self._shuffle()
        self.sanity_report = self._build_sanity_report()
        self._print_summary()

    def _shuffle(self) -> None:
        n = len(self.X)
        order = np.random.permutation(n)
        self.X = [self.X[i] for i in order]
        self.y_class = [self.y_class[i] for i in order]
        self.y_lambda = [self.y_lambda[i] for i in order]
        self.y_n_res = [self.y_n_res[i] for i in order]
        self.y_unitarity = [self.y_unitarity[i] for i in order]
        self.physics_id = [self.physics_id[i] for i in order]
        self.split = [self.split[i] for i in order]
        self.masses = [self.masses[i] for i in order]
        self.couplings = [self.couplings[i] for i in order]
        self.widths = [self.widths[i] for i in order]

    def _print_summary(self) -> None:
        r = self.sanity_report
        print(
            f"Dataset: {r['n_samples']} próbek | {r['n_physics']} physics_id | "
            f"dim={r['feature_dim']} | norm_scale={r['norm_scale']:.4g}"
        )
        print(f"Klasy: {r['class_counts']}")
        print(f"Split: {r['split_counts']}")
        print(f"Checks: {r['checks']}")

    def save(self, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            output_path = _REPO_ROOT / "data" / "amplitudes.npz"
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = _REPO_ROOT / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        max_n_res = 2
        masses = np.full((len(self.X), max_n_res), np.nan, dtype=np.float64)
        couplings = np.full((len(self.X), max_n_res), np.nan, dtype=np.float64)
        widths = np.full((len(self.X), max_n_res), np.nan, dtype=np.float64)
        for i in range(len(self.X)):
            masses[i] = self.masses[i]
            couplings[i] = self.couplings[i]
            widths[i] = self.widths[i]

        np.savez(
            output_path,
            X=np.asarray(self.X, dtype=np.float32),
            y_class=np.asarray(self.y_class, dtype=np.int32),
            y_lambda=np.asarray(self.y_lambda, dtype=np.float32),
            y_n_res=np.asarray(self.y_n_res, dtype=np.int32),
            y_unitarity=np.asarray(self.y_unitarity, dtype=np.float32),
            physics_id=np.asarray(self.physics_id, dtype=np.int32),
            split=np.asarray(self.split, dtype=np.int32),
            masses=masses.astype(np.float32),
            couplings=couplings.astype(np.float32),
            widths=widths.astype(np.float32),
            seed=np.asarray([self.config["seed"]], dtype=np.int32),
            s_grid=np.asarray(self.s, dtype=np.float64),
            rho=np.asarray(self.rho, dtype=np.float64),
            s_th=np.asarray([self.s_th], dtype=np.float64),
            norm_scale=np.asarray([self.norm_scale], dtype=np.float64),
            class_names=np.asarray(self.classes_names),
            split_names=np.asarray(SPLIT_NAMES),
        )
        print(f"Zapisano dataset: {output_path}")
        return output_path

    def save_sanity_report(self, output_dir: str | Path | None = None) -> Path:
        if output_dir is None:
            output_dir = _REPO_ROOT / "results" / "sanity"
        else:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = _REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.sanity_report, f, indent=2)
        print(f"Zapisano raport sanity: {out_file}")
        return out_file

    def draw(self, output_dir: str | Path | None = None) -> Path:
        if output_dir is None:
            output_dir = _REPO_ROOT / "results" / "sanity"
        else:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = _REPO_ROOT / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        y_class = np.asarray(self.y_class)
        y_unitarity = np.asarray(self.y_unitarity)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for c_idx, c_name in enumerate(self.classes_names):
            candidates = np.where(y_class == c_idx)[0]
            if candidates.size == 0:
                continue
            sample_idx = int(np.random.choice(candidates))
            vec = self.X[sample_idx]
            n = len(vec) // 2
            t_comp = vec[:n] + 1j * vec[n:]
            axes[0].plot(self.s, np.real(t_comp), label=c_name)
            axes[1].plot(self.s, np.imag(t_comp), label=c_name)
            axes[2].plot(self.s, np.abs(t_comp) ** 2, label=c_name)

        axes[0].set_title("Re(T)")
        axes[1].set_title("Im(T)")
        axes[2].set_title("|T|^2")
        for ax in axes:
            ax.set_xlabel("s")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = output_dir / "dataset_sample.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for c_idx, c_name in enumerate(self.classes_names):
            mask = y_class == c_idx
            ax2.scatter(
                np.full(np.sum(mask), c_idx),
                y_unitarity[mask],
                alpha=0.35,
                s=12,
                label=c_name,
            )
        ax2.set_xticks(range(len(self.classes_names)))
        ax2.set_xticklabels(self.classes_names)
        ax2.set_ylabel("Unitarity violation")
        ax2.set_title("Unitarity by class")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        unit_file = output_dir / "unitarity_by_class.png"
        fig2.savefig(unit_file, dpi=150)
        plt.close(fig2)

        print(f"Zapisano figurę sanity: {out_file}")
        print(f"Zapisano unitarity plot: {unit_file}")
        return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate amplitude dataset")
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
    generator.save_sanity_report()
    generator.draw()
