import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")


class Graphs:
    def __init__(self, file_path: Optional[Path] = None):
        """Optional base output directory."""
        self.file_path = Path(file_path) if file_path else Path(".")
        (self.file_path / "graphs").mkdir(parents=True, exist_ok=True)

    def set_filepath(self, file_path: Path):
        """Set/replace base output directory."""
        self.file_path = Path(file_path)
        (self.file_path / "graphs").mkdir(parents=True, exist_ok=True)

    def _ensure_dir(self):
        if self.file_path is None:
            raise RuntimeError(
                "Output directory not set. Call set_filepath or pass file_path to constructor."
            )

    def plot_raw_bvp(
        self,
        bvp_signal,
        sampling_rate=64,
        title="Raw BVP Signal",
        out_name: Optional[str] = None,
    ):
        self._ensure_dir()
        time = np.arange(len(bvp_signal)) / sampling_rate
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(time, bvp_signal, linewidth=0.5, color="#2E86AB")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("BVP Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        prefix = f"{out_name}_" if out_name else ""
        out_path = self.file_path / "graphs" / f"{prefix}raw_bvp.png"
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_signal_comparison(
        self, original, cleaned, sampling_rate=64, out_name: Optional[str] = None
    ):
        self._ensure_dir()
        time = np.arange(len(original)) / sampling_rate
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        axes[0].plot(time, original, linewidth=0.5, color="#A23B72", alpha=0.8)
        axes[0].set_title("Original BVP Signal")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(time, cleaned, linewidth=0.5, color="#2E86AB", alpha=0.8)
        axes[1].set_title("Cleaned BVP Signal")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        prefix = f"{out_name}_" if out_name else ""
        out_path = self.file_path / "graphs" / f"{prefix}signal_comparison.png"
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_peaks_detection(
        self,
        bvp_cleaned,
        peaks,
        sampling_rate=64,
        zoom_start=0,
        zoom_duration=10,
        out_name: Optional[str] = None,
    ):
        self._ensure_dir()
        time = np.arange(len(bvp_cleaned)) / sampling_rate
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes[0].plot(
            time, bvp_cleaned, linewidth=0.5, color="#2E86AB", label="BVP Signal"
        )
        axes[0].scatter(
            time[peaks],
            bvp_cleaned[peaks],
            color="#F18F01",
            s=50,
            zorder=5,
            label=f"Peaks (n={len(peaks)})",
        )
        axes[0].set_title("Peak Detection - Full Signal")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        zoom_end = zoom_start + zoom_duration
        zs = int(zoom_start * sampling_rate)
        ze = int(zoom_end * sampling_rate)
        zoom_peaks = peaks[(peaks >= zs) & (peaks < ze)]
        axes[1].plot(time[zs:ze], bvp_cleaned[zs:ze], linewidth=1.2, color="#2E86AB")
        axes[1].scatter(
            time[zoom_peaks], bvp_cleaned[zoom_peaks], color="#F18F01", s=90, zorder=5
        )
        axes[1].set_title(f"Peak Detection - Zoomed ({zoom_start}-{zoom_end}s)")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        prefix = f"{out_name}_" if out_name else ""
        out_path = self.file_path / "graphs" / f"{prefix}peak_detection.png"
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_rr_intervals(
        self, rr_intervals, filtered_rr, out_name: Optional[str] = None
    ):
        self._ensure_dir()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(
            rr_intervals,
            linewidth=1,
            color="#A23B72",
            marker="o",
            markersize=3,
            alpha=0.7,
        )
        axes[0, 0].axhline(y=300, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].axhline(y=2000, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].set_title("Original RR Intervals")
        axes[0, 0].set_xlabel("Beat Number")
        axes[0, 0].set_ylabel("RR (ms)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(
            filtered_rr,
            linewidth=1,
            color="#2E86AB",
            marker="o",
            markersize=3,
            alpha=0.7,
        )
        axes[0, 1].set_title("Filtered RR Intervals")
        axes[0, 1].set_xlabel("Beat Number")
        axes[0, 1].set_ylabel("RR (ms)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].hist(
            rr_intervals, bins=50, color="#A23B72", alpha=0.7, edgecolor="black"
        )
        axes[1, 0].axvline(x=300, color="red", linestyle="--", alpha=0.7)
        axes[1, 0].axvline(x=2000, color="red", linestyle="--", alpha=0.7)
        axes[1, 0].set_title("Distribution - Original")
        axes[1, 0].set_xlabel("RR (ms)")
        axes[1, 0].set_ylabel("Freq")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].hist(
            filtered_rr, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black"
        )
        axes[1, 1].set_title("Distribution - Filtered")
        axes[1, 1].set_xlabel("RR (ms)")
        axes[1, 1].set_ylabel("Freq")
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        prefix = f"{out_name}_" if out_name else ""
        out_path = self.file_path / "graphs" / f"{prefix}rr_intervals.png"
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_hrv_features_summary(
        self, hrv_features, top_n=20, out_name: Optional[str] = None
    ):
        self._ensure_dir()
        if hrv_features is None or hrv_features.empty:
            return None
        features = hrv_features.iloc[0]
        numeric_features = features[
            features.notna() & (features != np.inf) & (features != -np.inf)
        ]
        sorted_features = (
            numeric_features.abs().sort_values(ascending=False).head(top_n)
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        ax.barh(
            range(len(sorted_features)),
            numeric_features[sorted_features.index],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features.index)
        ax.set_xlabel("Value")
        ax.set_title(f"Top {top_n} HRV Features")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        prefix = f"{out_name}_" if out_name else ""
        out_path = self.file_path / "graphs" / f"{prefix}hrv_features.png"
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_feature_coverage(
        self, available_count, missing_count, out_name: Optional[str] = None
    ):
        self._ensure_dir()
        if available_count == 0 and missing_count == 0:
            return None
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ["Available", "Missing"]
        sizes = [available_count, missing_count]
        colors = ["#2E86AB", "#F18F01"]
        ax.pie(
            sizes,
            explode=(0.05, 0.05),
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax.set_title("Feature Coverage")
        plt.tight_layout()
        prefix = f"{out_name}_" if out_name else ""
        out_path = self.file_path / "graphs" / f"{prefix}feature_coverage.png"
        plt.savefig(out_path)
        plt.close(fig)
        return out_path
