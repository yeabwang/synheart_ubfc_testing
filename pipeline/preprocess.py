import numpy as np
import pandas as pd
import warnings
import traceback
import neurokit2 as nk
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from pipeline.logger_util import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)


class Preprocessing:
    """BVP -> HRV preprocessing pipeline.

    Responsibilities:
    - Load and clean BVP signal
    - Detect peaks and derive RR intervals
    - Filter RR intervals and compute heart rate
    - Extract HRV features (NeuroKit2)
    - Align extracted features to a training feature set
    - Persist results (features + HR metadata) to disk
    """

    def __init__(self, training_features: List[str], file_path: Optional[Path] = None):
        self.training_features = training_features
        self.file_path = Path(file_path) if file_path else Path(".")
        # Ensure output subdirectories
        (self.file_path / "graphs").mkdir(parents=True, exist_ok=True)
        (self.file_path / "preprocessed").mkdir(parents=True, exist_ok=True)

    def set_filepath(self, file_path: Path) -> None:
        """Set/replace base output directory."""
        self.file_path = Path(file_path)
        (self.file_path / "graphs").mkdir(parents=True, exist_ok=True)
        (self.file_path / "preprocessed").mkdir(parents=True, exist_ok=True)

    def _ensure_dir(self) -> None:
        if self.file_path is None:
            raise RuntimeError(
                "Output directory not set. Call set_filepath or pass file_path to constructor."
            )

    # -----------------------------------------------------
    # Core steps
    # -----------------------------------------------------
    def load_bvp_signal(self, file_path: Path) -> np.ndarray:
        """Load BVP signal from CSV file."""
        data = pd.read_csv(file_path, header=None)
        return data.values.flatten()

    def clean_bvp_signal(
        self, bvp_signal: np.ndarray, sampling_rate: int = 64
    ) -> np.ndarray:
        """Clean BVP signal using NeuroKit2."""
        return nk.ppg_clean(bvp_signal, sampling_rate=sampling_rate)

    def detect_peaks(
        self, bvp_cleaned: np.ndarray, sampling_rate: int = 64
    ) -> np.ndarray:
        """Detect peaks in cleaned BVP signal."""
        peaks_dict = nk.ppg_peaks(bvp_cleaned, sampling_rate=sampling_rate)
        return peaks_dict[1]["PPG_Peaks"]

    def calculate_rr_intervals(
        self, peaks: np.ndarray, sampling_rate: int = 64
    ) -> np.ndarray:
        """Calculate RR intervals (ms) from peak indices."""
        rr_samples = np.diff(peaks)
        return (rr_samples / sampling_rate) * 1000.0

    def filter_rr_intervals(
        self, rr_intervals: np.ndarray, min_rr: int = 300, max_rr: int = 2000
    ) -> np.ndarray:
        """Filter physiologically implausible RR intervals."""
        valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        filtered = rr_intervals[valid_mask]
        total = len(rr_intervals)
        removed_count = total - len(filtered)
        percent = (removed_count / total * 100.0) if total > 0 else 0.0
        log.info(f"Removed {removed_count} outlier RR intervals ({percent:.2f}%)")
        return filtered

    def compute_heart_rate(self, filtered_rr: np.ndarray) -> float:
        """Compute average heart rate (BPM) from filtered RR intervals."""
        if len(filtered_rr) == 0:
            return float("nan")
        return 60000.0 / np.mean(filtered_rr)

    def extract_hrv_features(
        self, rr_intervals_ms: np.ndarray, sampling_rate: int = 64
    ) -> Optional[pd.DataFrame]:
        """Extract HRV indices using NeuroKit2 given RR intervals in ms.

        Returns full HRV feature dataframe (not yet filtered to training set).
        """
        try:
            rr_intervals_ms = np.array(rr_intervals_ms).flatten()
            rr_seconds = rr_intervals_ms / 1000.0
            peak_positions = np.round(
                np.cumsum(np.concatenate([[0], rr_seconds])) * sampling_rate
            ).astype(int)
            hrv_indices = nk.hrv(
                peaks=peak_positions, sampling_rate=sampling_rate, show=False
            )
            log.info(f"Extracted {len(hrv_indices.columns)} HRV features.")
            return hrv_indices
        except Exception as e:
            log.exception(f"Error extracting HRV features: {e}")
            return None

    def align_features_with_training(
        self, hrv_features: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Align extracted features with the training feature set, filling missing with NaN."""
        required = self.training_features
        if hrv_features is None:
            aligned_df = pd.DataFrame(columns=required)
            aligned_df.loc[0] = np.nan
            log.error(
                "HRV feature extraction failed. All training features marked missing."
            )
            return aligned_df, [], required
        available = [f for f in required if f in hrv_features.columns]
        missing = [f for f in required if f not in hrv_features.columns]
        aligned_df = pd.DataFrame()
        for feat in required:
            aligned_df[feat] = (
                hrv_features[feat] if feat in hrv_features.columns else np.nan
            )
        log.info(f"Feature alignment: {len(available)} available, {len(missing)} missing.")
        return aligned_df, available, missing

    def save_features(self, df: pd.DataFrame, info: Dict, filename: str) -> Path:
        """Save dataframe with processing info columns to preprocessed directory."""
        self._ensure_dir()
        out_dir = self.file_path / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        df_to_save = df.copy()
        # Append metadata columns (avoid overwriting existing)
        for k, v in info.items():
            if k not in df_to_save.columns:
                df_to_save[k] = v
        df_to_save.to_csv(out_path, index=False)
        log.info(f"Saved preprocessed features -> {out_path}")
        return out_path
