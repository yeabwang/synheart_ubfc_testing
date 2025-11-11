from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd

from pipeline.graphs import Graphs
from pipeline.preprocess import Preprocessing
from pipeline.logger_util import get_logger

TRAINING_FEATURES = [
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_MedianNN",
    "HRV_MadNN",
    "HRV_MCVNN",
    "HRV_IQRNN",
    "HRV_SDRMSSD",
    "HRV_Prc20NN",
    "HRV_Prc80NN",
    "HRV_pNN50",
    "HRV_pNN20",
    "HRV_MinNN",
    "HRV_MaxNN",
    "HRV_HTI",
    "HRV_TINN",
    "HRV_LF",
    "HRV_HF",
    "HRV_VHF",
    "HRV_TP",
    "HRV_LFHF",
    "HRV_LFn",
    "HRV_HFn",
    "HRV_LnHF",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",
    "HRV_CSI",
    "HRV_CVI",
    "HRV_CSI_Modified",
    "HRV_PIP",
    "HRV_IALS",
    "HRV_PSS",
    "HRV_PAS",
    "HRV_GI",
    "HRV_SI",
    "HRV_AI",
    "HRV_PI",
    "HRV_C1d",
    "HRV_C1a",
    "HRV_SD1d",
    "HRV_SD1a",
    "HRV_C2d",
    "HRV_C2a",
    "HRV_SD2d",
    "HRV_SD2a",
    "HRV_Cd",
    "HRV_Ca",
    "HRV_SDNNd",
    "HRV_SDNNa",
    "HRV_DFA_alpha1",
    "HRV_MFDFA_alpha1_Width",
    "HRV_MFDFA_alpha1_Peak",
    "HRV_MFDFA_alpha1_Mean",
    "HRV_MFDFA_alpha1_Max",
    "HRV_MFDFA_alpha1_Delta",
    "HRV_MFDFA_alpha1_Asymmetry",
    "HRV_MFDFA_alpha1_Fluctuation",
    "HRV_MFDFA_alpha1_Increment",
    "HRV_DFA_alpha2",
    "HRV_MFDFA_alpha2_Width",
    "HRV_MFDFA_alpha2_Peak",
    "HRV_MFDFA_alpha2_Mean",
    "HRV_MFDFA_alpha2_Max",
    "HRV_MFDFA_alpha2_Delta",
    "HRV_MFDFA_alpha2_Asymmetry",
    "HRV_MFDFA_alpha2_Fluctuation",
    "HRV_MFDFA_alpha2_Increment",
    "HRV_ApEn",
    "HRV_SampEn",
    "HRV_ShanEn",
    "HRV_FuzzyEn",
    "HRV_MSEn",
    "HRV_CMSEn",
    "HRV_RCMSEn",
    "HRV_CD",
    "HRV_HFD",
    "HRV_KFD",
    "HRV_LZC",
]


def run_single_file(
    file_path: Path,
    subject_id: str,
    task: str,
    sampling_rate: int = 64,
    base_output: Optional[Path] = None,
    save: bool = True,
):
    """Run preprocessing and generate plots step-by-step for one file.

    Returns (aligned_features_df, processing_info_dict, plot_paths_dict).
    """

    log = get_logger(__name__)
    # Choose subject-specific output directory (prefer parent named s<subject_id>)
    def _find_subject_dir(p: Path, sid: str) -> Path:
        target = f"s{sid}"
        for parent in [p.parent] + list(p.parents):
            if parent.name == target:
                return parent
        return p.parent

    subject_dir = _find_subject_dir(file_path, subject_id)
    base = base_output or subject_dir
    proc = Preprocessing(training_features=TRAINING_FEATURES, file_path=base)
    graphs = Graphs(file_path=base)
    prefix = f"s{subject_id}_{task}"

    # 1) Load
    bvp_signal = proc.load_bvp_signal(file_path)
    plot_paths: Dict[str, Path] = {}
    log.info(f"[bold]Processing[/bold] subject={subject_id} task={task} file={file_path}")
    plot_paths["raw_bvp"] = graphs.plot_raw_bvp(
        bvp_signal,
        sampling_rate=sampling_rate,
        title=f"Subject {subject_id} - {task} - Raw BVP",
        out_name=prefix,
    )

    # 2) Clean
    bvp_cleaned = proc.clean_bvp_signal(bvp_signal, sampling_rate=sampling_rate)
    plot_paths["signal_comparison"] = graphs.plot_signal_comparison(
        bvp_signal, bvp_cleaned, sampling_rate=sampling_rate, out_name=prefix
    )

    # 3) Peaks
    peaks = proc.detect_peaks(bvp_cleaned, sampling_rate=sampling_rate)
    plot_paths["peak_detection"] = graphs.plot_peaks_detection(
        bvp_cleaned,
        peaks,
        sampling_rate=sampling_rate,
        zoom_start=5,
        zoom_duration=10,
        out_name=prefix,
    )

    # 4) RR intervals and filtering
    rr_intervals = proc.calculate_rr_intervals(peaks, sampling_rate=sampling_rate)
    filtered_rr = proc.filter_rr_intervals(rr_intervals)
    plot_paths["rr_intervals"] = graphs.plot_rr_intervals(
        rr_intervals, filtered_rr, out_name=prefix
    )
    heart_rate = proc.compute_heart_rate(filtered_rr)
    log.info(f"RR filtered: {len(filtered_rr)} | HR: {heart_rate:.2f} BPM")

    # 5) HRV features
    hrv_features_full = proc.extract_hrv_features(
        filtered_rr, sampling_rate=sampling_rate
    )
    if hrv_features_full is not None:
        plot_paths["hrv_features"] = graphs.plot_hrv_features_summary(
            hrv_features_full, top_n=20, out_name=prefix
        )

    # 6) Align and coverage
    aligned_features, available, missing = proc.align_features_with_training(
        hrv_features_full
    )
    plot_paths["feature_coverage"] = graphs.plot_feature_coverage(
        len(available), len(missing), out_name=prefix
    )

    # Augment with HR summary
    aligned_features["HeartRateBPM"] = heart_rate
    aligned_features["RR_Mean_ms"] = (
        np.mean(filtered_rr) if len(filtered_rr) else np.nan
    )
    aligned_features["RR_SD_ms"] = np.std(filtered_rr) if len(filtered_rr) else np.nan
    aligned_features["RR_Count"] = len(filtered_rr)

    duration_sec = len(bvp_signal) / sampling_rate
    info: Dict = {
        "subject_id": subject_id,
        "task": task,
        "signal_length": len(bvp_signal),
        "duration_sec": duration_sec,
        "peaks_detected": int(len(peaks)),
        "avg_heart_rate": heart_rate,
        "rr_intervals_total": int(len(rr_intervals)),
        "rr_intervals_filtered": int(len(filtered_rr)),
        "features_available": int(len(available)),
        "features_missing": int(len(missing)),
        "extraction_success": hrv_features_full is not None,
    }

    if save:
        filename = f"preprocessed_s{subject_id}_{task}.csv"
        saved_path = proc.save_features(aligned_features, info, filename)
        log.info(f"Saved features -> {saved_path}")

    return aligned_features, info, plot_paths
