"""Runner for UBFC preprocessing + plotting."""

from pathlib import Path
from typing import List, Tuple
import re

from pipeline.run import run_single_file
from pipeline.logger_util import get_logger

files: List[str] = [
    "ubfc_2/s6/bvp_s6_T1.csv",
    "ubfc_2/s6/bvp_s6_T2.csv",
    "ubfc_2/s6/bvp_s6_T3.csv",
]


def infer_subject_task(p: Path) -> Tuple[str, str]:
    """Infer (subject_id, task) from a path.
    - Filename segment: bvp_s6_T1.csv (captures s6, T1)
    - Any 's<number>' directory in the parent chain
    - Task tokens like '_T2' or 'T3' in filename; defaults to T1 if missing.
    """
    name = p.name
    m_subject = re.search(r"_s(\d+)|\bs(\d+)\b", name)
    m_task = re.search(r"_T(\d+)|\bT(\d+)\b", name)
    subject = None
    task = None
    if m_subject:
        subject = m_subject.group(1) or m_subject.group(2)
    if m_task:
        task_num = m_task.group(1) or m_task.group(2)
        task = f"T{task_num}"
    # Fallback: search parent directories for 's<id>'
    if subject is None:
        for parent in p.parents:
            m_dir = re.fullmatch(r"s(\d+)", parent.name)
            if m_dir:
                subject = m_dir.group(1)
                break
    if subject is None:
        raise ValueError(f"Cannot infer subject id from: {p}")
    if task is None:
        task = "T1"  # default if not encoded
    return subject, task


def main():
    log = get_logger(__name__)
    # List of BVP CSV files to process (EDIT THIS LIST ONLY)
    sampling_rate = 64
    save = True

    file_paths = [Path(f) for f in files]
    for fp in file_paths:
        if not fp.exists():
            log.warning(f"File not found, skipping -> {fp}")

    log.info(f"Processing {len(file_paths)} file(s).\n")
    summary = []
    for fp in file_paths:
        if not fp.exists():
            continue
        try:
            subject_id, task = infer_subject_task(fp)
        except ValueError as e:
            log.error(f"Skipping {fp}: {e}")
            continue
        log.info(f"Running pipeline for {fp} (subject {subject_id}, task {task})")
        aligned_df, info, plots = run_single_file(
            file_path=fp,
            subject_id=subject_id,
            task=task,
            sampling_rate=sampling_rate,
            save=save,
        )
        log.info(f"Heart Rate (BPM): {info['avg_heart_rate']:.2f}")
        log.info(f"Plots generated: {len([p for p in plots.values() if p is not None])}")
        summary.append(info)

    log.info("\n" + "=" * 80)
    log.info("Summary")
    for i in summary:
        log.info(
            f"Subject {i['subject_id']} - Task {i['task']}: "
            f"HR={i['avg_heart_rate']:.2f} BPM, Peaks={i['peaks_detected']}, Features Available={i['features_available']}"
        )


if __name__ == "__main__":
    main()
