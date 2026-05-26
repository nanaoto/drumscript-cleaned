#!/usr/bin/env python3
"""
benchmark_idmt.py — Evaluate DrumScript against IDMT-SMT-Drums dataset.

Usage:
    python scripts/benchmark_idmt.py --dataset /path/to/IDMT-SMT-DRUMS-V2
    python scripts/benchmark_idmt.py --dataset /path/to/IDMT-SMT-DRUMS-V2 --subset RealDrum
    python scripts/benchmark_idmt.py --dataset /path/to/IDMT-SMT-DRUMS-V2 --output results.csv

Dataset download: https://zenodo.org/record/7544164

The dataset must be unzipped. Expected structure:
    IDMT-SMT-DRUMS-V2/
        RealDrum01_00#MIX.wav
        RealDrum01_00#KD.svl   (or .xml)
        RealDrum01_00#SD.svl
        RealDrum01_00#HH.svl
        ...

Metrics (via mir_eval, 50ms onset window):
    Per instrument: Precision / Recall / F-measure
    Averaged across all files in chosen subset.
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── DrumScript imports ────────────────────────────────────────────────────────
try:
    from drumscript.audio_processor.audio_loader import load_audio, normalise_audio
    from drumscript.audio_processor.onset_detector import detect_onsets
    from drumscript.drum_classifier.classify import classify_events
    from drumscript.notation_generator.constants import SAMPLE_RATE
except ImportError as e:
    sys.exit(f"[ERROR] Could not import DrumScript. Is the venv active? ({e})")

# ── mir_eval import ───────────────────────────────────────────────────────────
try:
    import mir_eval
except ImportError:
    sys.exit("[ERROR] mir_eval not installed. Run: pip install mir_eval")


# ─────────────────────────────────────────────────────────────────────────────
# IDMT instrument code → DrumScript label(s)
# HH maps to both open and closed hat; we treat any hi-hat detection as a match.
# ─────────────────────────────────────────────────────────────────────────────
IDMT_TO_DS = {
    "KD": ["kick"],
    "SD": ["snare"],
    "HH": ["hi_hat_closed", "hi_hat_open"],
}

ONSET_WINDOW = 0.050  # 50ms tolerance (standard in ADT literature)


def parse_svl_annotation(svl_path: Path, sr: int = 44100) -> np.ndarray:
    """
    Parse a Sonic Visualiser Layer (.svl) or plain XML annotation file.
    Returns a sorted numpy array of onset times in seconds.

    IDMT SVL format:
        <sv><data><dataset id="1" dimensions="1">
            <point frame="12800" label="KD" />
        </dataset></data></sv>

    The <point frame="..."> attribute is in audio samples at `sr`.
    """
    try:
        tree = ET.parse(svl_path)
        root = tree.getroot()

        frames = []
        # Handle both direct <point> children and nested <dataset><point>
        for point in root.iter("point"):
            frame = point.get("frame")
            if frame is not None:
                frames.append(int(frame))

        if not frames:
            return np.array([])

        onset_times = np.array(sorted(frames)) / sr
        return onset_times

    except ET.ParseError as e:
        print(f"  [WARN] Could not parse {svl_path.name}: {e}")
        return np.array([])


def get_annotation_path(mix_path: Path, instrument_code: str) -> Path | None:
    """
    Given  RealDrum01_00#MIX.wav  and  'KD',
    returns RealDrum01_00#KD.svl  (or .xml if .svl not found).
    """
    stem = mix_path.stem.replace("#MIX", "")
    for ext in [".svl", ".xml", ".SVL", ".XML"]:
        candidate = mix_path.parent / f"{stem}#{instrument_code}{ext}"
        if candidate.exists():
            return candidate
    return None


def run_drumscript(wav_path: Path) -> dict[str, list[float]]:
    """
    Run the full DrumScript pipeline on a WAV file.
    Returns a dict mapping instrument label → list of onset times (seconds).
    """
    y, sr = load_audio(str(wav_path), sr=SAMPLE_RATE)
    y = normalise_audio(y)
    onsets = detect_onsets(y, sr)
    events = classify_events(y, sr, onsets)

    per_instrument: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        for inst in ev["instruments"]:
            per_instrument[inst].append(ev["time_sec"])

    return {k: sorted(v) for k, v in per_instrument.items()}


def evaluate_file(mix_path: Path) -> dict | None:
    """
    Evaluate one MIX file. Returns per-instrument metrics dict or None on error.
    """
    print(f"  {mix_path.name}")

    try:
        predictions = run_drumscript(mix_path)
    except Exception as e:
        print(f"    [ERROR] DrumScript failed: {e}")
        return None

    file_metrics = {}

    for idmt_code, ds_labels in IDMT_TO_DS.items():
        ann_path = get_annotation_path(mix_path, idmt_code)
        if ann_path is None:
            print(f"    [WARN] No annotation found for {idmt_code}, skipping.")
            continue

        ref_onsets = parse_svl_annotation(ann_path)
        if len(ref_onsets) == 0:
            print(f"    [WARN] Empty annotation for {idmt_code}, skipping.")
            continue

        # Merge all DrumScript hits that correspond to this IDMT instrument
        est_times = []
        for label in ds_labels:
            est_times.extend(predictions.get(label, []))
        est_onsets = np.array(sorted(est_times))

        if len(est_onsets) == 0:
            p, r, f = 0.0, 0.0, 0.0
        else:
            p, r, f = mir_eval.onset.f_measure(ref_onsets, est_onsets, window=ONSET_WINDOW)

        file_metrics[idmt_code] = {"precision": p, "recall": r, "f_measure": f,
                                   "n_ref": len(ref_onsets), "n_est": len(est_onsets)}

        print(f"    {idmt_code}: P={p:.3f}  R={r:.3f}  F={f:.3f}  "
              f"(ref={len(ref_onsets)}, est={len(est_onsets)})")

    return file_metrics


def find_mix_files(dataset_dir: Path, subset: str | None) -> list[Path]:
    """Recursively find all #MIX.wav files, optionally filtered by subset name."""
    all_mix = sorted(dataset_dir.rglob("*#MIX.wav"))
    if subset:
        all_mix = [p for p in all_mix if subset.lower() in p.name.lower()]
    return all_mix


def summarise(all_results: list[dict]) -> dict:
    """Compute macro-average F/P/R per instrument across all files."""
    accum = defaultdict(lambda: defaultdict(list))
    for file_res in all_results:
        for inst, metrics in file_res.items():
            for metric, val in metrics.items():
                accum[inst][metric].append(val)

    summary = {}
    for inst, metrics in accum.items():
        summary[inst] = {k: float(np.mean(v)) for k, v in metrics.items()}
    return summary


def write_csv(all_file_results: list[dict], mix_paths: list[Path], output_path: Path):
    instruments = list(IDMT_TO_DS.keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["file"]
        for inst in instruments:
            header += [f"{inst}_P", f"{inst}_R", f"{inst}_F", f"{inst}_n_ref", f"{inst}_n_est"]
        writer.writerow(header)

        for path, res in zip(mix_paths, all_file_results):
            if res is None:
                continue
            row = [path.name]
            for inst in instruments:
                m = res.get(inst, {})
                row += [
                    f"{m.get('precision', ''):.4f}",
                    f"{m.get('recall', ''):.4f}",
                    f"{m.get('f_measure', ''):.4f}",
                    m.get("n_ref", ""),
                    m.get("n_est", ""),
                ]
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark DrumScript on IDMT-SMT-Drums.")
    parser.add_argument("--dataset", required=True, help="Path to unzipped IDMT-SMT-DRUMS-V2 directory")
    parser.add_argument("--subset", default=None,
                        help="Filter by subset name, e.g. 'RealDrum', 'WaveDrum', 'TechnoDrum'")
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Output CSV path (default: benchmark_results.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N files (useful for quick smoke tests)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    mix_files = find_mix_files(dataset_dir, args.subset)
    if not mix_files:
        sys.exit(f"[ERROR] No #MIX.wav files found in {dataset_dir}")

    if args.limit:
        mix_files = mix_files[: args.limit]

    print(f"\nDataset : {dataset_dir}")
    print(f"Subset  : {args.subset or 'all'}")
    print(f"Files   : {len(mix_files)}")
    print(f"Window  : {int(ONSET_WINDOW * 1000)}ms\n")
    print("─" * 60)

    all_results = []
    for mix_path in mix_files:
        result = evaluate_file(mix_path)
        all_results.append(result)

    valid_results = [r for r in all_results if r]

    print("\n" + "─" * 60)
    print(f"Evaluated {len(valid_results)} / {len(mix_files)} files successfully.\n")

    summary = summarise(valid_results)
    print("── SUMMARY (macro-average across files) ──")
    print(f"{'Instrument':<12} {'Precision':>10} {'Recall':>10} {'F-measure':>10}")
    print("-" * 44)
    for inst, m in summary.items():
        print(f"{inst:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f_measure']:>10.3f}")

    output_path = Path(args.output)
    write_csv(all_results, mix_files, output_path)
    print(f"\nPer-file results saved to: {output_path}")


if __name__ == "__main__":
    main()
