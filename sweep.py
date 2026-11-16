#!/usr/bin/env python3
"""
sweep_v7.py — jalankan beberapa varian predict_possession_v7.py sekaligus.

Tiap varian disimpan dengan nama tim sendiri (evaluation/results/<nama>.csv), jadi
semuanya muncul berdampingan di satu scoreboard dan tidak saling menimpa.

Dari root folder repositori more-than-a-hack-2027:
    python sweep_v7.py            # jalankan semua varian
    python sweep_v7.py --dry-run  # hanya cetak perintah, tidak menjalankan
    python evaluation/run_evaluation.py

Setiap varian membaca ulang data tracking (2 pass), jadi total waktu = jumlah varian
x waktu satu kali jalan. Gunakan --only untuk memilih sebagian.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "predict_possession_v7.py")

# (nama tim, argumen tambahan). Default v7 = window-team 50, restart-delay 12.
# Tujuan: cari titik tengah antara win50 (spell F1 tinggi) dan win60 (macro_f1 tinggi).
VARIANTS = [
    ("v7 win50", []),
    ("v7 win52", ["--window-team", "52"]),
    ("v7 win55", ["--window-team", "55"]),
    ("v7 win50 ms60", ["--min-spell", "60"]),
    ("v7 win55 ms60", ["--window-team", "55", "--min-spell", "60"]),
]


def main():
    parser = argparse.ArgumentParser(description="Sweep varian predict_possession_v7.py")
    parser.add_argument("--dry-run", action="store_true", help="Cetak perintah saja")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Jalankan hanya varian yang namanya mengandung salah satu teks ini",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Folder data opsional")
    args = parser.parse_args()

    chosen = [
        (name, extra)
        for name, extra in VARIANTS
        if not args.only or any(key.lower() in name.lower() for key in args.only)
    ]
    if not chosen:
        print("Tidak ada varian yang cocok dengan --only.")
        return 1

    for i, (name, extra) in enumerate(chosen, 1):
        cmd = [sys.executable, SCRIPT, "--team", name] + extra
        if args.data_dir:
            cmd += ["--data-dir", args.data_dir]
        print(f"[{i}/{len(chosen)}] {' '.join(repr(c) if ' ' in c else c for c in cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"      Gagal untuk varian '{name}' (kode {result.returncode}); dilanjutkan.")

    if not args.dry_run:
        print("\nSelesai. Berikutnya: python evaluation/run_evaluation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
