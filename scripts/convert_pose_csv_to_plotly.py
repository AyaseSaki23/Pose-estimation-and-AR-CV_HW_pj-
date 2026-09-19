#!/usr/bin/env python3
"""Convert pose-estimation CSV files to Plotly Chart Studio friendly XYZ tracks.

The input pose uses OpenCV object-to-camera extrinsics:

    X_cam = R * X_obj + t

For plotting the camera trajectory in the object/world coordinate system, the
camera center is:

    C_obj = -R^T * t
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def rodrigues_to_matrix(rx: float, ry: float, rz: float) -> list[list[float]]:
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    kx, ky, kz = rx / theta, ry / theta, rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c

    return [
        [c + kx * kx * one_c, kx * ky * one_c - kz * s, kx * kz * one_c + ky * s],
        [ky * kx * one_c + kz * s, c + ky * ky * one_c, ky * kz * one_c - kx * s],
        [kz * kx * one_c - ky * s, kz * ky * one_c + kx * s, c + kz * kz * one_c],
    ]


def camera_center_from_pose(row: dict[str, str]) -> tuple[float, float, float]:
    rvec = [float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])]
    tvec = [float(row["tvec_x"]), float(row["tvec_y"]), float(row["tvec_z"])]
    rotation = rodrigues_to_matrix(*rvec)

    # C = -R^T t
    cx = -(rotation[0][0] * tvec[0] + rotation[1][0] * tvec[1] + rotation[2][0] * tvec[2])
    cy = -(rotation[0][1] * tvec[0] + rotation[1][1] * tvec[1] + rotation[2][1] * tvec[2])
    cz = -(rotation[0][2] * tvec[0] + rotation[1][2] * tvec[1] + rotation[2][2] * tvec[2])
    return cx, cy, cz


def is_valid_pose(row: dict[str, str]) -> bool:
    pose_valid = row.get("pose_valid")
    if pose_valid is None:
        return True
    return pose_valid.strip() in {"1", "true", "True", "yes", "YES"}


def convert(input_path: Path, output_path: Path) -> int:
    required = {"frame_id", "rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"}
    rows_written = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")

        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as dst:
            fieldnames = [
                "frame_id",
                "timestamp_sec",
                "camera_x",
                "camera_y",
                "camera_z",
                "tvec_x",
                "tvec_y",
                "tvec_z",
                "reprojection_error",
                "top_view",
                "used_pose_projection",
            ]
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if not is_valid_pose(row):
                    continue
                cx, cy, cz = camera_center_from_pose(row)
                writer.writerow({
                    "frame_id": row.get("frame_id", ""),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                    "camera_x": f"{cx:.6f}",
                    "camera_y": f"{cy:.6f}",
                    "camera_z": f"{cz:.6f}",
                    "tvec_x": row.get("tvec_x", ""),
                    "tvec_y": row.get("tvec_y", ""),
                    "tvec_z": row.get("tvec_z", ""),
                    "reprojection_error": row.get("pose_reprojection_error", row.get("mean_reprojection_error", "")),
                    "top_view": row.get("top_view", row.get("reference_name", "")),
                    "used_pose_projection": row.get("used_pose_projection", ""),
                })
                rows_written += 1

    return rows_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/processed/gen6d_like/poses_gen6d_like.csv",
        help="Pose CSV with OpenCV rvec/tvec columns.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="data/processed/gen6d_like/camera_trajectory_plotly.csv",
        help="Output CSV for Plotly Chart Studio.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    rows_written = convert(input_path, output_path)
    print(f"Wrote {rows_written} valid camera centers to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
