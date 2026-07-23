"""Merge ASCII PLY point clouds and convert CARLA's Y axis to a right-handed frame.

The output uses the standard PCD ``DATA binary`` representation and needs no
third-party Python packages.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlyInfo:
    path: Path
    point_count: int
    fields: tuple[str, ...]
    header_lines: int


def read_ply_header(path: Path) -> PlyInfo:
    """Read and validate the subset of PLY used by CARLA LiDAR exports."""
    fields: list[str] = []
    point_count: int | None = None
    current_element: str | None = None

    with path.open("r", encoding="ascii") as source:
        if source.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        format_line = source.readline().strip()
        if format_line != "format ascii 1.0":
            raise ValueError(
                f"Only ASCII PLY 1.0 is supported: {path} ({format_line!r})"
            )

        header_lines = 2
        for line in source:
            header_lines += 1
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    point_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"Unsupported vertex property in {path}: {line.strip()}")
                fields.append(parts[2])
            elif parts[0] == "end_header":
                break
        else:
            raise ValueError(f"PLY header has no end_header: {path}")

    if point_count is None or not fields:
        raise ValueError(f"PLY has no vertex data: {path}")
    if "y" not in fields:
        raise ValueError(f"PLY has no y coordinate: {path}")
    return PlyInfo(path, point_count, tuple(fields), header_lines)


def iter_points(info: PlyInfo):
    """Yield one vertex at a time as floats and verify the declared count."""
    with info.path.open("r", encoding="ascii") as source:
        for _ in range(info.header_lines):
            source.readline()

        for index in range(info.point_count):
            line = source.readline()
            if not line:
                raise ValueError(
                    f"{info.path}: expected {info.point_count} vertices, found {index}"
                )
            values = [float(value) for value in line.split()]
            if len(values) != len(info.fields):
                raise ValueError(
                    f"{info.path}:{info.header_lines + index + 1}: "
                    f"expected {len(info.fields)} values, found {len(values)}"
                )
            yield values


def write_pcd(infos: list[PlyInfo], output_path: Path, binary: bool = True) -> int:
    fields = infos[0].fields
    for info in infos[1:]:
        if info.fields != fields:
            raise ValueError(
                f"PLY fields do not match: {infos[0].path.name} has {fields}, "
                f"but {info.path.name} has {info.fields}"
            )

    total = sum(info.point_count for info in infos)
    y_index = fields.index("y")
    field_count = len(fields)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(['4'] * field_count)}\n"
        f"TYPE {' '.join(['F'] * field_count)}\n"
        f"COUNT {' '.join(['1'] * field_count)}\n"
        f"WIDTH {total}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {total}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    ).encode("ascii")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as target:
            target.write(header)
            pack_point = struct.Struct("<" + "f" * field_count).pack
            for info in infos:
                print(f"Processing {info.path.name}: {info.point_count:,} points")
                for values in iter_points(info):
                    values[y_index] = -values[y_index]
                    if binary:
                        target.write(pack_point(*values))
                    else:
                        target.write((" ".join(map(str, values)) + "\n").encode("ascii"))
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return total


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Merge CARLA ASCII PLY files into one PCD and negate Y."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=script_dir / "sensor_lidar_top",
        help="directory containing PLY files (default: sensor_lidar_top beside script)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=script_dir / "合并-右手系.pcd",
        help="output PCD path (default: 合并-右手系.pcd beside script)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="write a larger human-readable PCD instead of binary PCD",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    ply_paths = sorted(input_dir.glob("*.ply"))
    if not ply_paths:
        raise SystemExit(f"No .ply files found in: {input_dir}")

    infos = [read_ply_header(path) for path in ply_paths]
    total = write_pcd(infos, output_path, binary=not args.ascii)
    print(f"Done: {len(infos)} files, {total:,} points")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
