#!/usr/bin/env python3
"""Overlay exported stereo map points and trajectories on the Gazebo world."""

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_columns(path, names):
    values = {name: [] for name in names}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                for name in names:
                    values[name].append(float(row[name]))
            except (KeyError, TypeError, ValueError):
                continue
    return values


def parse_vector(element, count):
    if element is None or not element.text:
        return [0.0] * count
    values = [float(value) for value in element.text.split()]
    return (values + [0.0] * count)[:count]


def world_boxes(world_path):
    root = ET.parse(world_path).getroot()
    boxes = []
    for model in root.findall(".//world/model"):
        name = model.get("name", "model")
        if name == "floor":
            continue
        pose = parse_vector(model.find("pose"), 6)
        geometry = model.find("./link/collision/geometry/box/size")
        if geometry is None:
            geometry = model.find("./link/visual/geometry/box/size")
        if geometry is None:
            continue
        size = parse_vector(geometry, 3)
        boxes.append((name, pose[0], pose[1], pose[5], size[0], size[1]))
    return boxes


def add_world(ax, boxes, x_limit):
    for name, x, y, yaw, width, height in boxes:
        if x + width / 2 < x_limit[0] or x - width / 2 > x_limit[1]:
            continue
        category = "wall" if "wall" in name else "structure"
        color = "#60656f" if category == "wall" else "#d6a84b"
        alpha = 0.45 if category == "wall" else 0.65
        rectangle = Rectangle(
            (x - width / 2, y - height / 2), width, height,
            angle=math.degrees(yaw), rotation_point="center",
            facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.7,
            label="Gazebo walls" if category == "wall" else "Gazebo structures",
        )
        ax.add_patch(rectangle)


def deduplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--route-length", type=float, default=45.0)
    parser.add_argument("--point-stride", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    points = read_columns(
        run_dir / "stable_map_points.csv", ["x_gazebo", "y_gazebo", "z_gazebo"])
    visual = read_columns(
        run_dir / "trajectory_visual.csv", ["x_gazebo", "y_gazebo", "z_gazebo"])
    truth = read_columns(run_dir / "trajectory_truth.csv", ["x", "y", "z"])
    boxes = world_boxes(args.world.resolve())
    output = (args.output or run_dir / "map_alignment.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    stride = max(1, args.point_stride)
    x_limit = (-1.0, args.route_length + 12.0)
    fig, (top, side) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    add_world(top, boxes, x_limit)
    top.scatter(points["x_gazebo"][::stride], points["y_gazebo"][::stride], s=2,
                c="#168aad", alpha=0.35, linewidths=0, label="Stable visual map")
    top.plot(truth["x"], truth["y"], c="#202124", linewidth=1.8, label="Gazebo truth")
    top.plot(visual["x_gazebo"], visual["y_gazebo"], c="#d1495b", linewidth=1.2,
             label="Stereo VO")
    top.set(xlim=x_limit, ylim=(-6.0, 6.0), ylabel="Gazebo Y (m)",
            title=f"Top view: {run_dir.parent.name}/{run_dir.name}")
    top.set_aspect("equal", adjustable="box")
    top.grid(alpha=0.2)
    deduplicate_legend(top)

    side.scatter(points["x_gazebo"][::stride], points["z_gazebo"][::stride], s=2,
                 c="#168aad", alpha=0.35, linewidths=0, label="Stable visual map")
    side.plot(truth["x"], truth["z"], c="#202124", linewidth=1.8, label="Gazebo truth")
    side.plot(visual["x_gazebo"], visual["z_gazebo"], c="#d1495b", linewidth=1.2,
              label="Stereo VO")
    side.axhspan(0.0, 3.0, color="#e8e9eb", alpha=0.7, label="Scene height")
    side.set(xlim=x_limit, ylim=(-4.5, 4.5), xlabel="Gazebo X (m)",
             ylabel="Gazebo Z (m)", title="Side view and vertical outliers")
    side.grid(alpha=0.2)
    deduplicate_legend(side)

    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
