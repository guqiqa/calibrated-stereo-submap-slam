#!/usr/bin/env python3
"""Deterministic hall/double-ring/radial-corridor scene generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import networkx as nx
from PIL import Image, ImageDraw
import yaml
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import unary_union


FAMILIES = ("regular_baseline", "perturbed_ring", "asymmetric_extension")
OBSTACLE_MODES = ("none", "visual_only", "topology_change")
SCHEMA_VERSION = 1


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def square_ray_intersection(half_x: float, half_y: float, angle: float) -> tuple[float, float]:
    dx, dy = math.cos(angle), math.sin(angle)
    scale = min(half_x / max(abs(dx), 1e-9), half_y / max(abs(dy), 1e-9))
    return dx * scale, dy * scale


def perimeter_position(point: tuple[float, float], half_x: float, half_y: float) -> float:
    x, y = point
    if abs(y + half_y) < 1e-5:
        return x + half_x
    if abs(x - half_x) < 1e-5:
        return 2 * half_x + y + half_y
    if abs(y - half_y) < 1e-5:
        return 2 * half_x + 2 * half_y + half_x - x
    return 4 * half_x + 2 * half_y + half_y - y


def square_points(half_x: float, half_y: float, connector_angles: list[float]):
    points = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    points.extend(square_ray_intersection(half_x, half_y, angle) for angle in connector_angles)
    unique = {(round(x, 6), round(y, 6)) for x, y in points}
    return sorted(unique, key=lambda p: perimeter_position(p, half_x, half_y))


def sample_line(line: LineString, spacing: float, lateral: float = 0.0):
    count = max(1, int(math.ceil(line.length / spacing)))
    points = []
    for index in range(count + 1):
        distance = min(line.length, index * line.length / count)
        point = line.interpolate(distance)
        if lateral:
            before = line.interpolate(max(0.0, distance - 0.05))
            after = line.interpolate(min(line.length, distance + 0.05))
            dx, dy = after.x - before.x, after.y - before.y
            norm = math.hypot(dx, dy) or 1.0
            point = Point(point.x - dy / norm * lateral, point.y + dx / norm * lateral)
        points.append((point.x, point.y))
    return points


class TopologyScene:
    def __init__(self, config: dict, family: str, seed: int, obstacle_mode: str, split: str | None):
        if family not in FAMILIES:
            raise ValueError(f"unsupported family: {family}")
        if obstacle_mode not in OBSTACLE_MODES:
            raise ValueError(f"unsupported obstacle mode: {obstacle_mode}")
        self.config = config
        self.family = family
        self.seed = int(seed)
        self.obstacle_mode = obstacle_mode
        self.split = split
        self.random = random.Random(self.seed)
        self.graph = nx.Graph()
        self.edge_lines: dict[tuple[str, str], LineString] = {}
        self.free_space = None
        self.wall_lines: list[LineString] = []
        self.obstacles = []
        self.submaps = []
        self.routes = {}
        self.feature_points = []
        self.parameters = self._resolve_parameters()

    def _resolve_parameters(self):
        geometry = dict(self.config["geometry"])
        visual = dict(self.config["visual"])
        if self.family == "perturbed_ring":
            geometry["hall_size_m"] *= self.random.uniform(0.85, 1.15)
            geometry["inner_half_extent_m"] *= self.random.uniform(0.92, 1.08)
            geometry["outer_half_extent_m"] *= self.random.uniform(0.92, 1.08)
            geometry["inner_width_m"] *= self.random.uniform(0.90, 1.10)
            geometry["outer_width_m"] *= self.random.uniform(0.90, 1.10)
            geometry["radial_count"] = self.random.choice(geometry["radial_counts"])
            geometry["radial_width_m"] = self.random.choice(geometry["radial_widths_m"])
        elif self.family == "asymmetric_extension":
            geometry["hall_size_m"] = self.random.choice(geometry["hall_sizes_m"])
            geometry["inner_half_extent_m"] *= self.random.uniform(0.90, 1.06)
            geometry["outer_half_extent_m"] *= self.random.uniform(0.94, 1.10)
            geometry["radial_count"] = self.random.choice([6, 8])
            geometry["radial_width_m"] = self.random.choice(geometry["radial_widths_m"])
        geometry["inner_half_extent_y_m"] = geometry["inner_half_extent_m"] * (
            self.random.uniform(0.90, 1.10) if self.family != "regular_baseline" else 1.0)
        geometry["outer_half_extent_y_m"] = geometry["outer_half_extent_m"] * (
            self.random.uniform(0.88, 1.12) if self.family != "regular_baseline" else 1.0)
        visual["texture_repeat_ratio"] = min(0.95, max(0.0,
            visual["texture_repeat_ratio"] + self.random.uniform(-0.12, 0.12)))
        return {
            "geometry": geometry,
            "submaps": dict(self.config["submaps"]),
            "visual": visual,
            "obstacles": dict(self.config["obstacles"]),
            "routes": dict(self.config["routes"]),
        }

    def build(self):
        self._build_graph()
        self._build_geometry()
        self._place_visual_features()
        self._place_obstacles()
        self._build_submaps()
        self._ensure_overlap_features()
        self._build_routes()
        return self

    def _add_node(self, node_id, x, y, node_type, region):
        self.graph.add_node(node_id, x=float(x), y=float(y), node_type=node_type, region=region)

    def _add_edge(self, u, v, line, width, region, visual_group, open_edge=True):
        key = tuple(sorted((u, v)))
        self.edge_lines[key] = line
        self.graph.add_edge(
            u, v, length=float(line.length), width=float(width), region=region,
            direction="bidirectional", open=bool(open_edge), visual_group=visual_group,
            geometry=json.dumps([[round(x, 5), round(y, 5)] for x, y in line.coords]),
        )

    def _build_graph(self):
        g = self.parameters["geometry"]
        count = int(g["radial_count"])
        phase = 0.0
        if self.family == "perturbed_ring":
            phase = self.random.uniform(-math.pi / count * 0.35, math.pi / count * 0.35)
        angles = [phase + 2 * math.pi * index / count for index in range(count)]
        hall_half = g["hall_size_m"] / 2.0
        inner_x, inner_y = g["inner_half_extent_m"], g["inner_half_extent_y_m"]
        outer_x, outer_y = g["outer_half_extent_m"], g["outer_half_extent_y_m"]
        ring_specs = {
            "inner": (inner_x, inner_y, g["inner_width_m"]),
            "outer": (outer_x, outer_y, g["outer_width_m"]),
        }
        connector_lookup = {"inner": {}, "outer": {}}
        for ring, (half_x, half_y, width) in ring_specs.items():
            ring_points = square_points(half_x, half_y, angles)
            ids = []
            connector_points = [square_ray_intersection(half_x, half_y, angle) for angle in angles]
            for index, (x, y) in enumerate(ring_points):
                node_id = f"{ring}_{index:02d}"
                is_connector = any(math.hypot(x - cx, y - cy) < 1e-4 for cx, cy in connector_points)
                self._add_node(node_id, x, y, "junction" if is_connector else "ring_segment", ring)
                ids.append(node_id)
                if is_connector:
                    match = min(range(count), key=lambda i: math.hypot(x - connector_points[i][0], y - connector_points[i][1]))
                    connector_lookup[ring][match] = node_id
            for index, u in enumerate(ids):
                v = ids[(index + 1) % len(ids)]
                a = (self.graph.nodes[u]["x"], self.graph.nodes[u]["y"])
                b = (self.graph.nodes[v]["x"], self.graph.nodes[v]["y"])
                self._add_edge(u, v, LineString([a, b]), width, ring,
                               f"repeat_{index % self.parameters['visual']['repeated_group_period']}")

        hall_nodes = {}
        for index, angle in enumerate(angles):
            hx, hy = square_ray_intersection(hall_half, hall_half, angle)
            hall_id = f"hall_{index:02d}"
            self._add_node(hall_id, hx, hy, "hall", "hall")
            hall_nodes[index] = hall_id
            inner_id = connector_lookup["inner"][index]
            outer_id = connector_lookup["outer"][index]
            self._add_edge(hall_id, inner_id, LineString([
                (hx, hy), (self.graph.nodes[inner_id]["x"], self.graph.nodes[inner_id]["y"])
            ]), g["radial_width_m"], "radial_inner", f"radial_{index % 2}")
            self._add_edge(inner_id, outer_id, LineString([
                (self.graph.nodes[inner_id]["x"], self.graph.nodes[inner_id]["y"]),
                (self.graph.nodes[outer_id]["x"], self.graph.nodes[outer_id]["y"]),
            ]), g["radial_width_m"], "radial_outer", f"radial_{index % 2}")

        self._add_node("hall_center", 0.0, 0.0, "hall", "hall")
        for index, hall_id in hall_nodes.items():
            self._add_edge("hall_center", hall_id, LineString([
                (0.0, 0.0), (self.graph.nodes[hall_id]["x"], self.graph.nodes[hall_id]["y"])
            ]), g["hall_size_m"], "hall", f"hall_{index % 2}")

        if self.family == "asymmetric_extension":
            # Close one redundant radial connector and add a declared side dead end.
            radial_edges = [(u, v) for u, v, data in self.graph.edges(data=True)
                            if data["region"] == "radial_outer"]
            u, v = radial_edges[self.seed % len(radial_edges)]
            self.graph.edges[u, v]["open"] = False
            outer_nodes = [node for node, data in self.graph.nodes(data=True)
                           if data["region"] == "outer"]
            anchor = max(outer_nodes, key=lambda node: self.graph.nodes[node]["x"])
            ax, ay = self.graph.nodes[anchor]["x"], self.graph.nodes[anchor]["y"]
            dead = "dead_end_00"
            length = self.random.uniform(8.0, 13.0)
            self._add_node(dead, ax + length, ay + self.random.uniform(-2.0, 2.0), "dead_end", "side_branch")
            self._add_edge(anchor, dead, LineString([(ax, ay),
                (self.graph.nodes[dead]["x"], self.graph.nodes[dead]["y"])]),
                g["radial_width_m"], "side_branch", "unique_side")

    def open_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.graph.nodes(data=True))
        graph.add_edges_from((u, v, data) for u, v, data in self.graph.edges(data=True) if data["open"])
        return graph

    def _build_geometry(self):
        hall_size = self.parameters["geometry"]["hall_size_m"]
        parts = [box(-hall_size / 2, -hall_size / 2, hall_size / 2, hall_size / 2)]
        for u, v, data in self.graph.edges(data=True):
            if data["open"]:
                parts.append(self.edge_lines[tuple(sorted((u, v)))].buffer(
                    data["width"] / 2.0, cap_style=2, join_style=2))
        self.free_space = unary_union(parts).buffer(0)
        boundary = self.free_space.boundary
        self.wall_lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]

    def _place_visual_features(self):
        spacing = float(self.parameters["visual"]["stable_feature_spacing_m"])
        candidates = []
        for wall in self.wall_lines:
            candidates.extend(sample_line(wall, spacing))
        for index, point in enumerate(candidates):
            kind = "repeated_panel" if self.random.random() < self.parameters["visual"]["texture_repeat_ratio"] else "unique_panel"
            self.feature_points.append({"id": f"feature_{index:04d}", "x": point[0], "y": point[1],
                                        "kind": kind, "visual_group": f"repeat_{index % 4}" if kind.startswith("repeated") else f"unique_{index}"})
        # Junction pillars make submap overlap areas observable even when the
        # overlap lies in the middle of a wide crossing, away from wall panels.
        offset = len(self.feature_points)
        for index, (node, data) in enumerate(sorted(self.graph.nodes(data=True))):
            if self.open_graph().degree(node) < 3:
                continue
            self.feature_points.append({
                "id": f"feature_{offset + index:04d}", "x": data["x"], "y": data["y"],
                "kind": "junction_pillar", "visual_group": f"junction_{node}",
            })

    def _edge_midpoint_obstacle(self, u, v, index, topology_change=False):
        data = self.graph.edges[u, v]
        line = self.edge_lines[tuple(sorted((u, v)))]
        point = line.interpolate(0.5, normalized=True)
        before, after = line.interpolate(0.45, normalized=True), line.interpolate(0.55, normalized=True)
        yaw = math.atan2(after.y - before.y, after.x - before.x)
        if topology_change:
            size_x, size_y = 0.35, max(0.5, data["width"] - 0.05)
            offset = 0.0
        else:
            size_x, size_y = 0.65, 0.65
            offset = max(0.0, data["width"] / 2.0 - 0.75)
        x = point.x - math.sin(yaw) * offset
        y = point.y + math.cos(yaw) * offset
        return {"id": f"obstacle_{index:02d}", "x": x, "y": y, "yaw": yaw,
                "size_x": size_x, "size_y": size_y, "height": 1.2,
                "class": "topology_change" if topology_change else "visual_only",
                "edge": [u, v]}

    def _place_obstacles(self):
        open_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data["open"] and data["region"] != "hall"]
        self.random.shuffle(open_edges)
        if self.obstacle_mode == "visual_only":
            count = min(int(self.parameters["obstacles"]["visual_only_count"]), len(open_edges))
            self.obstacles = [self._edge_midpoint_obstacle(u, v, i) for i, (u, v) in enumerate(open_edges[:count])]
        elif self.obstacle_mode == "topology_change":
            # Prefer a ring edge so one blocked edge does not disconnect the scene.
            ring_edges = [(u, v) for u, v in open_edges if self.graph.edges[u, v]["region"] in {"inner", "outer"}]
            count = min(int(self.parameters["obstacles"]["topology_closed_edges"]), len(ring_edges))
            for index, (u, v) in enumerate(ring_edges[:count]):
                self.graph.edges[u, v]["open"] = False
                self.obstacles.append(self._edge_midpoint_obstacle(u, v, index, topology_change=True))

    def _build_submaps(self):
        length = float(self.parameters["submaps"]["length_m"])
        overlap = float(self.parameters["submaps"]["overlap_m"])
        step = length - overlap
        submap_id = 0
        for u, v, data in sorted(self.graph.edges(data=True)):
            if not data["open"]:
                continue
            line = self.edge_lines[tuple(sorted((u, v)))]
            start = 0.0
            while start < line.length - 1e-6:
                end = min(line.length, start + length)
                segment = LineString([
                    line.interpolate(start), line.interpolate(end)
                ])
                polygon = segment.buffer(data["width"] / 2.0, cap_style=2).intersection(self.free_space)
                self.submaps.append({
                    "id": f"submap_{submap_id:04d}", "kind": "corridor", "edge": [u, v],
                    "region": data["region"], "start_m": start, "end_m": end,
                    "center": [segment.centroid.x, segment.centroid.y], "polygon": polygon,
                })
                submap_id += 1
                if end >= line.length:
                    break
                start += step
        for node, data in sorted(self.graph.nodes(data=True)):
            if self.open_graph().degree(node) < 3:
                continue
            radius = max(2.0, min(3.5, self.parameters["geometry"]["radial_width_m"] * 0.6))
            polygon = Point(data["x"], data["y"]).buffer(radius).intersection(self.free_space)
            self.submaps.append({"id": f"submap_{submap_id:04d}", "kind": "junction",
                                 "node": node, "region": data["region"], "center": [data["x"], data["y"]],
                                 "polygon": polygon})
            submap_id += 1
        for item in self.submaps:
            item["neighbors"] = []
        for index, left in enumerate(self.submaps):
            for right in self.submaps[index + 1:]:
                if left["polygon"].intersection(right["polygon"]).area > 0.05:
                    left["neighbors"].append(right["id"])
                    right["neighbors"].append(left["id"])

    def _ensure_overlap_features(self):
        """Add sparse markers only where intended overlap otherwise lacks structure."""
        added = 0
        for index, left in enumerate(self.submaps):
            for right in self.submaps[index + 1:]:
                intended = (
                    (left.get("edge") is not None and left.get("edge") == right.get("edge")) or
                    left["kind"] == "junction" or right["kind"] == "junction"
                )
                if not intended:
                    continue
                overlap = left["polygon"].intersection(right["polygon"])
                if overlap.area <= 0.05 or overlap.length >= 2.0:
                    continue
                if any(overlap.buffer(1.0).covers(Point(item["x"], item["y"]))
                       for item in self.feature_points):
                    continue
                point = overlap.representative_point()
                self.feature_points.append({
                    "id": f"overlap_feature_{added:04d}", "x": point.x, "y": point.y,
                    "kind": "overlap_marker", "visual_group": f"overlap_unique_{added}",
                })
                added += 1

    def _edge_points(self, u, v, lateral=0.0):
        key = tuple(sorted((u, v)))
        line = self.edge_lines[key]
        if line.coords[0] != (self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]):
            line = LineString(list(line.coords)[::-1])
        return sample_line(line, self.parameters["routes"]["waypoint_spacing_m"], lateral)

    def _node_path_points(self, nodes, lateral=0.0):
        points = []
        for u, v in zip(nodes, nodes[1:]):
            edge_points = self._edge_points(u, v, lateral)
            points.extend(edge_points if not points else edge_points[1:])
        return points

    def _route_rows(self, name, points, query_type="mapped"):
        rows = []
        for index, (x, y) in enumerate(points):
            if index + 1 < len(points):
                nx_, ny_ = points[index + 1]
            elif index:
                nx_, ny_ = points[index - 1]
                nx_, ny_ = 2 * x - nx_, 2 * y - ny_
            else:
                nx_, ny_ = x + 1.0, y
            yaw = math.atan2(ny_ - y, nx_ - x)
            legal = [item["id"] for item in self.submaps if item["polygon"].covers(Point(x, y))]
            rows.append({"route": name, "waypoint": index, "x": x, "y": y, "yaw": yaw,
                         "query_type": query_type, "legal_submaps": legal})
        return rows

    def _build_routes(self):
        graph = self.open_graph()
        multi = nx.MultiGraph(graph)
        eulerized = nx.eulerize(multi)
        circuit = list(nx.eulerian_circuit(eulerized, source="hall_center"))
        mapping_nodes = [circuit[0][0]] + [v for _, v in circuit]
        mapping_points = self._node_path_points(mapping_nodes)
        self.routes["mapping_route"] = self._route_rows("mapping_route", mapping_points)

        outer = [node for node, data in graph.nodes(data=True) if data["region"] == "outer"]
        inner = [node for node, data in graph.nodes(data=True) if data["region"] == "inner"]
        endpoints = sorted(outer, key=lambda node: math.atan2(graph.nodes[node]["y"], graph.nodes[node]["x"]))
        query_specs = []
        for index in range(int(self.parameters["routes"]["query_count"])):
            target = endpoints[(index * max(1, len(endpoints) // 3)) % len(endpoints)]
            nodes = nx.shortest_path(graph, "hall_center", target, weight="length")
            if index % 2:
                nodes = list(reversed(nodes))
            query_specs.append((f"query_normal_{index + 1:02d}", nodes, "mapped"))
        outer_cycle = nx.cycle_basis(graph.subgraph(outer))
        if outer_cycle:
            cycle = outer_cycle[0] + [outer_cycle[0][0]]
            query_specs.append(("query_reverse_ring", list(reversed(cycle)), "mapped"))
        if inner and outer:
            cross = nx.shortest_path(graph, inner[0], outer[len(outer) // 2], weight="length")
            query_specs.append(("query_cross_hall", cross, "mapped"))
            query_specs.append(("query_cold_start", nx.shortest_path(graph, outer[-1], "hall_center", weight="length"), "cold_start"))
        lateral = float(self.parameters["routes"]["lateral_offset_m"])
        for index, (name, nodes, query_type) in enumerate(query_specs):
            self.routes[name] = self._route_rows(name, self._node_path_points(nodes, lateral if index % 2 == 0 else -lateral), query_type)

        max_x = self.free_space.bounds[2]
        none_points = [(max_x + 2.0 + i * 0.5, self.free_space.centroid.y) for i in range(12)]
        self.routes["query_none_of_the_above"] = self._route_rows(
            "query_none_of_the_above", none_points, "none_of_the_above")

    def validation(self):
        open_graph = self.open_graph()
        cycles = nx.cycle_basis(open_graph)
        dead_ends = [node for node in open_graph if open_graph.degree(node) == 1]
        declared_dead_ends = [node for node, data in open_graph.nodes(data=True) if data["node_type"] == "dead_end"]
        topology_change_endpoints = {
            node for u, v, data in self.graph.edges(data=True) if not data["open"]
            for node in (u, v)
        } if self.obstacle_mode == "topology_change" else set()
        allowed_dead_ends = set(declared_dead_ends) | topology_change_endpoints
        required_cycles = 2 if self.family in {"regular_baseline", "perturbed_ring"} else 1
        overlap_pairs, weak_overlaps = 0, []
        visual_groups = {}
        for u, v, data in self.graph.edges(data=True):
            visual_groups.setdefault(data["visual_group"], []).append({u, v})
        visual_conflict = any(
            left.isdisjoint(right)
            for edges in visual_groups.values()
            for index, left in enumerate(edges)
            for right in edges[index + 1:]
        )
        mapping_positions = {(round(row["x"], 4), round(row["y"], 4))
                             for row in self.routes["mapping_route"]}
        query_positions = {(round(row["x"], 4), round(row["y"], 4))
                           for name, rows in self.routes.items()
                           if name not in {"mapping_route", "query_none_of_the_above"}
                           for row in rows}
        for index, left in enumerate(self.submaps):
            for right in self.submaps[index + 1:]:
                overlap = left["polygon"].intersection(right["polygon"])
                if overlap.area <= 0.05:
                    continue
                overlap_pairs += 1
                intended_overlap = (
                    (left.get("edge") is not None and left.get("edge") == right.get("edge")) or
                    left["kind"] == "junction" or right["kind"] == "junction"
                )
                if not intended_overlap:
                    continue
                nearby = sum(overlap.buffer(1.0).covers(Point(item["x"], item["y"])) for item in self.feature_points)
                if nearby < 1 and overlap.length < 2.0:
                    weak_overlaps.append([left["id"], right["id"]])
        checks = {
            "geometry_valid": bool(self.free_space.is_valid and not self.free_space.is_empty),
            "open_graph_connected": nx.is_connected(open_graph),
            "expected_cycle_count": len(cycles) >= required_cycles,
            "radial_count_matches": sum(1 for _, _, data in self.graph.edges(data=True)
                                          if data["region"] == "radial_inner") == self.parameters["geometry"]["radial_count"],
            "only_declared_dead_ends": set(dead_ends).issubset(allowed_dead_ends),
            "submaps_generated": len(self.submaps) > 0,
            "overlap_regions_generated": overlap_pairs > 0,
            "overlap_structure_proxy": not weak_overlaps,
            "visual_topology_conflict_pair": visual_conflict,
            "mapping_query_pose_separation": not mapping_positions.intersection(query_positions),
            "none_of_the_above_available": all(not row["legal_submaps"]
                                                for row in self.routes["query_none_of_the_above"]),
            "visual_only_does_not_close_edges": (
                self.obstacle_mode != "visual_only" or
                all(item["class"] == "visual_only" for item in self.obstacles)
            ),
            "topology_change_declares_closed_edges": (
                self.obstacle_mode != "topology_change" or
                any(not data["open"] for _, _, data in self.graph.edges(data=True))
            ),
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "cycle_count": len(cycles), "dead_ends": dead_ends,
            "declared_dead_ends": declared_dead_ends,
            "topology_change_endpoints": sorted(topology_change_endpoints),
            "submap_count": len(self.submaps), "overlap_pairs": overlap_pairs,
            "weak_overlap_pairs": weak_overlaps,
            "bounds": list(self.free_space.bounds),
        }

    def write(self, output: Path, export_rmf=True, export_gazebo=True, preview=True):
        output.mkdir(parents=True, exist_ok=True)
        (output / "query_routes").mkdir(exist_ok=True)
        resolved = {
            "schema_version": SCHEMA_VERSION, "family": self.family, "seed": self.seed,
            "split": self.split, "obstacle_mode": self.obstacle_mode,
            **self.parameters,
        }
        (output / "scenario_resolved.yaml").write_text(
            yaml.safe_dump(json_ready(resolved), sort_keys=False), encoding="utf-8")
        self._write_topology(output)
        self._write_submaps(output)
        self._write_routes(output)
        self._write_obstacles(output)
        self._write_occupancy(output)
        if export_rmf:
            self._write_rmf(output)
        if export_gazebo:
            self._write_world(output)
        if preview:
            self._write_previews(output)
        validation = self.validation()
        (output / "validation_report.json").write_text(
            json.dumps(json_ready(validation), indent=2), encoding="utf-8")
        self._write_licenses(output)
        self._write_manifest(output, validation)
        return validation

    def _write_topology(self, output):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "nodes": [{"id": node, **json_ready(data)} for node, data in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **json_ready(data)} for u, v, data in self.graph.edges(data=True)],
            "map_version": f"{self.family}-{self.seed}-{self.obstacle_mode}",
        }
        (output / "topology_graph.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        graphml = nx.Graph()
        for node, data in self.graph.nodes(data=True):
            graphml.add_node(node, **{key: str(value) for key, value in data.items()})
        for u, v, data in self.graph.edges(data=True):
            graphml.add_edge(u, v, **{key: str(value) for key, value in data.items()})
        nx.write_graphml(graphml, output / "navigation_graph.graphml")

    def _write_submaps(self, output):
        features = []
        records = []
        for item in self.submaps:
            record = {key: json_ready(value) for key, value in item.items() if key != "polygon"}
            record["geometry"] = mapping(item["polygon"])
            records.append(record)
            features.append({"type": "Feature", "properties": {"id": item["id"], "kind": item["kind"],
                                                                  "legal_region": True},
                             "geometry": mapping(item["polygon"])})
        (output / "submaps.json").write_text(json.dumps({"submaps": records}, indent=2), encoding="utf-8")
        (output / "legal_submap_regions.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")

    @staticmethod
    def _write_route(path, rows):
        fields = ["route", "waypoint", "x", "y", "yaw", "query_type", "legal_submaps"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                record = dict(row)
                record["legal_submaps"] = json.dumps(record["legal_submaps"])
                writer.writerow(record)

    def _write_routes(self, output):
        self._write_route(output / "mapping_route.csv", self.routes["mapping_route"])
        labels = []
        for name, rows in sorted(self.routes.items()):
            if name == "mapping_route":
                continue
            self._write_route(output / "query_routes" / f"{name}.csv", rows)
            for row in rows:
                labels.append({
                    "route": name, "waypoint": row["waypoint"], "x": round(row["x"], 6),
                    "y": round(row["y"], 6), "legal_submaps": row["legal_submaps"],
                    "coverage_label_available": True,
                    "none_of_the_above": row["query_type"] == "none_of_the_above",
                    "candidate_coverage_miss": None,
                    "geometry_verified": None, "topology_prior_correct": None,
                    "map_version": f"{self.family}-{self.seed}-{self.obstacle_mode}",
                })
        with (output / "candidate_ground_truth.jsonl").open("w", encoding="utf-8") as handle:
            for row in labels:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_obstacles(self, output):
        payload = {"mode": self.obstacle_mode, "map_version": f"{self.family}-{self.seed}-{self.obstacle_mode}",
                   "obstacles": json_ready(self.obstacles),
                   "closed_edges": [[u, v] for u, v, data in self.graph.edges(data=True) if not data["open"]]}
        (output / "obstacles.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_occupancy(self, output):
        resolution = 0.1
        min_x, min_y, max_x, max_y = self.free_space.bounds
        margin = 3.0
        min_x, min_y, max_x, max_y = min_x - margin, min_y - margin, max_x + margin, max_y + margin
        width, height = int(math.ceil((max_x - min_x) / resolution)), int(math.ceil((max_y - min_y) / resolution))
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        def pixels(poly):
            return [((x - min_x) / resolution, height - (y - min_y) / resolution) for x, y in poly.exterior.coords]
        polygons = list(self.free_space.geoms) if hasattr(self.free_space, "geoms") else [self.free_space]
        for polygon in polygons:
            draw.polygon(pixels(polygon), fill=254)
            for interior in polygon.interiors:
                draw.polygon([((x - min_x) / resolution, height - (y - min_y) / resolution)
                              for x, y in interior.coords], fill=0)
        for obstacle in self.obstacles:
            footprint = box(-obstacle["size_x"] / 2, -obstacle["size_y"] / 2,
                            obstacle["size_x"] / 2, obstacle["size_y"] / 2)
            from shapely.affinity import rotate, translate
            footprint = translate(rotate(footprint, math.degrees(obstacle["yaw"])), obstacle["x"], obstacle["y"])
            draw.polygon(pixels(footprint), fill=0)
        image.save(output / "occupancy.pgm")
        occupancy = {"image": "occupancy.pgm", "resolution": resolution, "origin": [min_x, min_y, 0.0],
                     "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}
        (output / "occupancy.yaml").write_text(yaml.safe_dump(occupancy, sort_keys=False), encoding="utf-8")

    def _write_rmf(self, output):
        vertices, vertex_index = [], {}
        def add_vertex(x, y, name=""):
            key = (round(x, 5), round(y, 5))
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append([key[0], key[1], 0.0, name, {"is_charger": [1, False]}])
            return vertex_index[key]
        lanes = []
        for u, v, data in self.graph.edges(data=True):
            a = self.graph.nodes[u]; b = self.graph.nodes[v]
            ai, bi = add_vertex(a["x"], a["y"], u), add_vertex(b["x"], b["y"], v)
            lanes.append([ai, bi, {"bidirectional": [3, True], "graph_idx": [2, 0],
                                   "speed_limit": [2, 0.5], "is_open": [1, bool(data["open"])]}])
        wall_vertices, walls = [], []
        for wall in self.wall_lines:
            previous = None
            for x, y in wall.coords:
                wall_vertices.append([round(x, 5), round(y, 5), 0.0, "", {}])
                current = len(wall_vertices) - 1
                if previous is not None:
                    walls.append([previous, current, {"texture_name": [1, "default"]}])
                previous = current
        building = {
            "building_name": f"{self.family}_{self.seed}", "coordinate_system": "reference_image",
            "reference_level_name": "L1", "levels": {"L1": {"elevation": 0.0,
                "vertices": vertices + wall_vertices,
                "lanes": lanes,
                "walls": [[a + len(vertices), b + len(vertices), opts] for a, b, opts in walls],
                "doors": [], "floors": [], "models": []}},
        }
        (output / "building.yaml").write_text(yaml.safe_dump(building, sort_keys=False), encoding="utf-8")
        rmf_tool = shutil.which("building_map_generator")
        compatibility = {"available": bool(rmf_tool), "tool": rmf_tool,
                         "status": "not_run" if rmf_tool else "unavailable_use_classic_sdf_backend"}
        (output / "rmf_compatibility.json").write_text(json.dumps(compatibility, indent=2), encoding="utf-8")

    @staticmethod
    def _box_model(world, name, x, y, z, sx, sy, sz, yaw=0.0, color="0.45 0.47 0.50 1", collision=True):
        model = ET.SubElement(world, "model", {"name": name})
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = f"{x:.5f} {y:.5f} {z:.5f} 0 0 {yaw:.6f}"
        link = ET.SubElement(model, "link", {"name": "link"})
        if collision:
            coll = ET.SubElement(link, "collision", {"name": "collision"})
            geom = ET.SubElement(coll, "geometry"); shape = ET.SubElement(geom, "box")
            ET.SubElement(shape, "size").text = f"{sx:.5f} {sy:.5f} {sz:.5f}"
        visual = ET.SubElement(link, "visual", {"name": "visual"})
        geom = ET.SubElement(visual, "geometry"); shape = ET.SubElement(geom, "box")
        ET.SubElement(shape, "size").text = f"{sx:.5f} {sy:.5f} {sz:.5f}"
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = color
        ET.SubElement(material, "diffuse").text = color

    @staticmethod
    def _box_link(model, name, x, y, z, sx, sy, sz, yaw=0.0,
                  color="0.45 0.47 0.50 1", collision=True, material_name=None):
        link = model.find("link")
        if link is None:
            link = ET.SubElement(model, "link", {"name": "environment_link"})
        pose = f"{x:.5f} {y:.5f} {z:.5f} 0 0 {yaw:.6f}"
        if collision:
            coll = ET.SubElement(link, "collision", {"name": f"{name}_collision"})
            ET.SubElement(coll, "pose").text = pose
            geom = ET.SubElement(coll, "geometry"); shape = ET.SubElement(geom, "box")
            ET.SubElement(shape, "size").text = f"{sx:.5f} {sy:.5f} {sz:.5f}"
        visual = ET.SubElement(link, "visual", {"name": f"{name}_visual"})
        ET.SubElement(visual, "pose").text = pose
        geom = ET.SubElement(visual, "geometry"); shape = ET.SubElement(geom, "box")
        ET.SubElement(shape, "size").text = f"{sx:.5f} {sy:.5f} {sz:.5f}"
        material = ET.SubElement(visual, "material")
        if material_name:
            script = ET.SubElement(material, "script")
            ET.SubElement(script, "uri").text = (
                "model://visual_slam_gazebo/materials/scripts/visual_slam.material")
            ET.SubElement(script, "name").text = material_name
        else:
            ET.SubElement(material, "ambient").text = color
            ET.SubElement(material, "diffuse").text = color

    def _write_world(self, output):
        sdf = ET.Element("sdf", {"version": "1.6"})
        world = ET.SubElement(sdf, "world", {"name": f"{self.family}_{self.seed}"})
        ET.SubElement(world, "gravity").text = "0 0 -9.81"
        physics = ET.SubElement(world, "physics", {"type": "ode"})
        ET.SubElement(physics, "max_step_size").text = "0.001"
        ET.SubElement(physics, "real_time_update_rate").text = "1000"
        scene = ET.SubElement(world, "scene")
        ET.SubElement(scene, "ambient").text = "0.38 0.38 0.38 1"
        ET.SubElement(scene, "background").text = "0.08 0.10 0.14 1"
        state_plugin = ET.SubElement(
            world, "plugin", {"name": "gazebo_ros_state", "filename": "libgazebo_ros_state.so"})
        ET.SubElement(state_plugin, "update_rate").text = "20.0"
        include = ET.SubElement(world, "include"); ET.SubElement(include, "uri").text = "model://sun"
        robot = ET.SubElement(world, "include"); ET.SubElement(robot, "uri").text = "model://stereo_robot"
        ET.SubElement(robot, "pose").text = "0 0 0 0 0 0"
        environment = ET.SubElement(world, "model", {"name": "generated_static_environment"})
        ET.SubElement(environment, "static").text = "true"
        min_x, min_y, max_x, max_y = self.free_space.bounds
        self._box_link(environment, "floor", (min_x + max_x) / 2, (min_y + max_y) / 2, -0.05,
                       max_x - min_x + 6, max_y - min_y + 6, 0.1, color="0.30 0.32 0.34 1")
        wall_height = self.parameters["geometry"]["wall_height_m"]
        wall_thickness = self.parameters["geometry"]["wall_thickness_m"]
        wall_index = 0
        for boundary in self.wall_lines:
            coords = list(boundary.coords)
            for a, b in zip(coords, coords[1:]):
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                if length < 0.03:
                    continue
                yaw = math.atan2(b[1] - a[1], b[0] - a[0])
                self._box_link(environment, f"wall_{wall_index:04d}", (a[0] + b[0]) / 2,
                               (a[1] + b[1]) / 2, wall_height / 2,
                               length, wall_thickness, wall_height, yaw,
                               material_name=("VisualSlam/WallLeft" if wall_index % 2 == 0
                                              else "VisualSlam/WallRight"))
                wall_index += 1
        colors = ["0.72 0.20 0.16 1", "0.10 0.48 0.62 1", "0.75 0.61 0.12 1", "0.20 0.58 0.30 1"]
        for index, feature in enumerate(self.feature_points):
            color = colors[index % len(colors)] if feature["kind"] == "repeated_panel" else colors[(index * 3 + 1) % len(colors)]
            self._box_link(environment, feature["id"], feature["x"], feature["y"], 1.3,
                           0.28, 0.28, 2.4, color=color, collision=False)
        for obstacle in self.obstacles:
            self._box_link(environment, obstacle["id"], obstacle["x"], obstacle["y"], obstacle["height"] / 2,
                           obstacle["size_x"], obstacle["size_y"], obstacle["height"], obstacle["yaw"],
                           color="0.62 0.22 0.12 1")
        ET.indent(sdf, space="  ")
        ET.ElementTree(sdf).write(output / "scene.world", encoding="utf-8", xml_declaration=True)

    def _write_previews(self, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, semantics in (("preview_topology.png", False), ("preview_semantics.png", True)):
            fig, ax = plt.subplots(figsize=(9, 9))
            polygons = list(self.free_space.geoms) if hasattr(self.free_space, "geoms") else [self.free_space]
            for polygon in polygons:
                x, y = polygon.exterior.xy
                ax.fill(x, y, color="#f3f4f2", edgecolor="#303438", linewidth=1.4)
            if semantics:
                colors = {"inner": "#2a6f97", "outer": "#d1495b", "hall": "#2f855a",
                          "radial_inner": "#d69e2e", "radial_outer": "#d69e2e", "side_branch": "#805ad5"}
                for u, v, data in self.graph.edges(data=True):
                    line = self.edge_lines[tuple(sorted((u, v)))]
                    x, y = line.xy
                    ax.plot(x, y, color=colors.get(data["region"], "#555"), linewidth=2.2,
                            linestyle="-" if data["open"] else "--")
                for item in self.submaps:
                    cx, cy = item["center"]
                    ax.text(cx, cy, item["id"].split("_")[-1], fontsize=5, color="#222")
                for obstacle in self.obstacles:
                    ax.scatter([obstacle["x"]], [obstacle["y"]], marker="s", c="#111", s=24)
            else:
                pos = {node: (data["x"], data["y"]) for node, data in self.graph.nodes(data=True)}
                nx.draw_networkx(self.open_graph(), pos=pos, ax=ax, node_size=18, with_labels=False,
                                 width=1.2, edge_color="#2a6f97", node_color="#d1495b")
            ax.set_aspect("equal"); ax.set_xlabel("Gazebo X (m)"); ax.set_ylabel("Gazebo Y (m)")
            ax.set_title(f"{self.family} / seed {self.seed} / {self.obstacle_mode}")
            fig.tight_layout(); fig.savefig(output / name, dpi=160); plt.close(fig)

    def _write_licenses(self, output):
        payload = {
            "runtime_dependencies": [
                {"name": "Open-RMF building map format/tools", "license": "Apache-2.0",
                 "version": "1.6.3", "required": False},
                {"name": "NetworkX", "license": "BSD-3-Clause", "version": "3.4.2", "required": True},
                {"name": "Shapely", "license": "BSD-3-Clause", "version": "1.8.5.post1", "required": True},
                {"name": "Jinja2", "license": "BSD-3-Clause", "version": "3.0.3", "required": False},
                {"name": "PyYAML", "license": "MIT", "version": "5.4.1", "required": True},
                {"name": "Matplotlib", "license": "PSF-compatible", "version": "3.10.9", "required": True},
                {"name": "Pillow", "license": "HPND", "version": "9.0.1", "required": True},
            ],
            "reference_only": ["pcg_gazebo", "arena-tools", "Arena-Rosnav-3D"],
        }
        (output / "licenses.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_manifest(self, output, validation):
        root = Path(__file__).resolve().parents[3]
        generator_sources = [
            "src/visual_slam_gazebo/scripts/topology_scene_generator.py",
            "src/visual_slam_gazebo/scripts/generate_topology_scene.py",
            "config/topology_scene_defaults.yaml",
            "requirements-topology-scene.txt",
        ]
        files = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "scene_manifest.json":
                files[str(path.relative_to(output)).replace(os.sep, "/")] = {
                    "size": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "schema_version": SCHEMA_VERSION, "scene_id": f"{self.family}_{self.seed}_{self.obstacle_mode}",
            "family": self.family, "seed": self.seed, "split": self.split,
            "obstacle_mode": self.obstacle_mode, "map_version": f"{self.family}-{self.seed}-{self.obstacle_mode}",
            "generator_git_commit": git_value(root, "rev-parse", "HEAD"),
            "generator_git_dirty": bool(git_value(
                root, "status", "--porcelain", "--", *generator_sources)),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "coordinate_frame": "gazebo_world", "validation_status": validation["status"],
            "parameters": json_ready(self.parameters), "files": files,
        }
        (output / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def generate_scene(config_path, family, seed, obstacle_mode, split, output,
                   export_rmf=True, export_gazebo=True, preview=True):
    scene = TopologyScene(load_config(Path(config_path)), family, seed, obstacle_mode, split).build()
    validation = scene.write(Path(output), export_rmf=export_rmf,
                             export_gazebo=export_gazebo, preview=preview)
    if validation["status"] != "passed":
        raise RuntimeError(f"scene validation failed: {validation['checks']}")
    return validation

