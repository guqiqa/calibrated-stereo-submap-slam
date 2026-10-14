#!/usr/bin/env python3

import hashlib
from pathlib import Path
import tempfile
import unittest

from topology_scene_generator import TopologyScene, load_config
from waypoint_route_controller import WaypointPlan
from experiment_manager import resolve_scene_input


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "topology_scene_defaults.yaml"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TopologySceneTest(unittest.TestCase):
    def build(self, family="regular_baseline", seed=1000, mode="none"):
        return TopologyScene(load_config(CONFIG), family, seed, mode, "development").build()

    def test_families_validate(self):
        for family, seed in (("regular_baseline", 1000), ("perturbed_ring", 2000),
                             ("asymmetric_extension", 3000)):
            with self.subTest(family=family):
                self.assertEqual(self.build(family, seed).validation()["status"], "passed")

    def test_same_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            self.build("perturbed_ring", 2123).write(Path(left), preview=False)
            self.build("perturbed_ring", 2123).write(Path(right), preview=False)
            for name in ("topology_graph.json", "submaps.json", "mapping_route.csv",
                         "candidate_ground_truth.jsonl", "scene.world"):
                self.assertEqual(digest(Path(left) / name), digest(Path(right) / name), name)

    def test_world_exposes_entity_state_service_plugin(self):
        with tempfile.TemporaryDirectory() as output:
            self.build().write(Path(output), preview=False)
            world = (Path(output) / "scene.world").read_text(encoding="utf-8")
            self.assertIn('filename="libgazebo_ros_state.so"', world)
            self.assertIn("VisualSlam/WallLeft", world)
            self.assertIn("VisualSlam/WallRight", world)

    def test_different_seed_changes_topology(self):
        left = self.build("perturbed_ring", 2101)
        right = self.build("perturbed_ring", 2102)
        self.assertNotEqual(left.parameters["geometry"], right.parameters["geometry"])

    def test_obstacle_modes_are_separate(self):
        visual = self.build(mode="visual_only")
        self.assertTrue(visual.obstacles)
        self.assertTrue(all(item["class"] == "visual_only" for item in visual.obstacles))
        self.assertTrue(all(data["open"] for _, _, data in visual.graph.edges(data=True)))
        topology = self.build(mode="topology_change")
        self.assertTrue(any(not data["open"] for _, _, data in topology.graph.edges(data=True)))
        self.assertTrue(all(item["class"] == "topology_change" for item in topology.obstacles))

    def test_query_labels_include_overlap_and_none(self):
        scene = self.build()
        labels = [row for name, rows in scene.routes.items() if name != "mapping_route" for row in rows]
        self.assertTrue(any(len(row["legal_submaps"]) > 1 for row in labels))
        self.assertTrue(all(not row["legal_submaps"] for row in scene.routes["query_none_of_the_above"]))
        mapping = {(round(row["x"], 4), round(row["y"], 4)) for row in scene.routes["mapping_route"]}
        queries = {(round(row["x"], 4), round(row["y"], 4))
                   for name, rows in scene.routes.items()
                   if name not in {"mapping_route", "query_none_of_the_above"} for row in rows}
        self.assertFalse(mapping.intersection(queries))

    def test_waypoint_controller_turns_and_completes(self):
        plan = WaypointPlan([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.2)
        linear, angular = plan.command(0.0, 0.0, 0.0)
        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0)
        self.assertEqual(plan.command(1.0, 0.0, 0.0), (0.0, 0.0))
        self.assertTrue(plan.completed)

    def test_waypoint_path_prefers_repository_relative_input(self):
        with tempfile.TemporaryDirectory() as root:
            repository = Path(root)
            route = repository / "generated_scenes" / "query.csv"
            route.parent.mkdir(parents=True)
            route.touch()
            scene = repository / "generated_scenes" / "scene"
            scene.mkdir()
            resolved = resolve_scene_input("generated_scenes/query.csv", repository, scene)
            self.assertEqual(resolved, route.resolve())


if __name__ == "__main__":
    unittest.main()
