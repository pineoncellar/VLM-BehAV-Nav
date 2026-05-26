#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, String

try:
    from .local_types import GridInfo, Rollout, PathEval, CandidatePath, ArbiterEval
    from .callbacks import LocalCallbacksMixin
    from .rollout_library import RolloutLibraryMixin
    from .arbiter import ArbiterMixin
    from .group_selector import GroupSelectorMixin
    from .safety import SafetyMixin
    from .path_math import PathMathMixin
    from .publishers import LocalPublishersMixin
except ImportError:
    from local_types import GridInfo, Rollout, PathEval, CandidatePath, ArbiterEval
    from callbacks import LocalCallbacksMixin
    from rollout_library import RolloutLibraryMixin
    from arbiter import ArbiterMixin
    from group_selector import GroupSelectorMixin
    from safety import SafetyMixin
    from path_math import PathMathMixin
    from publishers import LocalPublishersMixin

class LocalRolloutSelector(
    Node,
    LocalCallbacksMixin,
    RolloutLibraryMixin,
    ArbiterMixin,
    GroupSelectorMixin,
    SafetyMixin,
    PathMathMixin,
    LocalPublishersMixin,
):
    def __init__(self, args):
        parameter_overrides = []
        if args.use_sim_time:
            parameter_overrides.append(Parameter("use_sim_time", Parameter.Type.BOOL, True))

        super().__init__("local_rollout_selector", parameter_overrides=parameter_overrides)
        self.args = args

        self.current_waypoint_local: Optional[Tuple[float, float]] = None
        self.last_waypoint_wall_time: float = 0.0

        # FAR route contract.
        self.far_path_xy: Optional[np.ndarray] = None
        self.last_far_path_wall_time: float = 0.0
        self.waypoint_queue_xy: Optional[np.ndarray] = None
        self.last_waypoint_queue_wall_time: float = 0.0
        self.current_subgoal_local: Optional[Tuple[float, float]] = None
        self.last_current_subgoal_wall_time: float = 0.0
        self.far_route_status: str = "UNKNOWN"

        self.goal_reached: bool = False

        self.selected_group_id: Optional[int] = None
        self.selected_rollout_id: Optional[int] = None
        self.selected_candidate_id: Optional[str] = None
        self.last_selected_rollout: Optional[Rollout] = None
        self.last_selected_path_points: Optional[np.ndarray] = None
        self.last_selected_eval: Optional[ArbiterEval] = None
        self.local_mode: str = "INIT"
        self.no_valid_plan_count: int = 0
        self.group_scores_ema = np.zeros(7, dtype=np.float32)
        self.group_scores_ema_ready: bool = False

        self.last_switch_wall_time: float = 0.0
        self.plan_count: int = 0
        self.last_plan_wall_time: float = 0.0

        self.rollouts = self.generate_rollout_library()
        self.rollouts_by_group: Dict[int, List[Rollout]] = {}
        self.representative_by_group: Dict[int, Rollout] = {}
        for r in self.rollouts:
            self.rollouts_by_group.setdefault(r.group_id, []).append(r)
            if r.is_representative:
                self.representative_by_group[r.group_id] = r

        self.footprint_offsets = self.make_footprint_offsets()

        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            args.grid_topic,
            self.grid_callback,
            10,
        )
        self.wp_sub = self.create_subscription(
            PoseStamped,
            args.current_waypoint_local_topic,
            self.waypoint_callback,
            10,
        )
        self.far_path_sub = self.create_subscription(
            Path,
            args.far_local_plan_topic,
            self.far_path_callback,
            10,
        )
        self.reference_path_sub = self.create_subscription(
            Path,
            args.reference_path_topic,
            self.reference_path_callback,
            10,
        )
        self.waypoint_queue_sub = self.create_subscription(
            PoseArray,
            args.waypoint_queue_topic,
            self.waypoint_queue_callback,
            10,
        )
        self.current_subgoal_sub = self.create_subscription(
            PoseStamped,
            args.current_subgoal_topic,
            self.current_subgoal_callback,
            10,
        )
        self.route_status_sub = self.create_subscription(
            String,
            args.route_status_topic,
            self.route_status_callback,
            10,
        )
        self.goal_reached_sub = self.create_subscription(
            Bool,
            args.far_goal_reached_topic,
            self.goal_reached_callback,
            10,
        )

        self.selected_pub = self.create_publisher(Path, args.selected_path_topic, 10)
        self.candidates_pub = self.create_publisher(Path, args.candidate_paths_topic, 10)
        self.local_ok_pub = self.create_publisher(Bool, args.local_planner_ok_topic, 10)
        self.local_status_pub = self.create_publisher(String, args.local_status_topic, 10)

        self.get_logger().info("local_rollout_selector.py started")
        self.get_logger().info(f"grid_topic       : {args.grid_topic}")
        self.get_logger().info(f"waypoint_topic   : {args.current_waypoint_local_topic}")
        self.get_logger().info(f"far_path_topic   : {args.far_local_plan_topic}")
        self.get_logger().info(f"reference_path   : {args.reference_path_topic}")
        self.get_logger().info(f"waypoint_queue   : {args.waypoint_queue_topic}")
        self.get_logger().info(f"current_subgoal  : {args.current_subgoal_topic}")
        self.get_logger().info(f"route_status     : {args.route_status_topic}")
        self.get_logger().info(f"selected_path    : {args.selected_path_topic}")
        self.get_logger().info(f"local_ok_topic   : {args.local_planner_ok_topic}")
        self.get_logger().info(f"local_status     : {args.local_status_topic}")
        self.get_logger().info(f"group_count      : 7")
        self.get_logger().info(f"candidate count  : {len(self.rollouts)}")
        self.get_logger().info("selection        : safety-first FAR route tube arbiter + rollout fallback")
        self.get_logger().info("vehicle control  : disabled")

def parse_args():
    parser = argparse.ArgumentParser(description="CMU-like group local rollout selector. No vehicle control.")

    parser.add_argument("--grid-topic", default="/local_traversability_grid")
    parser.add_argument("--current-waypoint-local-topic", default="/current_waypoint_local")
    parser.add_argument("--far-local-plan-topic", default="/far/local_plan")
    parser.add_argument("--reference-path-topic", default="/far/reference_path_local")
    parser.add_argument("--waypoint-queue-topic", default="/far/waypoint_queue_local")
    parser.add_argument("--current-subgoal-topic", default="/far/current_subgoal_local")
    parser.add_argument("--route-status-topic", default="/far/route_status")
    parser.add_argument("--selected-path-topic", default="/local_selected_trajectory")
    parser.add_argument("--candidate-paths-topic", default="/local_candidate_trajectories")
    parser.add_argument("--local-planner-ok-topic", default="/local_planner_ok")
    parser.add_argument("--local-status-topic", default="/local/status")
    parser.add_argument("--far-goal-reached-topic", default="/far/goal_reached")

    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")
    parser.add_argument("--local-frame-fallback", default="base_footprint")

    # Rollout library.
    parser.add_argument("--preview-distance", type=float, default=4.5)
    parser.add_argument("--trajectory-sample-step", type=float, default=0.30)
    parser.add_argument("--max-lateral-offset", type=float, default=2.5)
    parser.add_argument("--recovery-end-ratio", type=float, default=0.25)
    parser.add_argument("--plan-hz", type=float, default=5.0)

    # CMU-like group scoring / hysteresis.
    parser.add_argument("--min-safe-paths-per-group", type=int, default=1)
    parser.add_argument("--min-group-hold-time", type=float, default=2.5)
    parser.add_argument("--group-score-margin", type=float, default=3.0)
    parser.add_argument("--cross-center-score-margin", type=float, default=6.0)
    parser.add_argument("--emergency-group-score-margin", type=float, default=8.0)
    parser.add_argument("--representative-keep-margin", type=float, default=2.5)
    parser.add_argument("--rollout-switch-margin", type=float, default=2.5)
    parser.add_argument("--group-score-ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-hold-last-path-frames", type=int, default=3)
    parser.add_argument("--waypoint-timeout", type=float, default=1.0)
    parser.add_argument("--local-waypoint-stop-radius", type=float, default=0.6)

    # Footprint / validity.
    parser.add_argument("--centerline-stride", type=int, default=1)
    parser.add_argument("--vehicle-size", type=float, default=1.0)
    parser.add_argument("--footprint-sample-step", type=float, default=0.50)
    parser.add_argument("--footprint-check-stride", type=int, default=1)
    parser.add_argument("--allow-behind-origin", action="store_true", default=True)

    # Occupancy semantics.
    parser.add_argument("--occupied-threshold", type=int, default=50)
    parser.add_argument("--near-unknown-distance", type=float, default=0.6)
    parser.add_argument("--max-occupied-hits", type=int, default=2)
    parser.add_argument("--max-near-unknown-hits", type=int, default=50)
    parser.add_argument("--max-out-of-map-hits", type=int, default=8)
    parser.add_argument("--max-footprint-occupied-hits", type=int, default=0)
    parser.add_argument("--max-footprint-near-unknown-hits", type=int, default=120)

    # Group path score.
    parser.add_argument("--base-safe-score", type=float, default=10.0)
    parser.add_argument("--endpoint-dist-weight", type=float, default=2.2)
    parser.add_argument("--heading-error-weight", type=float, default=1.5)
    parser.add_argument("--center-penalty-weight", type=float, default=0.25)
    parser.add_argument("--footprint-penalty-weight", type=float, default=0.20)
    parser.add_argument("--shape-weight", type=float, default=0.10)
    parser.add_argument("--representative-bonus", type=float, default=1.0)
    parser.add_argument("--recovery-bonus", type=float, default=0.2)

    # Far-planner guide path adherence. This is the main local reconstruction change.
    parser.add_argument("--far-path-timeout", type=float, default=1.0)
    parser.add_argument("--far-path-weight", type=float, default=2.2)
    parser.add_argument("--far-path-endpoint-weight", type=float, default=0.6)
    parser.add_argument("--far-path-heading-weight", type=float, default=0.8)
    parser.add_argument("--far-path-heading-index", type=int, default=3)
    parser.add_argument("--far-path-rollout-stride", type=int, default=2)
    parser.add_argument("--far-path-min-point-gap", type=float, default=0.05)

    # Raw penalty weights.
    parser.add_argument("--free-weight", type=float, default=0.04)
    parser.add_argument("--occupied-weight", type=float, default=20.0)
    parser.add_argument("--unknown-near-weight", type=float, default=0.5)
    parser.add_argument("--unknown-far-weight", type=float, default=0.05)
    parser.add_argument("--out-of-map-weight", type=float, default=1.5)

    # Footprint penalties.
    parser.add_argument("--occupied-footprint-penalty", type=float, default=15.0)
    parser.add_argument("--unknown-near-footprint-penalty", type=float, default=2.5)
    parser.add_argument("--unknown-far-footprint-penalty", type=float, default=0.5)
    parser.add_argument("--out-of-map-footprint-penalty", type=float, default=2.0)

    # Arbiter: route tube + ordered waypoint queue.
    parser.add_argument("--waypoint-queue-timeout", type=float, default=1.0)
    parser.add_argument("--route-tube-radius", type=float, default=0.65)
    parser.add_argument("--route-tube-max-deviation", type=float, default=1.10)
    parser.add_argument("--min-route-progress", type=float, default=0.20)
    parser.add_argument("--ref-prepend-origin-distance", type=float, default=0.25)
    parser.add_argument("--ref-min-x", type=float, default=-0.2)
    parser.add_argument("--ref-max-x", type=float, default=4.0)
    parser.add_argument("--ref-y-limit", type=float, default=2.6)
    parser.add_argument("--ref-max-length", type=float, default=3.8)
    parser.add_argument("--ref-publish-step", type=float, default=0.25)
    parser.add_argument("--ref-smoothing-passes", type=int, default=1)
    parser.add_argument("--ref-metric-stride", type=int, default=2)
    parser.add_argument("--queue-sample-distances", type=float, nargs="+", default=[0.8, 1.5, 2.5, 3.3, 4.0])
    parser.add_argument("--queue-weights", type=float, nargs="+", default=[0.35, 0.30, 0.20, 0.10, 0.05])
    parser.add_argument("--ref-mean-weight", type=float, default=4.0)
    parser.add_argument("--ref-max-weight", type=float, default=1.8)
    parser.add_argument("--ref-endpoint-weight", type=float, default=1.2)
    parser.add_argument("--queue-weight", type=float, default=3.5)
    parser.add_argument("--progress-reward-weight", type=float, default=1.4)
    parser.add_argument("--progress-reward-cap", type=float, default=4.0)
    parser.add_argument("--clearance-weight", type=float, default=1.0)
    parser.add_argument("--unknown-cost-weight", type=float, default=0.8)
    parser.add_argument("--switch-smooth-weight", type=float, default=1.2)
    parser.add_argument("--curvature-weight", type=float, default=0.7)
    parser.add_argument("--reference-candidate-bonus", type=float, default=1.0)
    parser.add_argument("--rejoin-weight", type=float, default=2.0)
    parser.add_argument("--rollout-shape-weight", type=float, default=0.20)
    parser.add_argument("--arbiter-switch-margin", type=float, default=1.0)
    parser.add_argument("--switch-compare-step", type=float, default=0.25)

    parser.add_argument("--publish-candidates", action="store_true")
    parser.add_argument("--debug-rejections", action="store_true")
    parser.add_argument("--debug-switch", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)

    args, ros_args = parser.parse_known_args()
    return args, ros_args

def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = LocalRolloutSelector(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
