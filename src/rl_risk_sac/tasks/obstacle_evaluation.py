"""Episode-level stratified reports for the single Full avoidance pipeline."""
from __future__ import annotations

import numpy as np


class ObstacleEvaluation:
    def __init__(self):
        self.steps = {}
        self.episodes = []

    def record_step(self, episode, info):
        timing = {
            key: float(info.get(key, 0.)) for key in
            ("observation_to_action_ms", "obstacle_processing_ms", "action_execution_s")}
        # Step output geometry belongs to the next observation; align timing
        # with the observation that produced this action.
        timing["obstacle_processing_ms"] = float(info.get(
            "action_obstacle_processing_ms", timing["obstacle_processing_ms"]))
        self.steps.setdefault(episode, []).append(timing)

    def finish_episode(self, episode, info, success, collision):
        steps = self.steps.pop(episode, [])
        metadata = info.get("obstacle_metadata", [])
        self.episodes.append(dict(episode=int(episode), count=int(info.get("obstacle_count", 0)),
                                  obstacles=metadata, path_band=info.get("obstacle_sampling", {}).get("path_band"),
                                  success=bool(success), collision=bool(collision),
                                  duration_s=sum(s["action_execution_s"] for s in steps),
                                  timings=steps))

    def report(self):
        groups = {}
        for row in self.episodes:
            shapes = {o["shape"] for o in row["obstacles"]}
            sizes = {o["size_class"] for o in row["obstacles"]}
            keys = [f"count/{row['count']}", f"mixed_shapes/{len(shapes) > 1}"]
            if row.get("path_band"):
                keys.append(f"path_band/{row['path_band']}")
            keys += [f"shape/{name}" for name in shapes] + [f"size/{name}" for name in sizes]
            for key in keys:
                groups.setdefault(key, []).append(row)
        output = {}
        for key, rows in sorted(groups.items()):
            successful = [r["duration_s"] for r in rows if r["success"]]
            timings = [s for r in rows for s in r["timings"]]
            delays = [s["observation_to_action_ms"] for s in timings]
            output[key] = dict(episodes=len(rows), success_rate=float(np.mean([r["success"] for r in rows])),
                               collision_rate=float(np.mean([r["collision"] for r in rows])),
                               mean_completion_time_s=float(np.mean(successful)) if successful else None,
                               mean_episode_time_s=float(np.mean([r["duration_s"] for r in rows])),
                               mean_observation_to_action_ms=float(np.mean(delays)) if delays else None,
                               p95_observation_to_action_ms=float(np.percentile(delays, 95)) if delays else None,
                               mean_obstacle_processing_ms=float(np.mean([s["obstacle_processing_ms"] for s in timings])) if timings else None)
        return dict(obstacle_groups=output, obstacle_episodes=self.episodes,
                    timing_contract="wall clock from latest state acquisition through action dispatch; includes queue/learner wait; simulation paused during inference")
