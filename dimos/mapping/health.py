# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Health metrics for map and costmap validation."""

from dataclasses import asdict, dataclass
import json
import time
from typing import Any

from dimos_lcm.std_msgs import String
import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2


@dataclass(frozen=True)
class GlobalMapHealthMetrics:
    """Validation metrics for a global pointcloud map."""

    voxel_count: int
    bbox_extent: tuple[float, float, float]
    timestamp_lag_ms: float | None
    finite_point_ratio: float
    frame_id: str


@dataclass(frozen=True)
class CostmapHealthMetrics:
    """Validation metrics for a global occupancy/cost map."""

    unknown_percent: float
    occupied_percent: float
    free_percent: float
    bbox_extent: tuple[float, float, float]
    timestamp_lag_ms: float | None
    width: int
    height: int
    resolution: float
    frame_id: str


def _timestamp_lag_ms(ts: float | None, now: float) -> float | None:
    if ts is None or ts <= 0:
        return None
    return (now - ts) * 1000.0


def _pointcloud_bbox_extent(points: np.ndarray[Any, Any]) -> tuple[float, float, float]:
    if points.size == 0:
        return (0.0, 0.0, 0.0)

    finite_rows = np.all(np.isfinite(points), axis=1)
    finite_points = points[finite_rows]
    if finite_points.size == 0:
        return (0.0, 0.0, 0.0)

    extents = np.ptp(finite_points, axis=0)
    return (float(extents[0]), float(extents[1]), float(extents[2]))


def compute_global_map_health(
    global_map: PointCloud2,
    *,
    now: float | None = None,
) -> GlobalMapHealthMetrics:
    """Compute health metrics for a global map pointcloud."""
    now = time.time() if now is None else now
    points, _ = global_map.as_numpy()
    voxel_count = len(global_map)

    if points.size == 0:
        finite_point_ratio = 1.0
    else:
        finite_point_ratio = float(np.mean(np.all(np.isfinite(points), axis=1)))

    return GlobalMapHealthMetrics(
        voxel_count=voxel_count,
        bbox_extent=_pointcloud_bbox_extent(points),
        timestamp_lag_ms=_timestamp_lag_ms(global_map.ts, now),
        finite_point_ratio=finite_point_ratio,
        frame_id=global_map.frame_id,
    )


def compute_costmap_health(
    costmap: OccupancyGrid,
    *,
    now: float | None = None,
) -> CostmapHealthMetrics:
    """Compute health metrics for an occupancy/cost map."""
    now = time.time() if now is None else now
    return CostmapHealthMetrics(
        unknown_percent=costmap.unknown_percent,
        occupied_percent=costmap.occupied_percent,
        free_percent=costmap.free_percent,
        bbox_extent=(
            float(costmap.width * costmap.resolution),
            float(costmap.height * costmap.resolution),
            0.0,
        ),
        timestamp_lag_ms=_timestamp_lag_ms(costmap.ts, now),
        width=costmap.width,
        height=costmap.height,
        resolution=costmap.resolution,
        frame_id=costmap.frame_id,
    )


class MappingHealthMonitor(Module):
    """Publishes JSON health metrics for global_map and global_costmap streams."""

    global_map: In[PointCloud2]
    global_costmap: In[OccupancyGrid]
    mapping_health: Out[String]

    _latest_global_map: GlobalMapHealthMetrics | None
    _latest_costmap: CostmapHealthMetrics | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_global_map = None
        self._latest_costmap = None

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.global_map.subscribe(self._on_global_map)))
        self._disposables.add(Disposable(self.global_costmap.subscribe(self._on_global_costmap)))

    def _on_global_map(self, msg: PointCloud2) -> None:
        self._latest_global_map = compute_global_map_health(msg)
        self._publish()

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        self._latest_costmap = compute_costmap_health(msg)
        self._publish()

    def _publish(self) -> None:
        payload = {
            "global_map": asdict(self._latest_global_map)
            if self._latest_global_map is not None
            else None,
            "global_costmap": asdict(self._latest_costmap)
            if self._latest_costmap is not None
            else None,
        }
        self.mapping_health.publish(String(json.dumps(payload)))


mapping_health_monitor = MappingHealthMonitor.blueprint

__all__ = [
    "CostmapHealthMetrics",
    "GlobalMapHealthMetrics",
    "MappingHealthMonitor",
    "compute_costmap_health",
    "compute_global_map_health",
    "mapping_health_monitor",
]
