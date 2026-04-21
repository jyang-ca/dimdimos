import * as d3 from "d3";

import { Costmap, Path, Vector, ZoneMarker } from "../types";

export type WorldPoint = [number, number];

export interface WorldTransform {
  worldToPx: (x: number, y: number) => [number, number];
  pxToWorld: (x: number, y: number) => [number, number];
}

const AXIS_MARGIN = { left: 60, bottom: 40 };
const EXTRA_POINT_PADDING_M = 0.75;

export function collectDisplayPoints(
  robotPose: Vector | null,
  zoneMarkers: ZoneMarker[] | null,
  path: Path | null,
): WorldPoint[] {
  const points: WorldPoint[] = [];

  if (robotPose) {
    points.push([robotPose.coords[0]!, robotPose.coords[1]!]);
  }

  for (const zone of zoneMarkers ?? []) {
    points.push([zone.position.coords[0]!, zone.position.coords[1]!]);
  }

  for (const point of path?.coords ?? []) {
    points.push([point[0], point[1]]);
  }

  return points;
}

export function buildWorldTransform(
  costmap: Costmap | null,
  width: number,
  height: number,
  extraPoints: WorldPoint[],
): WorldTransform | undefined {
  const bounds = getWorldBounds(costmap, extraPoints);
  if (!bounds) {
    return undefined;
  }

  const availableWidth = Math.max(1, width - AXIS_MARGIN.left);
  const availableHeight = Math.max(1, height - AXIS_MARGIN.bottom);
  const worldWidth = Math.max(0.001, bounds.maxX - bounds.minX);
  const worldHeight = Math.max(0.001, bounds.maxY - bounds.minY);
  const scale = Math.min(availableWidth / worldWidth, availableHeight / worldHeight);
  const scaledWorldWidth = worldWidth * scale;
  const scaledWorldHeight = worldHeight * scale;
  const offsetX = AXIS_MARGIN.left + (availableWidth - scaledWorldWidth) / 2;
  const offsetY = (availableHeight - scaledWorldHeight) / 2;

  const xScale = d3
    .scaleLinear()
    .domain([bounds.minX, bounds.maxX])
    .range([offsetX, offsetX + scaledWorldWidth]);

  const yScale = d3
    .scaleLinear()
    .domain([bounds.minY, bounds.maxY])
    .range([offsetY + scaledWorldHeight, offsetY]);

  return {
    worldToPx: (x: number, y: number): [number, number] => [xScale(x), yScale(y)],
    pxToWorld: (x: number, y: number): [number, number] => [xScale.invert(x), yScale.invert(y)],
  };
}

function getWorldBounds(
  costmap: Costmap | null,
  extraPoints: WorldPoint[],
): { minX: number; maxX: number; minY: number; maxY: number } | undefined {
  let minX: number | undefined;
  let maxX: number | undefined;
  let minY: number | undefined;
  let maxY: number | undefined;

  if (costmap) {
    const rows = Math.max(1, costmap.grid.shape[0] ?? 1);
    const cols = Math.max(1, costmap.grid.shape[1] ?? 1);
    minX = costmap.origin.coords[0]!;
    minY = costmap.origin.coords[1]!;
    maxX = minX + cols * costmap.resolution;
    maxY = minY + rows * costmap.resolution;
  }

  for (const [x, y] of extraPoints) {
    minX = Math.min(minX ?? x, x - EXTRA_POINT_PADDING_M);
    maxX = Math.max(maxX ?? x, x + EXTRA_POINT_PADDING_M);
    minY = Math.min(minY ?? y, y - EXTRA_POINT_PADDING_M);
    maxY = Math.max(maxY ?? y, y + EXTRA_POINT_PADDING_M);
  }

  if (minX == undefined || maxX == undefined || minY == undefined || maxY == undefined) {
    return undefined;
  }

  return { minX, maxX, minY, maxY };
}
