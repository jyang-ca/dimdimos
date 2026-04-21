import { EncodedOptimizedGrid, OptimizedGrid } from "./optimizedCostmap";

export type EncodedVector = Encoded<"vector"> & {
  c: number[];
};

export class Vector {
  coords: number[];
  constructor(...coords: number[]) {
    this.coords = coords;
  }

  static decode(data: EncodedVector): Vector {
    return new Vector(...data.c);
  }
}

export type EncodedZoneMarker = {
  name: string;
  position: EncodedVector;
  color: string;
};

export class ZoneMarker {
  constructor(
    public name: string,
    public position: Vector,
    public color: string,
  ) {}

  static decode(data: EncodedZoneMarker): ZoneMarker {
    return new ZoneMarker(data.name, Vector.decode(data.position), data.color);
  }
}

export interface LatLon {
  lat: number;
  lon: number;
  alt?: number;
}

export type EncodedPath = Encoded<"path"> & {
  points: Array<[number, number]>;
};

export class Path {
  constructor(public coords: Array<[number, number]>) {}

  static decode(data: EncodedPath): Path {
    return new Path(data.points);
  }
}

export type EncodedCostmap = Encoded<"costmap"> & {
  grid: EncodedOptimizedGrid;
  origin: EncodedVector;
  resolution: number;
  origin_theta: number;
};

export class Costmap {
  constructor(
    public grid: Grid,
    public origin: Vector,
    public resolution: number,
    public origin_theta: number,
  ) {
    this.grid = grid;
    this.origin = origin;
    this.resolution = resolution;
    this.origin_theta = origin_theta;
  }

  static #decoders = new Map<string, OptimizedGrid>();

  static decode(data: EncodedCostmap, channel = "default"): Costmap {
    // Use one decoder per stream so delta updates from split robot maps do not share state.
    let decoder = Costmap.#decoders.get(channel);
    if (!decoder) {
      decoder = new OptimizedGrid();
      Costmap.#decoders.set(channel, decoder);
    }

    const float32Data = decoder.decode(data.grid);
    const shape = data.grid.shape;

    // Create a Grid object from the decoded data
    const grid = new Grid(float32Data, shape);

    return new Costmap(grid, Vector.decode(data.origin), data.resolution, data.origin_theta);
  }
}

export class Grid {
  constructor(
    public data: Float32Array | Float64Array | Int32Array | Int8Array,
    public shape: number[],
  ) {}
}

export type Drawable = Costmap | Vector | Path;

export type Encoded<T extends string> = {
  type: T;
};

export interface FullStateData {
  costmap?: EncodedCostmap;
  robot_pose?: EncodedVector;
  gps_location?: LatLon;
  gps_travel_goal_points?: LatLon[];
  zone_markers?: EncodedZoneMarker[];
  path?: EncodedPath;
  robots?: Record<string, EncodedRobotVisualizationState>;
}

export interface TwistCommand {
  linear: {
    x: number;
    y: number;
    z: number;
  };
  angular: {
    x: number;
    y: number;
    z: number;
  };
}

export interface AppState {
  costmap: Costmap | null;
  robotPose: Vector | null;
  gpsLocation: LatLon | null;
  gpsTravelGoalPoints: LatLon[] | null;
  zoneMarkers: ZoneMarker[] | null;
  path: Path | null;
  robots: Record<string, RobotVisualizationState>;
}

export interface EncodedRobotVisualizationState {
  costmap?: EncodedCostmap;
  robot_pose?: EncodedVector;
  path?: EncodedPath;
}

export interface RobotVisualizationState {
  costmap: Costmap | null;
  robotPose: Vector | null;
  path: Path | null;
}

export type AppAction =
  | { type: "SET_COSTMAP"; payload: Costmap }
  | { type: "SET_ROBOT_POSE"; payload: Vector }
  | { type: "SET_GPS_LOCATION"; payload: LatLon }
  | { type: "SET_GPS_TRAVEL_GOAL_POINTS"; payload: LatLon[] }
  | { type: "SET_ZONE_MARKERS"; payload: ZoneMarker[] }
  | { type: "SET_PATH"; payload: Path }
  | { type: "SET_NAMED_ROBOT_COSTMAP"; robot: string; payload: Costmap }
  | { type: "SET_NAMED_ROBOT_POSE"; robot: string; payload: Vector }
  | { type: "SET_NAMED_ROBOT_PATH"; robot: string; payload: Path }
  | { type: "SET_FULL_STATE"; payload: Partial<AppState> };
