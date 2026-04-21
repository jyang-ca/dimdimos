import { io, Socket } from "socket.io-client";

import {
  AppAction,
  Costmap,
  EncodedCostmap,
  EncodedPath,
  EncodedZoneMarker,
  EncodedVector,
  FullStateData,
  LatLon,
  Path,
  RobotVisualizationState,
  TwistCommand,
  Vector,
  ZoneMarker,
} from "./types";

function getSocketUrl(): string | undefined {
  const explicitUrl = new URLSearchParams(window.location.search).get("socketUrl");
  if (explicitUrl) {
    return explicitUrl;
  }

  const pathname = window.location.pathname.replace(/\/$/, "");
  if (pathname.endsWith("/command-center")) {
    return undefined;
  }

  return "ws://localhost:7779";
}

export default class Connection {
  socket: Socket;
  dispatch: React.Dispatch<AppAction>;

  constructor(dispatch: React.Dispatch<AppAction>) {
    this.dispatch = dispatch;
    const socketUrl = getSocketUrl();
    this.socket = socketUrl ? io(socketUrl) : io();

    this.socket.on("costmap", (data: EncodedCostmap) => {
      const costmap = Costmap.decode(data);
      this.dispatch({ type: "SET_COSTMAP", payload: costmap });
    });

    this.socket.on("robot_costmap", (data: { robot: string; costmap: EncodedCostmap }) => {
      const costmap = Costmap.decode(data.costmap, data.robot);
      this.dispatch({ type: "SET_NAMED_ROBOT_COSTMAP", robot: data.robot, payload: costmap });
    });

    this.socket.on("robot_pose", (data: EncodedVector) => {
      const robotPose = Vector.decode(data);
      this.dispatch({ type: "SET_ROBOT_POSE", payload: robotPose });
    });

    this.socket.on("robot_pose_named", (data: { robot: string; pose: EncodedVector }) => {
      const robotPose = Vector.decode(data.pose);
      this.dispatch({ type: "SET_NAMED_ROBOT_POSE", robot: data.robot, payload: robotPose });
    });

    this.socket.on("gps_location", (data: LatLon) => {
      this.dispatch({ type: "SET_GPS_LOCATION", payload: data });
    });

    this.socket.on("gps_travel_goal_points", (data: LatLon[]) => {
      this.dispatch({ type: "SET_GPS_TRAVEL_GOAL_POINTS", payload: data });
    });

    this.socket.on("zone_markers", (data: EncodedZoneMarker[]) => {
      const zoneMarkers = data.map((zoneMarker) => ZoneMarker.decode(zoneMarker));
      this.dispatch({ type: "SET_ZONE_MARKERS", payload: zoneMarkers });
    });

    this.socket.on("path", (data: EncodedPath) => {
      const path = Path.decode(data);
      this.dispatch({ type: "SET_PATH", payload: path });
    });

    this.socket.on("robot_path", (data: { robot: string; path: EncodedPath }) => {
      const path = Path.decode(data.path);
      this.dispatch({ type: "SET_NAMED_ROBOT_PATH", robot: data.robot, payload: path });
    });

    this.socket.on("full_state", (data: FullStateData) => {
      const state: Partial<{
        costmap: Costmap;
        robotPose: Vector;
        gpsLocation: LatLon;
        gpsTravelGoalPoints: LatLon[];
        zoneMarkers: ZoneMarker[];
        path: Path;
        robots: Record<string, RobotVisualizationState>;
      }> = {};

      if (data.costmap != undefined) {
        state.costmap = Costmap.decode(data.costmap);
      }
      if (data.robot_pose != undefined) {
        state.robotPose = Vector.decode(data.robot_pose);
      }
      if (data.gps_location != undefined) {
        state.gpsLocation = data.gps_location;
      }
      if (data.gps_travel_goal_points != undefined) {
        state.gpsTravelGoalPoints = data.gps_travel_goal_points;
      }
      if (data.zone_markers != undefined) {
        state.zoneMarkers = data.zone_markers.map((zoneMarker) => ZoneMarker.decode(zoneMarker));
      }
      if (data.path != undefined) {
        state.path = Path.decode(data.path);
      }
      if (data.robots != undefined) {
        state.robots = {};
        for (const [robot, robotData] of Object.entries(data.robots)) {
          state.robots[robot] = {
            costmap:
              robotData.costmap != undefined ? Costmap.decode(robotData.costmap, robot) : null,
            robotPose:
              robotData.robot_pose != undefined ? Vector.decode(robotData.robot_pose) : null,
            path: robotData.path != undefined ? Path.decode(robotData.path) : null,
          };
        }
      }

      this.dispatch({ type: "SET_FULL_STATE", payload: state });
    });
  }

  worldClick(worldX: number, worldY: number): void {
    this.socket.emit("click", [worldX, worldY]);
  }

  robotWorldClick(robot: string, worldX: number, worldY: number): void {
    this.socket.emit("robot_click", { robot, position: [worldX, worldY] });
  }

  startExplore(): void {
    this.socket.emit("start_explore");
  }

  stopExplore(): void {
    this.socket.emit("stop_explore");
  }

  sendMoveCommand(linear: [number, number, number], angular: [number, number, number]): void {
    const twist: TwistCommand = {
      linear: {
        x: linear[0],
        y: linear[1],
        z: linear[2],
      },
      angular: {
        x: angular[0],
        y: angular[1],
        z: angular[2],
      },
    };
    this.socket.emit("move_command", twist);
  }

  sendGpsGoal(goal: LatLon): void {
    this.socket.emit("gps_goal", goal);
  }

  stopMoveCommand(): void {
    const twist: TwistCommand = {
      linear: { x: 0, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: 0 },
    };
    this.socket.emit("move_command", twist);
  }

  disconnect(): void {
    this.socket.disconnect();
  }
}
