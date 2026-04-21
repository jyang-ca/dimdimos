import * as React from "react";

import Button from "./Button";
import Connection from "./Connection";
import ExplorePanel from "./ExplorePanel";
import GpsButton from "./GpsButton";
import KeyboardControlPanel from "./KeyboardControlPanel";
import LeafletMap from "./components/LeafletMap";
import VisualizerWrapper, { VisualizerData } from "./components/VisualizerWrapper";
import { AppAction, AppState, LatLon, RobotVisualizationState } from "./types";

function emptyRobotState(): RobotVisualizationState {
  return { costmap: null, robotPose: null, path: null };
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_COSTMAP":
      return { ...state, costmap: action.payload };
    case "SET_ROBOT_POSE":
      return { ...state, robotPose: action.payload };
    case "SET_GPS_LOCATION":
      return { ...state, gpsLocation: action.payload };
    case "SET_GPS_TRAVEL_GOAL_POINTS":
      return { ...state, gpsTravelGoalPoints: action.payload };
    case "SET_ZONE_MARKERS":
      return { ...state, zoneMarkers: action.payload };
    case "SET_PATH":
      return { ...state, path: action.payload };
    case "SET_NAMED_ROBOT_COSTMAP": {
      const previous = state.robots[action.robot] ?? emptyRobotState();
      return {
        ...state,
        robots: {
          ...state.robots,
          [action.robot]: { ...previous, costmap: action.payload },
        },
      };
    }
    case "SET_NAMED_ROBOT_POSE": {
      const previous = state.robots[action.robot] ?? emptyRobotState();
      return {
        ...state,
        robots: {
          ...state.robots,
          [action.robot]: { ...previous, robotPose: action.payload },
        },
      };
    }
    case "SET_NAMED_ROBOT_PATH": {
      const previous = state.robots[action.robot] ?? emptyRobotState();
      return {
        ...state,
        robots: {
          ...state.robots,
          [action.robot]: { ...previous, path: action.payload },
        },
      };
    }
    case "SET_FULL_STATE":
      return { ...state, ...action.payload };
    default:
      return state;
  }
}

const initialState: AppState = {
  costmap: null,
  robotPose: null,
  gpsLocation: null,
  gpsTravelGoalPoints: null,
  zoneMarkers: null,
  path: null,
  robots: {},
};

export default function App(): React.ReactElement {
  const [state, dispatch] = React.useReducer(appReducer, initialState);
  const [isGpsMode, setIsGpsMode] = React.useState(false);
  const connectionRef = React.useRef<Connection | null>(null);

  React.useEffect(() => {
    connectionRef.current = new Connection(dispatch);

    return () => {
      if (connectionRef.current) {
        connectionRef.current.disconnect();
      }
    };
  }, []);

  const handleWorldClick = React.useCallback((worldX: number, worldY: number) => {
    connectionRef.current?.worldClick(worldX, worldY);
  }, []);

  const handleRobotWorldClick = React.useCallback(
    (robot: string, worldX: number, worldY: number) => {
      connectionRef.current?.robotWorldClick(robot, worldX, worldY);
    },
    [],
  );

  const handleStartExplore = React.useCallback(() => {
    connectionRef.current?.startExplore();
  }, []);

  const handleStopExplore = React.useCallback(() => {
    connectionRef.current?.stopExplore();
  }, []);

  const handleGpsGoal = React.useCallback((goal: LatLon) => {
    connectionRef.current?.sendGpsGoal(goal);
  }, []);

  const handleSendMoveCommand = React.useCallback(
    (linear: [number, number, number], angular: [number, number, number]) => {
      connectionRef.current?.sendMoveCommand(linear, angular);
    },
    [],
  );

  const handleStopMoveCommand = React.useCallback(() => {
    connectionRef.current?.stopMoveCommand();
  }, []);

  const handleReturnHome = React.useCallback(() => {
    connectionRef.current?.worldClick(0, 0);
  }, []);

  const handleStop = React.useCallback(() => {
    if (state.robotPose) {
      connectionRef.current?.worldClick(state.robotPose.coords[0]!, state.robotPose.coords[1]!);
    }
  }, [state.robotPose]);

  const droneState = state.robots.drone ?? emptyRobotState();
  const go2State = state.robots.go2 ?? emptyRobotState();
  const hasSplitRobotMaps = Boolean(droneState.costmap || go2State.costmap);
  const droneVisualizerData: VisualizerData = {
    costmap: droneState.costmap,
    robotPose: droneState.robotPose,
    zoneMarkers: state.zoneMarkers,
    path: droneState.path,
  };
  const go2VisualizerData: VisualizerData = {
    costmap: go2State.costmap ?? state.costmap,
    robotPose: go2State.robotPose ?? state.robotPose,
    zoneMarkers: state.zoneMarkers,
    path: go2State.path ?? state.path,
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {isGpsMode ? (
        <LeafletMap
          gpsLocation={state.gpsLocation}
          gpsTravelGoalPoints={state.gpsTravelGoalPoints}
          onGpsGoal={handleGpsGoal}
        />
      ) : (
        <>
          {hasSplitRobotMaps ? (
            <div style={{ display: "grid", gridTemplateRows: "1fr 1fr", width: "100%", height: "100%" }}>
              <MapPane
                label="Drone"
                data={droneVisualizerData}
                onWorldClick={(x, y) => handleRobotWorldClick("drone", x, y)}
              />
              <MapPane
                label="Go2"
                data={go2VisualizerData}
                onWorldClick={(x, y) => handleRobotWorldClick("go2", x, y)}
              />
            </div>
          ) : (
            <VisualizerWrapper data={state} onWorldClick={handleWorldClick} />
          )}
        </>
      )}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          display: "flex",
          width: "100%",
          padding: 5,
          gap: 5,
          alignItems: "flex-end",
        }}
      >
        <GpsButton
          onUseGps={() => {
            setIsGpsMode(true);
          }}
          onUseCostmap={() => {
            setIsGpsMode(false);
          }}
        ></GpsButton>
        <ExplorePanel onStartExplore={handleStartExplore} onStopExplore={handleStopExplore} />
        <Button onClick={handleReturnHome} isActive={false}>
          Go Home
        </Button>
        <Button onClick={handleStop} isActive={false}>
          Stop
        </Button>
        <KeyboardControlPanel
          onSendMoveCommand={handleSendMoveCommand}
          onStopMoveCommand={handleStopMoveCommand}
        />
      </div>
    </div>
  );
}

function MapPane({
  label,
  data,
  onWorldClick,
}: {
  label: string;
  data: VisualizerData;
  onWorldClick: (worldX: number, worldY: number) => void;
}): React.ReactElement {
  return (
    <div style={{ position: "relative", minHeight: 0, borderBottom: "1px solid #1f2937" }}>
      <VisualizerWrapper data={data} onWorldClick={onWorldClick} />
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 8,
          padding: "3px 7px",
          border: "1px solid #334155",
          background: "rgba(2, 6, 23, 0.78)",
          color: "#e5e7eb",
          fontSize: 12,
          lineHeight: "16px",
          pointerEvents: "none",
        }}
      >
        {label}
      </div>
    </div>
  );
}
