import * as React from "react";

import { Costmap, Path, Vector, ZoneMarker } from "../types";
import CostmapLayer from "./CostmapLayer";
import PathLayer from "./PathLayer";
import VectorLayer from "./VectorLayer";
import { buildWorldTransform, collectDisplayPoints } from "./worldTransform";

interface VisualizerComponentProps {
  costmap: Costmap | null;
  robotPose: Vector | null;
  zoneMarkers: ZoneMarker[] | null;
  path: Path | null;
}

const VisualizerComponent: React.FC<VisualizerComponentProps> = ({
  costmap,
  robotPose,
  zoneMarkers,
  path,
}) => {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = React.useState({ width: 800, height: 600 });
  const { width, height } = dimensions;

  React.useEffect(() => {
    if (!svgRef.current?.parentElement) {
      return;
    }

    const updateDimensions = () => {
      const rect = svgRef.current?.parentElement?.getBoundingClientRect();
      if (rect) {
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    const observer = new ResizeObserver(updateDimensions);
    observer.observe(svgRef.current.parentElement);

    return () => {
      observer.disconnect();
    };
  }, []);

  const displayPoints = React.useMemo(
    () => collectDisplayPoints(robotPose, zoneMarkers, path),
    [robotPose, zoneMarkers, path],
  );
  const transform = React.useMemo(
    () => buildWorldTransform(costmap, width, height, displayPoints),
    [costmap, width, height, displayPoints],
  );
  const worldToPx = transform?.worldToPx;

  return (
    <div className="visualizer-container" style={{ width: "100%", height: "100%" }}>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        style={{
          backgroundColor: "black",
          pointerEvents: "none",
        }}
      >
        {costmap && (
          <CostmapLayer costmap={costmap} width={width} height={height} worldToPx={worldToPx} />
        )}
        {path && worldToPx && <PathLayer path={path} worldToPx={worldToPx} />}
        {zoneMarkers &&
          worldToPx &&
          zoneMarkers.map((zone) => (
            <VectorLayer
              key={zone.name}
              vector={zone.position}
              label={zone.name}
              color={zone.color}
              worldToPx={worldToPx}
            />
          ))}
        {robotPose && worldToPx && (
          <VectorLayer vector={robotPose} label="robot" worldToPx={worldToPx} />
        )}
      </svg>
    </div>
  );
};

export default React.memo(VisualizerComponent);
