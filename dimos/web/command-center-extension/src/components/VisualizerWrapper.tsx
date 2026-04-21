import * as React from "react";

import { Costmap, Path, Vector, ZoneMarker } from "../types";
import VisualizerComponent from "./VisualizerComponent";
import { buildWorldTransform, collectDisplayPoints } from "./worldTransform";

export interface VisualizerData {
  costmap: Costmap | null;
  robotPose: Vector | null;
  zoneMarkers: ZoneMarker[] | null;
  path: Path | null;
}

interface VisualizerWrapperProps {
  data: VisualizerData;
  onWorldClick: (worldX: number, worldY: number) => void;
}

const VisualizerWrapper: React.FC<VisualizerWrapperProps> = ({ data, onWorldClick }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const lastClickTime = React.useRef(0);
  const clickThrottleMs = 150;

  const handleClick = React.useCallback(
    (event: React.MouseEvent) => {
      if (!data.costmap || !containerRef.current) {
        return;
      }

      event.stopPropagation();

      const now = Date.now();
      if (now - lastClickTime.current < clickThrottleMs) {
        console.log("Click throttled");
        return;
      }
      lastClickTime.current = now;

      const svgElement = containerRef.current.querySelector("svg");
      if (!svgElement) {
        return;
      }

      const svgRect = svgElement.getBoundingClientRect();
      const clickX = event.clientX - svgRect.left;
      const clickY = event.clientY - svgRect.top;

      const displayPoints = collectDisplayPoints(data.robotPose, data.zoneMarkers, data.path);
      const transform = buildWorldTransform(
        data.costmap,
        svgRect.width,
        svgRect.height,
        displayPoints,
      );
      if (!transform) {
        return;
      }

      const [worldX, worldY] = transform.pxToWorld(clickX, clickY);

      onWorldClick(worldX, worldY);
    },
    [data.costmap, data.path, data.robotPose, data.zoneMarkers, onWorldClick],
  );

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%" }} onClick={handleClick}>
      <VisualizerComponent
        costmap={data.costmap}
        robotPose={data.robotPose}
        zoneMarkers={data.zoneMarkers}
        path={data.path}
      />
    </div>
  );
};

export default VisualizerWrapper;
