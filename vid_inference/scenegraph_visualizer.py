
import numpy as np
import cv2
import xml.etree.ElementTree as ET

class SceneGraphVisualizer():
    def __init__(self, all_labels):
        self._init_node_layout(all_labels)
        
    def _init_node_layout(self, all_labels: list[str], width=400, height=480):

        center = (width // 2, height // 2)
        radius = min(width, height) // 3

        self.node_positions = {}
        n = len(all_labels)

        for i, label in enumerate(all_labels):
            angle = 2 * np.pi * i / n
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            self.node_positions[label] = (x, y)

    def render_graph_panel(
        self,
        semantic_relations,
        geometric_relations,
        width=400,
        height=480
    ):
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 30

        # -------------------------------------------------
        # 1. Determine visible nodes this frame
        # -------------------------------------------------
        visible_nodes = set()

        for sub, _, obj, _ in semantic_relations:
            visible_nodes.add(sub)
            visible_nodes.add(obj)

        for sub, obj in geometric_relations:
            visible_nodes.add(sub)
            visible_nodes.add(obj)

        if not visible_nodes:
            return canvas

        # -------------------------------------------------
        # 2. Draw geometric edges (background)
        # -------------------------------------------------
        for sub, obj in geometric_relations:
            if sub not in visible_nodes or obj not in visible_nodes:
                continue

            p1 = self.node_positions[sub]
            p2 = self.node_positions[obj]

            cv2.line(canvas, p1, p2, (80, 80, 80), 1, cv2.LINE_AA)

        # -------------------------------------------------
        # 3. Group semantic relations
        # -------------------------------------------------
        pair_map = {}

        for sub, verb, obj, score in semantic_relations:
            pair_map.setdefault((sub, obj), []).append((verb, score))

        # -------------------------------------------------
        # 4. Draw semantic edges
        # -------------------------------------------------
        for (sub, obj), rels in pair_map.items():
            if sub not in visible_nodes or obj not in visible_nodes:
                continue

            p1 = self.node_positions[sub]
            p2 = self.node_positions[obj]

            # Thickness based on strongest score
            max_score = max(s for _, s in rels)
            thickness = int(2 + 4 * max_score)

            cv2.line(canvas, p1, p2, (0, 255, 0), thickness, cv2.LINE_AA)

            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            rels = sorted(rels, key=lambda x: -x[1])

            for i, (verb, score) in enumerate(rels):
                text = f"{verb} {score:.2f}"
                cv2.putText(
                    canvas,
                    text,
                    (mid[0] + 5, mid[1] + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        # -------------------------------------------------
        # 5. Draw visible nodes LAST
        # -------------------------------------------------
        for node in visible_nodes:
            pos = self.node_positions[node]

            cv2.circle(canvas, pos, 18, (0, 180, 255), -1)
            cv2.putText(
                canvas,
                node,
                (pos[0] - 25, pos[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas

    def _edge_endpoints(self, p1, p2, radius=18):
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            return p1, p2

        ux = dx / dist
        uy = dy / dist

        x1_new = x1 + radius * ux
        y1_new = y1 + radius * uy

        x2_new = x2 - radius * ux
        y2_new = y2 - radius * uy

        return (x1_new, y1_new), (x2_new, y2_new)



    def render_graph_svg(
        self,
        semantic_relations,
        geometric_relations,
        width=400,
        height=480,
        save_path=None
    ):
        """
        Render scene graph as SVG string (optionally save to file).
        """

        svg = ET.Element(
            "svg",
            width=str(width),
            height=str(height),
            version="1.1",
            xmlns="http://www.w3.org/2000/svg"
        )

        defs = ET.SubElement(svg, "defs")

        marker = ET.SubElement(defs, "marker",
            id="arrow",
            viewBox="0 0 10 10",
            refX="9",          # ← MAGIC VALUE
            refY="5",
            markerWidth="6",
            markerHeight="6",
            orient="auto",
            markerUnits="strokeWidth"
        )

        ET.SubElement(marker, "path",
            d="M 0 0 L 10 5 L 0 10 z",
            fill="rgb(236,204,160)"
        )

        # background
        ET.SubElement(svg, "rect", x="0", y="0",
                    width=str(width), height=str(height),
                    fill="rgb(255,255,255)")

        # -----------------------------------------
        # Visible nodes
        # -----------------------------------------
        visible_nodes = set()

        for sub, _, obj, _ in semantic_relations:
            visible_nodes.add(sub)
            visible_nodes.add(obj)

        for sub, obj in geometric_relations:
            visible_nodes.add(sub)
            visible_nodes.add(obj)

        if not visible_nodes:
            svg_str = ET.tostring(svg, encoding="unicode")
            if save_path:
                with open(save_path, "w") as f:
                    f.write(svg_str)
            return svg_str

        # -----------------------------------------
        # Geometric edges (thin grey)
        # -----------------------------------------
        for sub, obj in geometric_relations:
            if sub not in visible_nodes or obj not in visible_nodes:
                continue

            x1, y1 = self.node_positions[sub]
            x2, y2 = self.node_positions[obj]
            r = 18  # node radius

            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6

            x2_adj = x2 - r * dx / dist
            y2_adj = y2 - r * dy / dist

            ET.SubElement(svg, "line",
                x1=str(x1), y1=str(y1),
                x2=str(x2), y2=str(y2),
                stroke="rgb(120,120,120)",
                **{"stroke-width": "1"}
            )

        # -----------------------------------------
        # Group semantic relations
        # -----------------------------------------
        pair_map = {}
        for sub, verb, obj, score in semantic_relations:
            pair_map.setdefault((sub, obj), []).append((verb, score))

        # -----------------------------------------
        # Semantic edges
        # -----------------------------------------
        for (sub, obj), rels in pair_map.items():
            if sub not in visible_nodes or obj not in visible_nodes:
                continue

            p1 = self.node_positions[sub]
            p2 = self.node_positions[obj]
            (x1, y1), (x2, y2) = self._edge_endpoints(p1, p2)

            max_score = max(s for _, s in rels)
            thickness = 2 + 2 * max_score

            ET.SubElement(svg, "line",
                x1=str(x1), y1=str(y1),
                x2=str(x2), y2=str(y2),
                stroke="rgb(236,204,160)",
                **{
                    "stroke-width": str(thickness),
                    "marker-end": "url(#arrow)"
                }
            )

            midx = int((x1 + x2) / 2)
            midy = int((y1 + y2) / 2)

            rels = sorted(rels, key=lambda x: -x[1])

            for i, (verb, score) in enumerate(rels):
                ET.SubElement(svg, "text",
                    x=str(midx + 5),
                    y=str(midy + i * 14),
                    fill="rgb(236,204,160)",
                    **{"font-size": "16"}
                ).text = verb



        # -----------------------------------------
        # Draw nodes
        # -----------------------------------------
        for node in visible_nodes:
            x, y = self.node_positions[node]

            ET.SubElement(svg, "circle",
                cx=str(x), cy=str(y),
                r="18",
                fill="rgb(185,233,163)"
            )

            ET.SubElement(svg, "text",
                x=str(x - 25),
                y=str(y - 25),
                fill="black",
                **{"font-size": "14"}
            ).text = node

        # -----------------------------------------
        # Output
        # -----------------------------------------
        svg_str = ET.tostring(svg, encoding="unicode")

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg_str)

        return svg_str