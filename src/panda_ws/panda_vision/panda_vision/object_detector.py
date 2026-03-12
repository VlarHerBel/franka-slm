#!/usr/bin/env python3
import json
from collections import defaultdict

import cv2
import numpy as np

# Compatibilidad con tf_transformations/transforms3d en entornos con NumPy moderno.
if not hasattr(np, "float"):
    np.float = float

import rclpy
import tf2_ros
import tf_transformations
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")

        self.declare_parameter("target_frame", "panda_link0")
        self.declare_parameter("backend_mode", "hybrid")
        self.declare_parameter("model_path", "yolo11n-seg.pt")
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("max_sync_slop_sec", 0.15)
        self.declare_parameter("min_mask_pixels", 200)
        self.declare_parameter("min_object_height_m", 0.025)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("show_debug_window", True)
        self.declare_parameter("prefer_clusters_min_count", 2)
        self.declare_parameter("prefer_clusters_max_yolo_area_ratio", 0.45)
        self.declare_parameter("watershed_distance_ratio", 0.34)
        self.declare_parameter("table_depth_percentile", 70.0)
        self.declare_parameter("table_depth_tolerance_m", 0.035)
        self.declare_parameter("table_roi_ratio", 0.45)
        self.declare_parameter("table_mask_margin_px", 12)
        self.declare_parameter("max_table_object_area_ratio", 0.28)

        self.target_frame = self.get_parameter("target_frame").value
        self.backend_mode = self.get_parameter("backend_mode").value
        self.model_path = self.get_parameter("model_path").value
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self.max_sync_slop_sec = float(
            self.get_parameter("max_sync_slop_sec").value
        )
        self.min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)
        self.min_object_height_m = float(
            self.get_parameter("min_object_height_m").value
        )
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.prefer_clusters_min_count = int(
            self.get_parameter("prefer_clusters_min_count").value
        )
        self.prefer_clusters_max_yolo_area_ratio = float(
            self.get_parameter("prefer_clusters_max_yolo_area_ratio").value
        )
        self.watershed_distance_ratio = float(
            self.get_parameter("watershed_distance_ratio").value
        )
        self.table_depth_percentile = float(
            self.get_parameter("table_depth_percentile").value
        )
        self.table_depth_tolerance_m = float(
            self.get_parameter("table_depth_tolerance_m").value
        )
        self.table_roi_ratio = float(self.get_parameter("table_roi_ratio").value)
        self.table_mask_margin_px = int(
            self.get_parameter("table_mask_margin_px").value
        )
        self.max_table_object_area_ratio = float(
            self.get_parameter("max_table_object_area_ratio").value
        )
        self.debug_window_name = "object_detector_debug"
        self.last_logged_detection_summary = None

        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_policy,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth_image",
            self.depth_callback,
            qos_policy,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            qos_policy,
        )

        self.detections_pub = self.create_publisher(String, "/detections_3d", 10)
        self.debug_pub = self.create_publisher(Image, "/vision/debug_image", 10)

        self.camera_info = None
        self.latest_image_msg = None
        self.latest_depth_msg = None
        self.last_processed_pair = None
        self.model = None
        self.received_rgb = False
        self.received_depth = False
        self.received_camera_info = False
        self.last_status_summary = None

        self.try_load_model()
        if self.show_debug_window:
            self.render_waiting_window(
                [
                    "Object detector iniciado",
                    "Esperando /camera/image_raw",
                    "Esperando /camera/depth_image",
                    "Esperando /camera/camera_info",
                ]
            )
        self.status_timer = self.create_timer(1.5, self.status_callback)
        self.get_logger().info(
            "Object detector listo. Esperando RGB, depth y camera_info."
        )

    def try_load_model(self):
        if self.backend_mode == "clusters":
            self.get_logger().info("Usando solo backend por clústeres de profundidad.")
            return

        if not ULTRALYTICS_AVAILABLE:
            self.get_logger().warn(
                "Ultralytics no está instalado. Se usará el backend por clústeres."
            )
            return

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(
                f"Modelo Ultralytics cargado correctamente: {self.model_path}"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"No se pudo cargar el modelo '{self.model_path}': {exc}. "
                "Se usará el backend por clústeres."
            )
            self.model = None

    def camera_info_callback(self, msg):
        self.camera_info = msg
        if not self.received_camera_info:
            self.received_camera_info = True
            self.get_logger().info("Recibido primer mensaje en /camera/camera_info")

    def image_callback(self, msg):
        self.latest_image_msg = msg
        if not self.received_rgb:
            self.received_rgb = True
            self.get_logger().info("Recibido primer frame en /camera/image_raw")
        self.try_process_pair()

    def depth_callback(self, msg):
        self.latest_depth_msg = msg
        if not self.received_depth:
            self.received_depth = True
            self.get_logger().info("Recibido primer frame en /camera/depth_image")
        self.try_process_pair()

    def try_process_pair(self):
        if (
            self.camera_info is None
            or self.latest_image_msg is None
            or self.latest_depth_msg is None
        ):
            return

        image_stamp = self.to_seconds(self.latest_image_msg.header.stamp)
        depth_stamp = self.to_seconds(self.latest_depth_msg.header.stamp)
        if abs(image_stamp - depth_stamp) > self.max_sync_slop_sec:
            return

        pair_key = (
            self.latest_image_msg.header.stamp.sec,
            self.latest_image_msg.header.stamp.nanosec,
            self.latest_depth_msg.header.stamp.sec,
            self.latest_depth_msg.header.stamp.nanosec,
        )
        if pair_key == self.last_processed_pair:
            return

        self.last_processed_pair = pair_key
        self.process_scene(self.latest_image_msg, self.latest_depth_msg)

    def process_scene(self, image_msg, depth_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Error convirtiendo RGB/depth: {exc}")
            return

        depth_m = self.depth_to_meters(depth, depth_msg.encoding)
        detections, debug_frame, backend_used = self.detect_objects(bgr, depth_m)
        final_detections = self.build_3d_detections(
            detections,
            bgr,
            depth_m,
            image_msg.header.frame_id or self.camera_info.header.frame_id,
        )
        self.log_detections_if_changed(final_detections)
        debug_frame = self.annotate_debug_frame(
            debug_frame,
            backend_used,
            len(detections),
            final_detections,
        )

        payload = {
            "backend": backend_used,
            "frame_id": self.target_frame,
            "detections": final_detections,
        }

        ros_msg = String()
        ros_msg.data = json.dumps(payload)
        self.detections_pub.publish(ros_msg)

        if self.publish_debug_image:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
            debug_msg.header = image_msg.header
            self.debug_pub.publish(debug_msg)

        if self.show_debug_window:
            cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.debug_window_name, debug_frame)
            cv2.waitKey(1)

    def detect_objects(self, bgr, depth_m):
        if self.model is not None and self.backend_mode == "ultralytics":
            detections, annotated = self.run_ultralytics_segmentation(bgr)
            return detections, annotated, "ultralytics_seg"

        cluster_detections, cluster_annotated = self.run_depth_cluster_detection(bgr, depth_m)

        if self.model is not None and self.backend_mode == "hybrid":
            yolo_detections, yolo_annotated = self.run_ultralytics_segmentation(bgr)
            if self.should_prefer_clusters(yolo_detections, cluster_detections, bgr.shape):
                return cluster_detections, cluster_annotated, "depth_clusters"
            if yolo_detections:
                return yolo_detections, yolo_annotated, "ultralytics_seg"

        return cluster_detections, cluster_annotated, "depth_clusters"

    def run_ultralytics_segmentation(self, bgr):
        results = self.model.predict(
            source=bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return [], bgr.copy()

        result = results[0]
        annotated = result.plot()
        if result.masks is None or result.boxes is None:
            return [], annotated

        detections = []
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for index, class_id in enumerate(classes):
            mask = masks[index] > 0.5
            if int(mask.sum()) < self.min_mask_pixels:
                continue

            label = result.names.get(int(class_id), f"class_{class_id}")
            detections.append(
                {
                    "label": self.sanitize_label(label),
                    "confidence": float(scores[index]),
                    "bbox": boxes[index].tolist(),
                    "mask": mask,
                }
            )

        return detections, annotated

    def run_depth_cluster_detection(self, bgr, depth_m):
        debug_frame = bgr.copy()
        valid_depth = depth_m[np.isfinite(depth_m) & (depth_m > 0.05)]
        if valid_depth.size == 0:
            return [], debug_frame

        table_depth, table_mask = self.estimate_table_mask(depth_m)
        if table_mask is None:
            table_depth = float(np.percentile(valid_depth, 70))
            table_mask = np.ones_like(depth_m, dtype=bool)
        table_area = max(1, int(table_mask.sum()))

        foreground = (
            np.isfinite(depth_m)
            & (depth_m > 0.05)
            & (depth_m < table_depth - self.min_object_height_m)
            & table_mask
        ).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        detections = []
        for mask in self.split_foreground_instances(foreground, bgr):
            area = int(mask.sum())
            if area < self.min_mask_pixels:
                continue
            if area > table_area * self.max_table_object_area_ratio:
                continue

            geometry = self.describe_geometry(mask)
            if geometry is None:
                continue

            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            color_hint = self.describe_color(bgr, mask)
            shape_hint = self.describe_shape(geometry)
            label_parts = [part for part in [color_hint, shape_hint] if part]
            label_name = "_".join(label_parts) if label_parts else "object"

            detections.append(
                {
                    "label": label_name,
                    "confidence": 0.5,
                    "bbox": [x, y, x + w, y + h],
                    "mask": mask,
                    "color_hint": color_hint,
                    "geometry": geometry,
                }
            )

            cv2.polylines(
                debug_frame,
                [geometry["box_points"]],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )
            cv2.putText(
                debug_frame,
                f"{label_name} {geometry['orientation_deg']:.0f}deg",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        table_overlay = table_mask.astype(np.uint8) * 255
        table_contours, _ = cv2.findContours(
            table_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(debug_frame, table_contours, -1, (0, 200, 255), 2)

        return detections, debug_frame

    def estimate_table_mask(self, depth_m):
        height, width = depth_m.shape[:2]
        roi_ratio = max(0.2, min(0.9, self.table_roi_ratio))
        roi_h = int(height * roi_ratio)
        roi_w = int(width * roi_ratio)
        roi_y1 = max(0, (height - roi_h) // 2)
        roi_x1 = max(0, (width - roi_w) // 2)
        roi_y2 = min(height, roi_y1 + roi_h)
        roi_x2 = min(width, roi_x1 + roi_w)

        central_depth = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_central_depth = central_depth[
            np.isfinite(central_depth) & (central_depth > 0.05)
        ]
        if valid_central_depth.size == 0:
            return None, None

        table_depth = float(
            np.percentile(valid_central_depth, self.table_depth_percentile)
        )
        depth_band = (
            np.isfinite(depth_m)
            & (depth_m > 0.05)
            & (np.abs(depth_m - table_depth) < self.table_depth_tolerance_m)
        ).astype(np.uint8)
        depth_band = cv2.morphologyEx(
            depth_band, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            depth_band, connectivity=8
        )
        if num_labels <= 1:
            return table_depth, None

        roi_center_x = (roi_x1 + roi_x2) / 2.0
        roi_center_y = (roi_y1 + roi_y2) / 2.0
        best_label = None
        best_score = None

        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < max(self.min_mask_pixels * 2, 1000):
                continue

            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            distance_sq = (center_x - roi_center_x) ** 2 + (center_y - roi_center_y) ** 2
            score = distance_sq - float(area) * 0.02

            if best_score is None or score < best_score:
                best_score = score
                best_label = label_index

        if best_label is None:
            return table_depth, None

        x = int(stats[best_label, cv2.CC_STAT_LEFT])
        y = int(stats[best_label, cv2.CC_STAT_TOP])
        w = int(stats[best_label, cv2.CC_STAT_WIDTH])
        h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        margin = max(0, self.table_mask_margin_px)

        mask = np.zeros_like(depth_m, dtype=bool)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        mask[y1:y2, x1:x2] = True
        return table_depth, mask

    def split_foreground_instances(self, foreground, bgr):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground, connectivity=8
        )
        instance_masks = []

        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < self.min_mask_pixels:
                continue

            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            component_mask = (labels == label_index).astype(np.uint8)

            if area < self.min_mask_pixels * 4:
                instance_masks.append(component_mask.astype(bool))
                continue

            roi_mask = component_mask[y : y + h, x : x + w]
            roi_bgr = bgr[y : y + h, x : x + w].copy()
            split_masks = self.split_component_with_watershed(roi_mask, roi_bgr)

            if len(split_masks) <= 1:
                instance_masks.append(component_mask.astype(bool))
                continue

            for split_mask in split_masks:
                full_mask = np.zeros_like(component_mask, dtype=bool)
                full_mask[y : y + h, x : x + w] = split_mask
                if int(full_mask.sum()) >= self.min_mask_pixels:
                    instance_masks.append(full_mask)

        return instance_masks

    def split_component_with_watershed(self, component_mask, component_bgr):
        distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        if float(distance.max()) <= 0.0:
            return [component_mask.astype(bool)]

        sure_fg = (
            distance > self.watershed_distance_ratio * float(distance.max())
        ).astype(np.uint8)
        sure_fg = cv2.erode(sure_fg, np.ones((3, 3), np.uint8), iterations=1)
        if int(sure_fg.sum()) == 0:
            return [component_mask.astype(bool)]

        num_markers, markers = cv2.connectedComponents(sure_fg)
        if num_markers <= 2:
            return [component_mask.astype(bool)]

        sure_bg = cv2.dilate(component_mask, np.ones((3, 3), np.uint8), iterations=1)
        unknown = sure_bg - sure_fg

        markers = markers + 1
        markers[unknown == 1] = 0
        watershed_input = cv2.GaussianBlur(component_bgr, (3, 3), 0)
        watershed_markers = cv2.watershed(watershed_input, markers.astype(np.int32))

        split_masks = []
        for marker_id in sorted(np.unique(watershed_markers)):
            if marker_id <= 1:
                continue
            split_mask = (watershed_markers == marker_id) & (component_mask > 0)
            if int(split_mask.sum()) >= self.min_mask_pixels:
                split_masks.append(split_mask)

        return split_masks or [component_mask.astype(bool)]

    def should_prefer_clusters(self, yolo_detections, cluster_detections, image_shape):
        if len(cluster_detections) < self.prefer_clusters_min_count:
            return False
        if not yolo_detections:
            return True
        if len(yolo_detections) == 1:
            return True

        image_h, image_w = image_shape[:2]
        image_area = float(image_h * image_w)
        largest_yolo_ratio = max(
            self.bbox_area(detection["bbox"]) / image_area for detection in yolo_detections
        )
        return largest_yolo_ratio > self.prefer_clusters_max_yolo_area_ratio

    @staticmethod
    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def annotate_debug_frame(self, frame, backend_used, raw_count, final_detections):
        annotated = frame.copy()
        panel_height = max(100, 92 + 22 * min(6, len(final_detections)))
        cv2.rectangle(annotated, (8, 8), (470, panel_height), (30, 30, 30), -1)
        cv2.putText(
            annotated,
            f"backend: {backend_used}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"mask detections: {raw_count}",
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"3d detections: {len(final_detections)}",
            (18, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        for index, detection in enumerate(final_detections[:6]):
            x, y, z = detection["position"]
            cv2.putText(
                annotated,
                f"{detection['id']} {detection.get('orientation_deg', 0.0):.0f}deg: ({x:.3f}, {y:.3f}, {z:.3f})",
                (18, 102 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (255, 255, 255),
                1,
            )
        return annotated

    def render_waiting_window(self, lines):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas[:] = (35, 35, 35)
        y = 60
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
            )
            y += 42

        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.debug_window_name, canvas)
        cv2.waitKey(1)

    def status_callback(self):
        if self.camera_info is None or self.latest_image_msg is None or self.latest_depth_msg is None:
            missing = []
            if self.latest_image_msg is None:
                missing.append("/camera/image_raw")
            if self.latest_depth_msg is None:
                missing.append("/camera/depth_image")
            if self.camera_info is None:
                missing.append("/camera/camera_info")

            summary = tuple(missing)
            if summary != self.last_status_summary:
                self.get_logger().warn(
                    "Aun no han llegado todos los topicos de camara. Faltan: "
                    + ", ".join(missing)
                )
                self.last_status_summary = summary

            if self.show_debug_window:
                self.render_waiting_window(
                    ["Esperando datos de la camara"] + [f"Falta {topic}" for topic in missing]
                )
            return

        image_stamp = self.to_seconds(self.latest_image_msg.header.stamp)
        depth_stamp = self.to_seconds(self.latest_depth_msg.header.stamp)
        if abs(image_stamp - depth_stamp) > self.max_sync_slop_sec and self.show_debug_window:
            self.render_waiting_window(
                [
                    "Datos recibidos, esperando sincronizacion",
                    f"delta RGB-depth: {abs(image_stamp - depth_stamp):.3f}s",
                ]
            )

    def build_3d_detections(self, detections, bgr, depth_m, source_frame):
        if self.camera_info is None:
            return []

        transform = self.lookup_transform(source_frame)
        if transform is None:
            return []

        counts = defaultdict(int)
        final_detections = []

        for detection in detections:
            point_cam, pixel = self.compute_3d_point(detection["mask"], depth_m)
            if point_cam is None:
                continue

            point_base = self.transform_point(point_cam, transform)
            if point_base is None:
                continue

            counts[detection["label"]] += 1
            indexed_label = detection["label"]
            if sum(1 for d in detections if d["label"] == detection["label"]) > 1:
                indexed_label = f"{detection['label']}_{counts[detection['label']]}"

            final_detections.append(
                {
                    "id": indexed_label,
                    "label": detection["label"],
                    "confidence": round(float(detection["confidence"]), 4),
                    "position": [round(float(value), 4) for value in point_base],
                    "bbox": detection["bbox"],
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "orientation_deg": round(
                        float(detection.get("geometry", {}).get("orientation_deg", 0.0)), 2
                    ),
                    "source_frame": source_frame,
                    "color_hint": detection.get("color_hint"),
                }
            )

        return final_detections

    def compute_3d_point(self, mask, depth_m):
        valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0.05)
        ys, xs = np.where(valid_mask)
        if xs.size == 0:
            return None, None

        centroid_u = int(np.median(xs))
        centroid_v = int(np.median(ys))
        centroid_depth = float(np.median(depth_m[valid_mask]))

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        x = (centroid_u - cx) * centroid_depth / fx
        y = (centroid_v - cy) * centroid_depth / fy
        z = centroid_depth

        return np.array([x, y, z]), (centroid_u, centroid_v)

    def lookup_transform(self, source_frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(f"No se pudo obtener TF {source_frame}->{self.target_frame}: {exc}")
            return None

    def transform_point(self, point, transform):
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        )
        rotation = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]

        matrix = tf_transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = translation
        point_h = np.array([point[0], point[1], point[2], 1.0])
        transformed = matrix @ point_h
        return transformed[:3]

    def depth_to_meters(self, depth, encoding):
        if encoding == "16UC1":
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def describe_color(self, bgr, mask):
        if mask.sum() == 0:
            return None

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue_values = hsv[:, :, 0][mask]
        if hue_values.size == 0:
            return None

        median_hue = float(np.median(hue_values))
        if median_hue < 10 or median_hue > 170:
            return "red"
        if 20 <= median_hue <= 34:
            return "yellow"
        if 35 <= median_hue <= 85:
            return "green"
        if 86 <= median_hue <= 95:
            return "cyan"
        if 90 <= median_hue <= 140:
            return "blue"
        if 141 <= median_hue <= 169:
            return "magenta"
        return None

    def describe_geometry(self, mask):
        contour_mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if contour_area <= 1.0 or perimeter <= 1.0:
            return None

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (size_w, size_h), rect_angle = rect
        major = max(float(size_w), float(size_h))
        minor = max(1.0, min(float(size_w), float(size_h)))
        aspect_ratio = major / minor
        orientation_deg = float(rect_angle)
        if size_w < size_h:
            orientation_deg += 90.0
        if orientation_deg >= 90.0:
            orientation_deg -= 180.0

        rect_area = max(1.0, float(size_w) * float(size_h))
        fill_ratio = contour_area / rect_area
        circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        box_points = cv2.boxPoints(rect).astype(np.int32)

        return {
            "center": (float(center_x), float(center_y)),
            "major_axis_px": major,
            "minor_axis_px": minor,
            "aspect_ratio": aspect_ratio,
            "orientation_deg": orientation_deg,
            "fill_ratio": fill_ratio,
            "circularity": circularity,
            "vertex_count": len(approx),
            "box_points": box_points,
        }

    def describe_shape(self, geometry):
        if geometry is None:
            return "object"

        aspect_ratio = float(geometry["aspect_ratio"])
        fill_ratio = float(geometry["fill_ratio"])
        circularity = float(geometry["circularity"])
        vertex_count = int(geometry["vertex_count"])

        if aspect_ratio < 1.25 and circularity > 0.8 and fill_ratio < 0.9:
            return "cylinder"
        if aspect_ratio > 1.7:
            if fill_ratio < 0.83 and circularity > 0.45:
                return "cylinder"
            return "box"
        if vertex_count <= 6 and fill_ratio > 0.72:
            return "box"
        if circularity > 0.78:
            return "cylinder"
        return "box"

    def log_detections_if_changed(self, final_detections):
        summary = tuple(
            (
                detection["id"],
                tuple(round(value, 3) for value in detection["position"]),
            )
            for detection in final_detections
        )
        if summary == self.last_logged_detection_summary:
            return

        self.last_logged_detection_summary = summary
        if not final_detections:
            self.get_logger().info("Sin detecciones 3D validas en la escena.")
            return

        rendered = ", ".join(
            f"{detection['id']}@{round(float(detection.get('orientation_deg', 0.0)), 1)}deg="
            f"{tuple(round(value, 3) for value in detection['position'])}"
            for detection in final_detections
        )
        self.get_logger().info(f"Detecciones 3D: {rendered}")

    @staticmethod
    def sanitize_label(label):
        return label.strip().lower().replace(" ", "_")

    @staticmethod
    def to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_debug_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
