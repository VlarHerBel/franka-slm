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
        self.declare_parameter("camera_optical_frame", "camera_link_optical")
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
        self.declare_parameter("max_table_object_area_ratio", 0.15)
        self.declare_parameter("max_object_dimension_m", 0.25)
        # Merge "ghost duplicates" produced by splitting the same physical object.
        # Criteria: same label and close XY in the base frame.
        self.declare_parameter("merge_duplicate_xy_dist_m", 0.06)

        self.target_frame = self.get_parameter("target_frame").value
        self.backend_mode = self.get_parameter("backend_mode").value
        self.model_path = self.get_parameter("model_path").value
        self.camera_optical_frame = self.get_parameter("camera_optical_frame").value
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
        self.max_object_dimension_m = float(
            self.get_parameter("max_object_dimension_m").value
        )
        self.merge_duplicate_xy_dist_m = float(
            self.get_parameter("merge_duplicate_xy_dist_m").value
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
            self.get_logger().info(
                f"Recibido primer mensaje en /camera/camera_info: "
                f"width={msg.width}, height={msg.height}, "
                f"K=[{msg.k[0]:.1f}, {msg.k[2]:.1f}, {msg.k[4]:.1f}, {msg.k[5]:.1f}]"
            )

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
        if detections and not hasattr(self, "_logged_raw_det"):
            self._logged_raw_det = True
            self.get_logger().info(
                f"Detecciones crudas ({backend_used}): {len(detections)} objetos. "
                f"RGB shape={bgr.shape}, Depth shape={depth_m.shape}"
            )
            for d in detections:
                self.get_logger().info(
                    f"  -> label={d['label']}, mask_shape={d['mask'].shape}, "
                    f"mask_sum={int(d['mask'].sum())}, bbox={d['bbox']}"
                )
        final_detections = self.build_3d_detections(
            detections,
            bgr,
            depth_m,
            image_msg.header.frame_id,
            self.camera_info.header.frame_id if self.camera_info is not None else "",
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
            if cluster_detections and self.should_prefer_clusters(
                yolo_detections, cluster_detections, bgr.shape
            ):
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
        img_h, img_w = bgr.shape[:2]

        for index, class_id in enumerate(classes):
            raw_mask = masks[index] > 0.5
            if raw_mask.shape[0] != img_h or raw_mask.shape[1] != img_w:
                raw_mask = cv2.resize(
                    raw_mask.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask = raw_mask
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
            table_depth = float(np.percentile(valid_depth, 85))
            table_mask = np.ones_like(depth_m, dtype=bool)
        table_area = max(1, int(table_mask.sum()))

        height_threshold = self.min_object_height_m
        foreground = (
            np.isfinite(depth_m)
            & (depth_m > 0.05)
            & (depth_m < table_depth - height_threshold)
            & table_mask
        ).astype(np.uint8)

        fg_before_morph = int(foreground.sum())

        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_open)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_close)

        fg_after_morph = int(foreground.sum())

        if not hasattr(self, "_logged_cluster_diag"):
            self._logged_cluster_diag = True
            self.get_logger().info(
                f"Depth clusters: table_depth={table_depth:.4f}m, "
                f"threshold={table_depth - height_threshold:.4f}m, "
                f"fg_pixels_before_morph={fg_before_morph}, "
                f"fg_pixels_after_morph={fg_after_morph}, "
                f"table_mask_area={table_area}, "
                f"depth_range=[{float(valid_depth.min()):.3f}, {float(valid_depth.max()):.3f}]"
            )

        img_h, img_w = foreground.shape[:2]
        detections = []
        skipped_area = 0
        skipped_geometry = 0
        for mask in self.split_foreground_instances(foreground, bgr, depth_m):
            area = int(mask.sum())
            if area < self.min_mask_pixels:
                continue
            if area > table_area * self.max_table_object_area_ratio:
                skipped_area += 1
                continue
            bx, by, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
            edge_margin = 4
            touches_edge = (
                bx <= edge_margin
                or by <= edge_margin
                or (bx + bw) >= img_w - edge_margin
                or (by + bh) >= img_h - edge_margin
            )
            if touches_edge and min(bw, bh) < 30:
                continue

            local_table_depth = self.estimate_local_table_depth(
                mask, depth_m, table_mask, table_depth
            )
            geometry = self.describe_geometry(mask, depth_m, local_table_depth)
            if geometry is None:
                skipped_geometry += 1
                continue

            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            color_hint = self.describe_color(bgr, mask)
            shape_hint = geometry["shape"]
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
                f"{label_name} {geometry['grasp_yaw_deg']:.0f}deg",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        if not hasattr(self, "_logged_cluster_result"):
            self._logged_cluster_result = True
            self.get_logger().info(
                f"Depth clusters result: {len(detections)} detections, "
                f"skipped_area={skipped_area}, skipped_geometry={skipped_geometry}"
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
            np.percentile(valid_central_depth, max(self.table_depth_percentile, 80.0))
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

    def estimate_local_table_depth(self, object_mask, depth_m, table_mask, fallback_depth):
        object_mask_u8 = object_mask.astype(np.uint8)
        kernel = np.ones((31, 31), np.uint8)
        dilated = cv2.dilate(object_mask_u8, kernel, iterations=1).astype(bool)
        ring_mask = dilated & table_mask & (~object_mask)
        ring_depth = depth_m[ring_mask & np.isfinite(depth_m) & (depth_m > 0.05)]

        if ring_depth.size < 80:
            return float(fallback_depth)

        return float(np.percentile(ring_depth, 55))

    def split_foreground_instances(self, foreground, bgr, depth_m=None):
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

            depth_splits = self._try_depth_split(
                component_mask.astype(bool), depth_m
            )
            if depth_splits is not None and len(depth_splits) > 1:
                instance_masks.extend(depth_splits)
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

    def _try_depth_split(self, component_mask, depth_m, min_gap_m=0.025):
        """Split a merged foreground component using depth histogram valleys."""
        if depth_m is None:
            return None
        valid = component_mask & np.isfinite(depth_m) & (depth_m > 0.05)
        depths = depth_m[valid]
        if depths.size < self.min_mask_pixels * 2:
            return None

        depth_range = float(depths.max() - depths.min())
        if depth_range < min_gap_m * 2:
            return None

        n_bins = max(20, int(depth_range / 0.005))
        hist, bin_edges = np.histogram(depths, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]

        threshold = max(10, int(depths.size * 0.02))
        gap_bins = np.where(hist < threshold)[0]
        if gap_bins.size == 0:
            return None

        groups = np.split(gap_bins, np.where(np.diff(gap_bins) != 1)[0] + 1)
        best_gap = None
        best_gap_width = 0.0
        for g in groups:
            gap_width = float(len(g)) * bin_width
            if gap_width >= min_gap_m and gap_width > best_gap_width:
                best_gap = g
                best_gap_width = gap_width
        if best_gap is None:
            return None

        split_depth = float(bin_centers[best_gap[len(best_gap) // 2]])

        mask_near = component_mask & np.isfinite(depth_m) & (depth_m <= split_depth)
        mask_far = component_mask & np.isfinite(depth_m) & (depth_m > split_depth)

        kernel = np.ones((3, 3), np.uint8)
        mask_near_u8 = cv2.morphologyEx(
            mask_near.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )
        mask_far_u8 = cv2.morphologyEx(
            mask_far.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )

        results = []
        for m in [mask_near_u8, mask_far_u8]:
            nl, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for li in range(1, nl):
                a = int(st[li, cv2.CC_STAT_AREA])
                if a >= self.min_mask_pixels:
                    results.append((lbl == li).astype(bool))

        return results if len(results) >= 2 else None

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
        height, width = annotated.shape[:2]

        for det in final_detections:
            px = det.get("pixel")
            if not px:
                continue
            u, v = int(px[0]), int(px[1])
            shape = det.get("shape", "?")
            dims = det.get("dimensions_m", [0, 0, 0])
            yaw = det.get("grasp_yaw_deg", 0.0)
            label = f"{det['id']}:{shape}"
            dim_text = f"{dims[0]*100:.1f}x{dims[1]*100:.1f}x{dims[2]*100:.1f}cm"

            cv2.circle(annotated, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(
                annotated, label,
                (u + 8, max(14, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
            )
            cv2.putText(
                annotated, dim_text,
                (u + 8, v + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
            )

        bar_text = f"{backend_used} | masks:{raw_count} | objetos:{len(final_detections)}"
        text_w = cv2.getTextSize(bar_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.rectangle(annotated, (8, height - 34), (text_w + 24, height - 8), (30, 30, 30), -1)
        cv2.putText(
            annotated, bar_text,
            (16, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
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

    def build_3d_detections(
        self, detections, bgr, depth_m, image_frame_id="", camera_info_frame_id=""
    ):
        if self.camera_info is None:
            return []

        transform, source_frame = self.lookup_best_transform(
            [
                self.camera_optical_frame,
                image_frame_id,
                camera_info_frame_id,
                "camera_link_optical",
                "camera_link",
            ]
        )
        if transform is None or source_frame is None:
            if not hasattr(self, "_logged_tf_fail"):
                self._logged_tf_fail = True
                self.get_logger().warn(
                    f"No se encontro TF. Frames intentados: "
                    f"{[self.camera_optical_frame, image_frame_id, camera_info_frame_id]}"
                )
            return []

        counts = defaultdict(int)
        final_detections = []

        table_depth_est = self._quick_table_depth(depth_m)

        for detection in detections:
            geometry = detection.get("geometry")
            if geometry is None:
                geometry = self.describe_geometry(
                    detection["mask"], depth_m, table_depth_est
                )
                detection["geometry"] = geometry
            if detection.get("color_hint") is None:
                detection["color_hint"] = self.describe_color(bgr, detection["mask"])

            point_cam, pixel = self.compute_3d_point(
                detection["mask"],
                depth_m,
                geometry=geometry,
            )
            if point_cam is None:
                if not hasattr(self, "_logged_3d_fail"):
                    self._logged_3d_fail = True
                    self.get_logger().warn(
                        f"compute_3d_point devolvio None para '{detection['label']}'"
                    )
                continue

            point_base = self.transform_point(point_cam, transform)
            if point_base is None:
                continue

            bx, by, bz = float(point_base[0]), float(point_base[1]), float(point_base[2])
            if not (0.0 < bx < 1.0 and -0.5 < by < 0.5 and 0.0 < bz < 1.0):
                if not hasattr(self, "_logged_ws_fail"):
                    self._logged_ws_fail = True
                    self.get_logger().warn(
                        f"Punto base fuera de workspace para '{detection['label']}': "
                        f"bx={bx:.3f}, by={by:.3f}, bz={bz:.3f}"
                    )
                continue

            geo = geometry or {}
            det_label = detection["label"]
            color_hint = detection.get("color_hint")
            shape_hint = geo.get("shape")
            if color_hint and shape_hint:
                det_label = f"{color_hint}_{shape_hint}"

            # Merge detections that are likely the same physical object.
            # This avoids ghost duplicates caused by splitting/over-segmentation.
            mask_area = int(detection["mask"].sum()) if "mask" in detection else 0
            keep = True
            for prev in final_detections:
                prev_pos = prev.get("position", [0.0, 0.0, 0.0])
                prev_x, prev_y = float(prev_pos[0]), float(prev_pos[1])
                prev_z = float(prev_pos[2]) if len(prev_pos) >= 3 else 0.0
                # Don't merge across different colors; this prevents
                # accidentally losing the correct target label.
                prev_color = prev.get("color_hint")
                curr_color = detection.get("color_hint")
                if prev_color is not None and curr_color is not None and prev_color != curr_color:
                    continue
                dist_xy = float(np.hypot(prev_x - bx, prev_y - by))
                dist_z = abs(prev_z - bz)
                if dist_xy <= self.merge_duplicate_xy_dist_m and dist_z <= 0.05:
                    # Keep the largest segment as the representative.
                    prev_area = int(prev.get("mask_area", 0))
                    if mask_area > prev_area:
                        prev.update(
                            {
                                "position": [
                                    round(float(value), 4) for value in point_base
                                ],
                                "bbox": detection["bbox"],
                                "pixel": [int(pixel[0]), int(pixel[1])],
                                "orientation_deg": round(
                                    float(geo.get("orientation_deg", 0.0)), 2
                                ),
                                "grasp_yaw_deg": round(
                                    float(geo.get("grasp_yaw_deg", 0.0)), 2
                                ),
                                "dimensions_m": [
                                    round(float(value), 4)
                                    for value in geo.get(
                                        "dimensions_m", [0.0, 0.0, 0.0]
                                    )
                                ],
                                "height_m": round(float(geo.get("height_m", 0.0)), 4),
                                "grasp_type": geo.get("grasp_type"),
                                "color_hint": detection.get("color_hint"),
                                "confidence": round(float(detection["confidence"]), 4),
                                "mask_area": mask_area,
                            }
                        )
                    keep = False
                    break

            if not keep:
                continue

            counts[det_label] += 1
            indexed_label = det_label
            if counts[det_label] > 1:
                indexed_label = f"{det_label}_{counts[det_label]}"
            final_detections.append(
                {
                    "id": indexed_label,
                    "label": det_label,
                    "shape": geo.get("shape"),
                    "confidence": round(float(detection["confidence"]), 4),
                    "position": [round(float(value), 4) for value in point_base],
                    "bbox": detection["bbox"],
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "orientation_deg": round(float(geo.get("orientation_deg", 0.0)), 2),
                    "grasp_yaw_deg": round(float(geo.get("grasp_yaw_deg", 0.0)), 2),
                    "dimensions_m": [
                        round(float(value), 4)
                        for value in geo.get("dimensions_m", [0.0, 0.0, 0.0])
                    ],
                    "height_m": round(float(geo.get("height_m", 0.0)), 4),
                    "grasp_type": geo.get("grasp_type"),
                    "source_frame": source_frame,
                    "color_hint": detection.get("color_hint"),
                    "mask_area": mask_area,
                }
            )

        return final_detections

    def _quick_table_depth(self, depth_m):
        valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.05)]
        if valid.size == 0:
            return 0.76
        return float(np.percentile(valid, 85))

    def _get_scaled_intrinsics(self, image_height, image_width):
        """Return (fx, fy, cx, cy) scaled to match the actual image resolution.

        Gazebo may publish camera_info with width/height matching the image
        but K values computed for a smaller internal render resolution.
        We detect this when cx is far from image_width/2.
        """
        ci = self.camera_info
        fx = float(ci.k[0])
        fy = float(ci.k[4])
        cx = float(ci.k[2])
        cy = float(ci.k[5])

        expected_cx = image_width / 2.0
        expected_cy = image_height / 2.0

        if cx > 0 and (expected_cx / cx) > 1.5:
            sx = expected_cx / cx
            sy = expected_cy / max(1.0, cy)
            if not hasattr(self, "_logged_intrinsics_scale"):
                self._logged_intrinsics_scale = True
                self.get_logger().warn(
                    f"camera_info K inconsistente con imagen: "
                    f"cx={cx:.1f} (esperado ~{expected_cx:.1f}), "
                    f"cy={cy:.1f} (esperado ~{expected_cy:.1f}). "
                    f"Escalando x{sx:.2f}: fx {fx:.1f}->{fx*sx:.1f}, "
                    f"cx {cx:.1f}->{cx*sx:.1f}"
                )
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

        return fx, fy, cx, cy

    def compute_3d_point(self, mask, depth_m, geometry=None):
        valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0.05)
        ys, xs = np.where(valid_mask)
        if xs.size == 0:
            return None, None

        if geometry is not None and geometry.get("grasp_pixel") is not None:
            centroid_u, centroid_v = geometry["grasp_pixel"]
            centroid_depth = float(
                geometry.get("grasp_depth_m", np.median(depth_m[valid_mask]))
            )
        else:
            centroid_u = int(np.median(xs))
            centroid_v = int(np.median(ys))
            centroid_depth = float(np.median(depth_m[valid_mask]))

        img_h, img_w = depth_m.shape[:2]
        fx, fy, cx, cy = self._get_scaled_intrinsics(img_h, img_w)

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

    def lookup_best_transform(self, source_frames):
        tried = []
        for source_frame in source_frames:
            if not source_frame or source_frame in tried:
                continue
            tried.append(source_frame)
            transform = self.lookup_transform(source_frame)
            if transform is not None:
                return transform, source_frame
        return None, None

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

    # ------------------------------------------------------------------
    # 3D point-cloud extraction & geometry
    # ------------------------------------------------------------------

    def extract_object_pointcloud(self, mask, depth_m):
        """Convert a binary mask + depth map into Nx3 points in camera frame."""
        if self.camera_info is None:
            return np.empty((0, 3), dtype=np.float32)

        img_h, img_w = depth_m.shape[:2]
        fx, fy, cx, cy = self._get_scaled_intrinsics(img_h, img_w)

        valid = mask & np.isfinite(depth_m) & (depth_m > 0.05)
        vs, us = np.where(valid)
        if us.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        zs = depth_m[vs, us].astype(np.float32)
        xs = ((us.astype(np.float32) - cx) * zs / fx)
        ys = ((vs.astype(np.float32) - cy) * zs / fy)

        return np.column_stack((xs, ys, zs))

    @staticmethod
    def compute_obb_3d(points):
        """Oriented bounding box from an Nx3 point cloud.

        Returns (dimensions_sorted, axes, center) where dimensions_sorted is
        [largest, middle, smallest] in metres, axes is 3x3 (rows = principal
        directions), and center is the OBB centre.
        """
        if points.shape[0] < 4:
            return None

        center = points.mean(axis=0)
        centered = points - center

        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order].T

        projected = centered @ axes.T
        mins = projected.min(axis=0)
        maxs = projected.max(axis=0)
        dims = maxs - mins

        sort_idx = np.argsort(dims)[::-1]
        dims_sorted = dims[sort_idx]
        axes_sorted = axes[sort_idx]

        obb_center = center + axes.T @ ((mins + maxs) / 2.0)
        return dims_sorted, axes_sorted, obb_center

    def describe_geometry(self, mask, depth_m, table_depth):
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

        valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0.05)
        valid_depth_values = depth_m[valid_mask]
        if valid_depth_values.size == 0:
            return None

        pc = self.extract_object_pointcloud(mask, depth_m)
        if pc.shape[0] < 10:
            return None

        obb = self.compute_obb_3d(pc)
        if obb is None:
            return None

        dims_sorted, axes, obb_center = obb
        dim_x, dim_y, dim_z = [float(d) for d in dims_sorted]

        if dim_x > self.max_object_dimension_m:
            return None

        top_surface_depth = float(np.percentile(valid_depth_values, 12))
        height_m = max(0.0, float(table_depth) - top_surface_depth)
        grasp_pixel = self.compute_top_surface_grasp_pixel(
            valid_mask, depth_m, top_surface_depth
        )
        # Use local depth around the selected grasp pixel.
        # For elongated/sideways objects, depth is not constant across the
        # top band; using the global percentile depth can shift the 3D point.
        grasp_depth_m = top_surface_depth
        if grasp_pixel is not None and isinstance(grasp_pixel, (list, tuple)):
            if len(grasp_pixel) == 2:
                u, v = int(grasp_pixel[0]), int(grasp_pixel[1])
                img_h, img_w = depth_m.shape[:2]
                u = int(np.clip(u, 0, img_w - 1))
                v = int(np.clip(v, 0, img_h - 1))
                patch = depth_m[max(0, v - 1) : min(img_h, v + 2), max(0, u - 1) : min(img_w, u + 2)]
                patch_valid = patch[np.isfinite(patch) & (patch > 0.05)]
                if patch_valid.size > 0:
                    grasp_depth_m = float(np.median(patch_valid))
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (size_w, size_h), rect_angle = rect
        box_points = cv2.boxPoints(rect).astype(np.int32)

        orientation_deg = float(rect_angle)
        if size_w < size_h:
            orientation_deg += 90.0
        if orientation_deg >= 90.0:
            orientation_deg -= 180.0

        fill_ratio = contour_area / max(1.0, float(size_w) * float(size_h))
        circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)
        depth_std = float(np.std(pc[:, 2]))

        shape = self.describe_shape_3d(
            dim_x=dim_x, dim_y=dim_y, dim_z=dim_z,
            height_m=height_m,
            fill_ratio=fill_ratio,
            circularity=circularity,
            depth_std=depth_std,
        )
        grasp_type, grasp_yaw_deg = self.estimate_grasp(
            shape, orientation_deg, dim_x, dim_y, height_m
        )

        return {
            "center": (float(center_x), float(center_y)),
            "orientation_deg": orientation_deg,
            "fill_ratio": fill_ratio,
            "circularity": circularity,
            "box_points": box_points,
            "grasp_pixel": grasp_pixel,
            "grasp_depth_m": grasp_depth_m,
            "shape": shape,
            "grasp_type": grasp_type,
            "grasp_yaw_deg": grasp_yaw_deg,
            "height_m": height_m,
            "depth_std": depth_std,
            "dimensions_m": [dim_x, dim_y, dim_z],
        }

    def compute_top_surface_grasp_pixel(self, valid_mask, depth_m, top_surface_depth):
        top_band_mask = valid_mask & (depth_m <= (top_surface_depth + 0.012))
        if int(top_band_mask.sum()) < max(40, self.min_mask_pixels // 8):
            top_band_mask = valid_mask

        top_band_u8 = top_band_mask.astype(np.uint8)
        distance = cv2.distanceTransform(top_band_u8, cv2.DIST_L2, 5)

        ys, xs = np.where(top_band_mask)
        if xs.size == 0:
            return None

        # Default to the top-band centroid. This keeps the grasp point
        # aligned under the object region for elongated/sideways shapes.
        centroid_u = int(np.median(xs))
        centroid_v = int(np.median(ys))

        max_value = float(distance.max())
        if max_value > 0.0:
            _, _, _, max_loc = cv2.minMaxLoc(distance)
            max_u, max_v = int(max_loc[0]), int(max_loc[1])
            # If the "center" from distance-transform is too far from the
            # centroid, fall back to centroid (prevents endpoint picks).
            if (abs(max_u - centroid_u) + abs(max_v - centroid_v)) <= 60:
                return [max_u, max_v]

        return [centroid_u, centroid_v]

    # ------------------------------------------------------------------
    # 3D shape classifier
    # ------------------------------------------------------------------

    @staticmethod
    def describe_shape_3d(
        *, dim_x, dim_y, dim_z, height_m, fill_ratio, circularity, depth_std
    ):
        if max(dim_x, dim_y, dim_z) < 1e-4:
            return "unknown"

        sorted_dims = sorted([dim_x, dim_y, dim_z], reverse=True)
        d_large, d_mid, d_small = sorted_dims

        ratio_12 = d_mid / max(d_large, 1e-6)
        ratio_13 = d_small / max(d_large, 1e-6)

        if circularity > 0.78 and fill_ratio > 0.74:
            if ratio_12 > 0.60:
                return "cylinder"

        if ratio_12 > 0.65 and ratio_13 > 0.55:
            return "cube"

        if ratio_13 < 0.30:
            if ratio_12 > 0.55:
                return "flat_part"
            return "elongated"

        if depth_std > 0.015 and fill_ratio < 0.60:
            return "tetrahedron_like"

        if ratio_12 > 0.55 and ratio_13 >= 0.30:
            return "box"

        return "box"

    # ------------------------------------------------------------------
    # Grasp strategy
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_grasp(shape, orientation_deg, dim_x, dim_y, height_m):
        # For sideways/lying objects, the gripper should be perpendicular
        # to the principal axis of the object in the table plane.
        if shape == "cylinder":
            return "top_parallel", orientation_deg + 90.0
        if shape == "cube":
            return "top_center", 0.0
        if shape == "tetrahedron_like":
            return "top_center_cautious", orientation_deg

        min_horizontal = min(dim_x, dim_y)
        if min_horizontal > 0.08:
            return "top_parallel", orientation_deg + 90.0

        return "top_parallel", orientation_deg + 90.0

    def log_detections_if_changed(self, final_detections):
        summary = tuple(
            (
                detection["id"],
                detection.get("shape"),
                round(float(detection.get("grasp_yaw_deg", 0.0)), 1),
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

        header = f"{'ID':<22} {'Shape':<16} {'Grasp':<20} {'Dims(cm)':<22} {'Pos(m)'}"
        lines = [header, "-" * len(header)]
        for d in final_detections:
            dims = d.get("dimensions_m", [0, 0, 0])
            pos = d["position"]
            dims_str = f"{dims[0]*100:.1f}x{dims[1]*100:.1f}x{dims[2]*100:.1f}"
            pos_str = f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
            shape = d.get('shape') or '?'
            gtype = d.get('grasp_type') or '?'
            gyaw = d.get('grasp_yaw_deg') or 0
            grasp_str = f"{gtype} {gyaw:.0f}deg"
            lines.append(
                f"{d['id']:<22} {shape:<16} {grasp_str:<20} {dims_str:<22} {pos_str}"
            )
        self.get_logger().info("Detecciones 3D:\n" + "\n".join(lines))

    @staticmethod
    def sanitize_label(label):
        return label.strip().lower().replace(" ", "_")

    @staticmethod
    def to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) / 1e9


def main(args=None):
    """Deprecated entrypoint: use ``perception_node`` for the modular pipeline."""
    from panda_vision.nodes.perception_node import main as perception_main

    perception_main(args)


if __name__ == "__main__":
    main()
