"""
pairwise-gesture-system - GestureSystem

A robust gesture pipeline:

- record_gesture:   unified resampling, origin/scale normalization, smoothing
- recognize_gesture: shape matching with confidence scoring and timeout
- analyze_trajectory: full-path, direction and speed analysis
- save_gesture:     validated custom templates with augmentation and updating
- give_feedback:    in-progress, visual and error feedback
"""

import math
import time


class GestureSystem:
    SAMPLE_COUNT = 64
    SMOOTH_WINDOW = 3
    MIN_POINTS = 8
    RECOGNITION_TIMEOUT = 3.0
    MATCH_THRESHOLD = 0.6
    DISTANCE_SCALE = 0.5
    MAX_TEMPLATES_PER_GESTURE = 32

    def __init__(self):
        self.state = {}
        self.templates = {}

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def record_gesture(self, points=None, timestamps=None, sample_count=None):
        """Record a raw stroke into a canonical gesture representation.

        The raw input is smoothed (moving average), resampled to a fixed
        number of points spaced evenly along the path (so drawing speed no
        longer changes the sample count), and normalized so the gesture
        starts at the origin and fits a unit box (so the same gesture drawn
        at a different position or size is identical).
        """
        count = sample_count or self.SAMPLE_COUNT
        result = {
            "points": [],
            "raw_count": 0,
            "sample_count": count,
            "normalized": True,
            "smoothed": True,
            "duration": None,
            "valid": False,
            "reason": None,
        }
        if not points:
            result["reason"] = "no_input"
            return result

        pts = [(float(p[0]), float(p[1])) for p in points]
        result["raw_count"] = len(pts)
        if timestamps and len(timestamps) == len(pts):
            result["duration"] = float(timestamps[-1]) - float(timestamps[0])
        if len(pts) < 2:
            result["reason"] = "too_few_points"
            return result

        smoothed = self._smooth(pts, self.SMOOTH_WINDOW)
        resampled = self._resample(smoothed, count)
        normalized = self._normalize(resampled)

        result["points"] = normalized
        result["valid"] = True
        return result

    # ------------------------------------------------------------------
    # recognition
    # ------------------------------------------------------------------
    def recognize_gesture(self, points=None, templates=None, start_time=None,
                          timeout=None, sample_count=None):
        """Match a stroke against stored templates by shape.

        Returns a result dict that always carries a confidence score; a
        match is only reported when the confidence clears the threshold.
        Stalled input past the timeout is reported as a timeout instead of
        being matched against templates.
        """
        limit = self.RECOGNITION_TIMEOUT if timeout is None else float(timeout)
        result = {
            "matched": False,
            "name": None,
            "confidence": 0.0,
            "scores": {},
            "reason": None,
            "timed_out": False,
        }

        if start_time is not None and (time.time() - float(start_time)) > limit:
            result["reason"] = "timeout"
            result["timed_out"] = True
            return result
        if not points:
            result["reason"] = "no_input"
            return result

        library = templates if templates is not None else self.templates
        if not library:
            result["reason"] = "no_templates"
            return result

        recording = self.record_gesture(points, sample_count=sample_count)
        if not recording["valid"]:
            result["reason"] = recording["reason"]
            return result
        candidate = recording["points"]

        best_name, best_confidence = None, 0.0
        for name, variants in self._iter_templates(library):
            confidence = 0.0
            for variant in variants:
                prepared = self._prepare_template(variant, len(candidate))
                distance = self._best_shape_distance(candidate, prepared)
                confidence = max(confidence,
                                 max(0.0, 1.0 - distance / self.DISTANCE_SCALE))
            result["scores"][name] = confidence
            if confidence > best_confidence:
                best_name, best_confidence = name, confidence

        result["name"] = best_name
        result["confidence"] = best_confidence
        if best_confidence >= self.MATCH_THRESHOLD:
            result["matched"] = True
        else:
            result["reason"] = "low_confidence"
        return result

    # ------------------------------------------------------------------
    # trajectory analysis
    # ------------------------------------------------------------------
    def analyze_trajectory(self, points=None, timestamps=None):
        """Analyze the whole path, its direction and its speed profile."""
        result = {
            "start": None,
            "end": None,
            "path_length": 0.0,
            "displacement": 0.0,
            "straightness": 0.0,
            "closed": False,
            "signed_area": 0.0,
            "turning_angle": 0.0,
            "direction": "none",
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "min_speed": 0.0,
            "speed_variance": 0.0,
            "valid": False,
            "reason": None,
        }
        if not points or len(points) < 2:
            result["reason"] = "too_few_points"
            return result

        pts = [(float(p[0]), float(p[1])) for p in points]
        if timestamps and len(timestamps) == len(pts):
            times = [float(t) for t in timestamps]
        else:
            times = [float(i) for i in range(len(pts))]

        segments = []
        speeds = []
        for i in range(1, len(pts)):
            seg = math.hypot(pts[i][0] - pts[i - 1][0],
                             pts[i][1] - pts[i - 1][1])
            segments.append(seg)
            dt = times[i] - times[i - 1]
            speeds.append(seg / dt if dt > 0 else 0.0)

        path_length = sum(segments)
        displacement = math.hypot(pts[-1][0] - pts[0][0],
                                  pts[-1][1] - pts[0][1])
        signed_area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            signed_area += x1 * y2 - x2 * y1
        signed_area /= 2.0

        turning = 0.0
        for i in range(1, len(pts) - 1):
            a1 = math.atan2(pts[i][1] - pts[i - 1][1],
                            pts[i][0] - pts[i - 1][0])
            a2 = math.atan2(pts[i + 1][1] - pts[i][1],
                            pts[i + 1][0] - pts[i][0])
            delta = a2 - a1
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
            turning += delta

        extent = max(math.hypot(*(self._bbox_size(pts))), 1e-9)
        avg_speed = sum(speeds) / len(speeds)
        variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)

        result.update({
            "start": pts[0],
            "end": pts[-1],
            "path_length": path_length,
            "displacement": displacement,
            "straightness": displacement / path_length if path_length else 0.0,
            "closed": displacement < 0.1 * extent,
            "signed_area": signed_area,
            "turning_angle": turning,
            "direction": ("counterclockwise" if signed_area > 0
                          else "clockwise" if signed_area < 0 else "none"),
            "avg_speed": avg_speed,
            "max_speed": max(speeds),
            "min_speed": min(speeds),
            "speed_variance": variance,
            "valid": True,
        })
        return result

    # ------------------------------------------------------------------
    # custom gesture templates
    # ------------------------------------------------------------------
    def save_gesture(self, name=None, points=None, samples=None):
        """Save a custom gesture with validation, augmentation and updating.

        A stroke that is too short to be a real gesture is rejected. Each
        accepted sample is stored together with augmented variants (small
        rotations, scales and a smoothed copy) so natural drawing variance
        still matches. Saving again under an existing name updates the
        template set, letting it drift with the player's habits.
        """
        result = {"saved": False, "name": name, "template_count": 0,
                  "reason": None}
        if not name or not isinstance(name, str):
            result["reason"] = "invalid_name"
            return result

        raw_samples = []
        if samples:
            raw_samples.extend(samples)
        if points:
            raw_samples.append(points)
        raw_samples = [s for s in raw_samples if s]
        if not raw_samples:
            result["reason"] = "no_input"
            return result
        if any(len(s) < self.MIN_POINTS for s in raw_samples):
            result["reason"] = "too_few_points"
            return result

        templates = self.templates.setdefault(name, [])
        for sample in raw_samples:
            recording = self.record_gesture(sample)
            if not recording["valid"]:
                result["reason"] = recording["reason"]
                return result
            templates.extend(self._augment(recording["points"]))

        del templates[:-self.MAX_TEMPLATES_PER_GESTURE]
        result["saved"] = True
        result["template_count"] = len(templates)
        return result

    # ------------------------------------------------------------------
    # feedback
    # ------------------------------------------------------------------
    def give_feedback(self, status="idle", progress=None, result=None,
                      error=None):
        """Build feedback for any stage of the gesture flow.

        Feedback is produced while a gesture is still being drawn (not only
        on success), always includes a visual payload (path rendering hint,
        color and progress bar) and explains failures with a concrete error
        hint instead of staying silent.
        """
        in_progress = status in ("recording", "recognizing", "in_progress")
        visual = {
            "show_path": in_progress or status == "failed",
            "progress": progress,
            "color": "blue" if in_progress else "green",
        }
        feedback = {
            "status": status,
            "in_progress": in_progress,
            "text": "等待手势输入",
            "visual": visual,
            "error": None,
        }

        if in_progress:
            feedback["text"] = "正在识别手势，请继续"
            return feedback

        if status == "success" or (result and result.get("matched")):
            name = (result or {}).get("name")
            confidence = (result or {}).get("confidence")
            feedback["text"] = "识别成功：%s（置信度 %.2f）" % (name, confidence)
            return feedback

        reason = error or (result or {}).get("reason")
        hints = {
            "timeout": "识别超时：画到一半停住了，请一口气画完",
            "no_input": "没有检测到手势输入",
            "no_templates": "还没有保存任何手势模板",
            "too_few_points": "轨迹太短：一笔不算手势，请画完整形状",
            "low_confidence": "形状不匹配：请对照模板再画一次",
            "invalid_name": "手势名称无效",
        }
        if status == "failed" or reason:
            feedback["text"] = "识别失败"
            feedback["error"] = {
                "reason": reason,
                "hint": hints.get(reason, "手势无法识别，请重试"),
            }
            feedback["visual"]["color"] = "red"
        return feedback

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth(points, window):
        if window < 2 or len(points) <= 2:
            return list(points)
        half = window // 2
        smoothed = []
        for i in range(len(points)):
            lo = max(0, i - half)
            hi = min(len(points), i + half + 1)
            xs = sum(p[0] for p in points[lo:hi]) / (hi - lo)
            ys = sum(p[1] for p in points[lo:hi]) / (hi - lo)
            smoothed.append((xs, ys))
        return smoothed

    @staticmethod
    def _resample(points, count):
        if len(points) < 2:
            return list(points)
        cumulative = [0.0]
        for i in range(1, len(points)):
            cumulative.append(cumulative[-1] + math.hypot(
                points[i][0] - points[i - 1][0],
                points[i][1] - points[i - 1][1]))
        total = cumulative[-1]
        if total == 0:
            return [points[0]] * count
        step = total / (count - 1)
        resampled = [points[0]]
        target = step
        i = 1
        while i < len(points):
            if cumulative[i] >= target:
                prev_len = cumulative[i - 1]
                seg_len = cumulative[i] - prev_len
                ratio = 0.0 if seg_len == 0 else (target - prev_len) / seg_len
                x = points[i - 1][0] + ratio * (points[i][0] - points[i - 1][0])
                y = points[i - 1][1] + ratio * (points[i][1] - points[i - 1][1])
                resampled.append((x, y))
                target += step
                if len(resampled) >= count:
                    break
            else:
                i += 1
        while len(resampled) < count:
            resampled.append(points[-1])
        return resampled[:count]

    @staticmethod
    def _normalize(points):
        if not points:
            return []
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        scale = max(max_x - min_x, max_y - min_y)
        if scale == 0:
            scale = 1.0
        origin = points[0]
        return [((p[0] - origin[0]) / scale, (p[1] - origin[1]) / scale)
                for p in points]

    @staticmethod
    def _bbox_size(points):
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        return max_x - min_x, max_y - min_y

    @staticmethod
    def _shape_distance(a, b):
        if not a or not b:
            return float("inf")
        n = min(len(a), len(b))
        return sum(math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1])
                   for i in range(n)) / n

    @staticmethod
    def _centroid(points):
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))

    def _best_shape_distance(self, candidate, template):
        """Shape distance invariant to rotation ($1-recognizer style).

        Normalized strokes share the same origin (their first point), so
        the same shape drawn rotated only differs by a rotation about that
        origin. A golden-section search over the full angle range finds the
        best-fitting orientation before points are compared.
        """
        base = self._shape_distance(candidate, template)

        def distance_at(angle):
            rotated = self._rotate(candidate, angle, center=(0.0, 0.0))
            return self._shape_distance(rotated, template)

        best = base
        half = math.pi / 8.0
        for k in range(8):
            center = -math.pi + k * math.pi / 4.0
            best = min(best,
                       self._golden_section(distance_at,
                                            center - half, center + half))
        return best

    @staticmethod
    def _golden_section(fn, lo, hi, iterations=10):
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - ratio * (hi - lo)
        x2 = lo + ratio * (hi - lo)
        f1, f2 = fn(x1), fn(x2)
        for _ in range(iterations):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - ratio * (hi - lo)
                f1 = fn(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + ratio * (hi - lo)
                f2 = fn(x2)
        return min(f1, f2)

    def _prepare_template(self, template, count):
        pts = [(float(p[0]), float(p[1])) for p in template]
        if len(pts) != count:
            pts = self._resample(pts, count)
        return self._normalize(pts)

    @staticmethod
    def _iter_templates(library):
        for name, entry in library.items():
            if not entry:
                continue
            first = entry[0]
            if isinstance(first, (list, tuple)) and first and \
                    isinstance(first[0], (int, float)):
                yield name, [entry]
            else:
                yield name, list(entry)

    def _augment(self, points):
        variants = [list(points), self._smooth(points, 5)]
        for angle in (-30, -15, 15, 30):
            variants.append(self._rotate(points, math.radians(angle)))
        for factor in (0.9, 1.1):
            variants.append([(p[0] * factor, p[1] * factor) for p in points])
        return variants

    @staticmethod
    def _rotate(points, angle, center=None):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        if center is None:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
        else:
            cx, cy = center
        rotated = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            rotated.append((cx + dx * cos_a - dy * sin_a,
                            cy + dx * sin_a + dy * cos_a))
        return rotated
