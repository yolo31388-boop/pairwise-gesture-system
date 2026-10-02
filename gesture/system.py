"""
pairwise-gesture-system - GestureSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class GestureSystem:
    def __init__(self):
        self.state = {}

    def record_gesture(self, *args, **kwargs):
        """BUG: 手势录制不做采样率统一，快画和慢画的采样点数量不一样"""
        # Intentionally broken implementation
        return None

    def recognize_gesture(self, *args, **kwargs):
        """BUG: 手势识别只用简单点距离比较，不做形状匹配"""
        # Intentionally broken implementation
        return None

    def analyze_trajectory(self, *args, **kwargs):
        """BUG: 轨迹分析只看起点终点，不看中间路径"""
        # Intentionally broken implementation
        return None

    def save_gesture(self, *args, **kwargs):
        """BUG: 自定义手势只保存一个模板，不做模板增强"""
        # Intentionally broken implementation
        return None

    def give_feedback(self, *args, **kwargs):
        """BUG: 手势反馈只在识别成功后给，识别过程中没有反馈"""
        # Intentionally broken implementation
        return None

