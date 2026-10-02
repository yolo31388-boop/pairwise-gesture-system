import pytest
from gesture.system import GestureSystem


class TestGestureSystem:

    def test_record_gesture(self):
        """手势录制不做采样率统一，快画和慢画的采样点数量不一样"""
        obj = GestureSystem()
        result = obj.record_gesture()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_recognize_gesture(self):
        """手势识别只用简单点距离比较，不做形状匹配"""
        obj = GestureSystem()
        result = obj.recognize_gesture()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_analyze_trajectory(self):
        """轨迹分析只看起点终点，不看中间路径"""
        obj = GestureSystem()
        result = obj.analyze_trajectory()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_save_gesture(self):
        """自定义手势只保存一个模板，不做模板增强"""
        obj = GestureSystem()
        result = obj.save_gesture()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_give_feedback(self):
        """手势反馈只在识别成功后给，识别过程中没有反馈"""
        obj = GestureSystem()
        result = obj.give_feedback()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

