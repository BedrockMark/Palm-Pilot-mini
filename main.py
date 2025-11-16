"""
手势识别测试应用程序
使用 WxPython 构建图形界面，集成 MediaPipe 实时手部检测和 GestureClassifier 进行训练与推理
"""

import wx
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from typing import Optional, List
import logging

from classifier.classifier import GestureClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureRecognitionApp(wx.Frame):
    """
    主应用窗口
    
    属性:
    - cap: cv2.VideoCapture 摄像头对象
    - mp_hands: MediaPipe Hands 检测器
    - classifier: GestureClassifier 实例
    - timer: wx.Timer 视频刷新定时器
    - training_samples: list 暂存训练样本
    - current_gesture_name: str 当前训练的手势名称
    - is_training: bool 训练模式标志
    - current_landmarks: 当前检测到的手部关键点
    - fps_counter: FPS 计数器
    """
    
    def __init__(self):
        """初始化窗口和组件"""
        super().__init__(None, title="Palm Pilot Mini - V0", size=(800, 800))
        
        # 初始化变量
        self.cap: Optional[cv2.VideoCapture] = None
        self.mp_hands = None
        self.hands = None
        self.classifier: Optional[GestureClassifier] = None
        self.timer = None
        self.training_samples: List[np.ndarray] = []
        self.current_gesture_name: str = ""
        self.is_training: bool = False
        self.current_landmarks: Optional[np.ndarray] = None
        self.fps_counter = {"count": 0, "last_time": time.time(), "fps": 0}
        self.is_recording: bool = False
        
        # 初始化组件
        self.init_components()
        
        # 初始化摄像头和 MediaPipe
        self.init_camera()
        self.init_mediapipe()
        self.init_classifier()
        
        # 绑定关闭事件
        self.Bind(wx.EVT_CLOSE, self.on_close)
        
        # 绑定键盘事件
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)
        
        logger.info("Application initialized")
    
    def init_components(self):
        """Initiate components"""
        self.create_ui()
        self.Center()
    
    def create_ui(self):
        """Initiate UI"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="Palm Pilot Mini - V0", style=wx.ALIGN_CENTER)
        title_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.EXPAND, 10)
        
        # Video area
        self.video_panel = wx.StaticBitmap(panel, size=(640, 480))
        main_sizer.Add(self.video_panel, 0, wx.ALL | wx.CENTER, 10)
        
        # Button hbox
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Training button - Green
        self.btn_record = wx.Button(panel, label="Train", size=(120, 40))
        self.btn_record.SetBackgroundColour(wx.Colour(76, 175, 80))  # GREEN
        self.btn_record.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btn_record.Bind(wx.EVT_BUTTON, self.on_record_train)
        button_sizer.Add(self.btn_record, 0, wx.ALL, 5)
        
        # Inference button - Blue
        self.btn_inference = wx.Button(panel, label="Inference", size=(120, 40))
        self.btn_inference.SetBackgroundColour(wx.Colour(33, 150, 243))  # BLUE
        self.btn_inference.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btn_inference.Bind(wx.EVT_BUTTON, self.on_inference)
        button_sizer.Add(self.btn_inference, 0, wx.ALL, 5)
        
        # Clear button - Red - WARNING: This is clearing all train data!
        self.btn_clear = wx.Button(panel, label="Clear", size=(120, 40))
        self.btn_clear.SetBackgroundColour(wx.Colour(244, 67, 54))  # RED
        self.btn_clear.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btn_clear.Bind(wx.EVT_BUTTON, self.on_clear_model)
        button_sizer.Add(self.btn_clear, 0, wx.ALL, 5)
        
        # Save button - orange
        self.btn_save = wx.Button(panel, label="Save", size=(120, 40))
        self.btn_save.SetBackgroundColour(wx.Colour(255, 152, 0))  # ORANGE
        self.btn_save.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btn_save.Bind(wx.EVT_BUTTON, self.on_save_model)
        button_sizer.Add(self.btn_save, 0, wx.ALL, 5)
        
        # Load button - purple
        self.btn_load = wx.Button(panel, label="Load", size=(120, 40))
        self.btn_load.SetBackgroundColour(wx.Colour(156, 39, 176))  # PURPLE
        self.btn_load.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btn_load.Bind(wx.EVT_BUTTON, self.on_load_model)
        button_sizer.Add(self.btn_load, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        # Statements - vbox
        status_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.status_label = wx.StaticText(panel, label="Training State: None")
        status_sizer.Add(self.status_label, 0, wx.ALL, 5)
        
        self.sample_count_label = wx.StaticText(panel, label="Trained gesture count: 0")
        status_sizer.Add(self.sample_count_label, 0, wx.ALL, 5)
        
        self.gesture_label = wx.StaticText(panel, label="Current gesture: None")
        status_sizer.Add(self.gesture_label, 0, wx.ALL, 5)
        
        self.fps_label = wx.StaticText(panel, label="FPS: 0")
        status_sizer.Add(self.fps_label, 0, wx.ALL, 5)
        
        main_sizer.Add(status_sizer, 0, wx.ALL, 10)
        
        panel.SetSizer(main_sizer)
        self.Layout()
    
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 启动定时器更新视频流
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.refresh_frame, self.timer)
            self.timer.Start(33)  # 30 FPS
            
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            wx.MessageBox(f"Failed to initialize camera: {e}", "ERROR", wx.OK | wx.ICON_ERROR)
    
    def init_mediapipe(self):
        """初始化 MediaPipe"""
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            wx.MessageBox(f"MediaPipe Failed to initiate: {e}", "ERROR", wx.OK | wx.ICON_ERROR)
    
    def init_classifier(self):
        """初始化分类器"""
        try:
            self.classifier = GestureClassifier()
            logger.info("Classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            wx.MessageBox(f"Classifier Failed to initiate: {e}", "ERROR", wx.OK | wx.ICON_ERROR)
    
    def refresh_frame(self, event):
        """定时器回调，更新视频流"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.video_panel.Refresh(False)

        # 镜像翻转（更符合镜子效果）
        frame = cv2.flip(frame, 1)
        
        # 检测手部关键点
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # 绘制关键点
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # 提取关键点坐标
            self.current_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        else:
            self.current_landmarks = None
        
        # 计算并显示 FPS
        self.fps_counter["count"] += 1
        current_time = time.time()
        if current_time - self.fps_counter["last_time"] >= 1.0:
            self.fps_counter["fps"] = self.fps_counter["count"]
            self.fps_counter["count"] = 0
            self.fps_counter["last_time"] = current_time
            self.fps_label.SetLabel(f"FPS: {self.fps_counter['fps']}")
        
        # 显示检测状态
        status_text = "Hand State: "
        if self.current_landmarks is not None:
            status_text += "Yep"
        else:
            status_text += "Nope"
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps_counter['fps']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 转换为 wx.Bitmap 并显示
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = wx.Bitmap.FromBuffer(width, height, frame_rgb)
        self.video_panel.SetBitmap(image)
    
    def capture_snapshot(self) -> Optional[np.ndarray]:
        """
        捕获当前帧并提取关键点
        
        返回:
        - 归一化后的关键点数组，如果未检测到手部则返回 None
        """
        if self.current_landmarks is None:
            return None
        
        try:
            # 归一化关键点
            normalized = self.classifier.normalize(self.current_landmarks, rotation_normalize=True)
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize landmarks: {e}")
            return None
    
    def on_record_train(self, event):
        """录制训练按钮回调"""
        if self.is_training:
            wx.MessageBox("Training session already exists!", "WARNING", wx.OK | wx.ICON_INFORMATION)
            return
        
        if self.is_recording:
            self.continue_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        # 输入手势名称
        dialog = wx.TextEntryDialog(self, "Gesture name:", "Training...", "")
        if dialog.ShowModal() == wx.ID_OK:
            gesture_name = dialog.GetValue().strip()
            if not gesture_name:
                wx.MessageBox("Gesture name can't be null", "ERROR", wx.OK | wx.ICON_ERROR)
                dialog.Destroy()
                return
            dialog.Destroy()
            
            self.current_gesture_name = gesture_name
            self.training_samples = []
            self.is_recording = True
            self.show_recording_dialog()
        else:
            dialog.Destroy()
    
    def continue_recording(self):
        """继续录制样本"""
        if self.current_landmarks is None:
            wx.MessageBox("未检测到手势，请将手放在摄像头前", "提示", wx.OK | wx.ICON_WARNING)
            return
        
        normalized = self.capture_snapshot()
        if normalized is not None:
            self.training_samples.append(normalized)
            self.update_status_display()
            self.show_recording_dialog()
        else:
            wx.MessageBox("提取关键点失败，请重试", "错误", wx.OK | wx.ICON_ERROR)
    
    def show_recording_dialog(self):
        """显示录制对话框"""
        count = len(self.training_samples)
        message = f"已录制样本: {count}\n\n请保持手势，准备拍摄下一张..."
        
        dialog = wx.Dialog(self, title="录制样本", size=(300, 200))
        panel = wx.Panel(dialog)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        text = wx.StaticText(panel, label=message)
        sizer.Add(text, 0, wx.ALL | wx.CENTER, 20)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn_next = wx.Button(panel, label="拍摄下一张", id=wx.ID_OK)
        btn_next.Bind(wx.EVT_BUTTON, lambda e: dialog.EndModal(wx.ID_OK))
        btn_sizer.Add(btn_next, 0, wx.ALL, 5)
        
        btn_finish = wx.Button(panel, label="完成训练", id=wx.ID_CANCEL)
        btn_finish.Bind(wx.EVT_BUTTON, lambda e: dialog.EndModal(wx.ID_CANCEL))
        btn_sizer.Add(btn_finish, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)
        panel.SetSizer(sizer)
        
        result = dialog.ShowModal()
        dialog.Destroy()
        
        if result == wx.ID_OK:
            # 继续录制
            wx.CallAfter(self.continue_recording)
        else:
            # 完成训练
            self.finish_recording()
    
    def finish_recording(self):
        """完成录制并开始训练"""
        if len(self.training_samples) < 10:
            response = wx.MessageBox(
                f"当前只有 {len(self.training_samples)} 个样本，建议至少录制10个样本。\n是否继续训练？",
                "样本数量不足",
                wx.YES_NO | wx.ICON_QUESTION
            )
            if response == wx.NO:
                self.is_recording = False
                return
        
        # 在后台线程中训练，避免界面冻结
        self.is_training = True
        self.is_recording = False
        self.btn_record.Enable(False)
        self.btn_inference.Enable(False)
        
        # 创建训练线程
        thread = threading.Thread(target=self.train_model_thread, daemon=True)
        thread.start()
    
    def train_model_thread(self):
        """训练模型的线程函数"""
        try:
            # 保存样本数量（在清空之前）
            sample_count = len(self.training_samples)
            
            # 确定标签ID：如果手势已存在，使用其ID；否则使用下一个ID
            if self.current_gesture_name in self.classifier.gesture_map.values():
                # 已存在的类别，获取其ID
                gesture_id = next(k for k, v in self.classifier.gesture_map.items() 
                                if v == self.current_gesture_name)
            else:
                # 新类别，使用下一个ID
                gesture_id = self.classifier.next_gesture_id
            
            # 准备训练数据
            labels = [gesture_id] * sample_count
            
            # 调用训练方法
            wx.CallAfter(self.update_status_display, f"训练中... (样本数: {sample_count})")
            
            history = self.classifier.train(
                training_data=self.training_samples,
                labels=labels,
                gesture_name=self.current_gesture_name,
                epochs=50,
                lr=0.001,
                use_augmentation=True
            )
            
            # 训练完成，更新UI（传递样本数量）
            wx.CallAfter(self.on_training_complete, history, sample_count)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            wx.CallAfter(self.on_training_error, str(e))
    
    def on_training_complete(self, history, sample_count):
        """训练完成回调"""
        self.is_training = False
        self.btn_record.Enable(True)
        self.btn_inference.Enable(True)
        
        # 保存手势名称（在清空之前）
        gesture_name = self.current_gesture_name
        
        # 清空训练样本
        self.training_samples = []
        self.current_gesture_name = ""
        
        # 更新状态
        gesture_list = ", ".join(self.classifier.gesture_map.values())
        self.update_status_display()
        
        # 显示成功提示
        final_loss = history['loss'][-1] if history['loss'] else 0
        final_acc = history['accuracy'][-1] if history['accuracy'] else 0
        
        wx.MessageBox(
            f"训练完成！\n\n"
            f"手势名称: {gesture_name}\n"
            f"样本数量: {sample_count}\n"
            f"最终损失: {final_loss:.4f}\n"
            f"最终准确率: {final_acc:.2f}%",
            "训练成功",
            wx.OK | wx.ICON_INFORMATION
        )
        
        logger.info("Training completed successfully")
    
    def on_training_error(self, error_msg):
        """训练错误回调"""
        self.is_training = False
        self.btn_record.Enable(True)
        self.btn_inference.Enable(True)
        
        wx.MessageBox(f"训练失败: {error_msg}", "错误", wx.OK | wx.ICON_ERROR)
        logger.error(f"Training error: {error_msg}")
    
    def on_inference(self, event):
        """开始推理按钮回调"""
        if self.classifier is None:
            wx.MessageBox("分类器未初始化", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if len(self.classifier.gesture_map) == 0:
            wx.MessageBox("请先录制并训练手势", "提示", wx.OK | wx.ICON_WARNING)
            return
        
        if self.current_landmarks is None:
            wx.MessageBox("未检测到手势，请将手放在摄像头前", "提示", wx.OK | wx.ICON_WARNING)
            return
        
        try:
            # 归一化关键点
            normalized = self.classifier.normalize(self.current_landmarks, rotation_normalize=True)
            
            # 推理
            gesture_id, gesture_name, confidence = self.classifier.inference(normalized, confidence_threshold=0.7)
            
            # 显示结果
            if gesture_id != -1:
                # 识别成功
                dialog = wx.MessageDialog(
                    self,
                    f"手势编号: {gesture_id}\n"
                    f"手势名称: {gesture_name}\n"
                    f"置信度: {confidence:.4f}",
                    "识别成功!",
                    wx.OK | wx.ICON_INFORMATION
                )
                dialog.ShowModal()
                dialog.Destroy()
            else:
                # 识别失败
                dialog = wx.MessageDialog(
                    self,
                    "未识别到已知手势\n\n请确保:\n"
                    "- 手势清晰可见\n"
                    "- 已完成训练\n"
                    "- 手势与训练样本相似\n\n"
                    f"当前置信度: {confidence:.4f}",
                    "识别失败",
                    wx.OK | wx.ICON_WARNING
                )
                dialog.ShowModal()
                dialog.Destroy()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            wx.MessageBox(f"推理失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
    
    def on_clear_model(self, event):
        """清除模型按钮回调"""
        # 确认对话框
        response = wx.MessageBox(
            "确定要清除所有训练数据和模型吗？\n此操作不可恢复！",
            "确认清除",
            wx.YES_NO | wx.ICON_QUESTION
        )
        
        if response == wx.YES:
            try:
                # 重新初始化分类器
                self.classifier = GestureClassifier()
                self.training_samples = []
                self.current_gesture_name = ""
                self.is_recording = False
                
                # 更新状态
                self.update_status_display()
                
                wx.MessageBox("模型已清除", "成功", wx.OK | wx.ICON_INFORMATION)
                logger.info("Model cleared")
            except Exception as e:
                logger.error(f"Failed to clear model: {e}")
                wx.MessageBox(f"清除模型失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
    
    def on_save_model(self, event):
        """保存模型按钮回调"""
        if self.classifier is None or len(self.classifier.gesture_map) == 0:
            wx.MessageBox("没有可保存的模型，请先训练", "提示", wx.OK | wx.ICON_WARNING)
            return
        
        # 选择保存路径
        with wx.FileDialog(
            self,
            "保存模型",
            wildcard="PyTorch模型 (*.pth)|*.pth",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            model_path = fileDialog.GetPath()
            metadata_path = model_path.replace('.pth', '_metadata.json')
            
            try:
                self.classifier.save_model(model_path, metadata_path)
                wx.MessageBox(f"模型已保存到:\n{model_path}", "成功", wx.OK | wx.ICON_INFORMATION)
                logger.info(f"Model saved to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
                wx.MessageBox(f"保存模型失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
    
    def on_load_model(self, event):
        """加载模型按钮回调"""
        # 选择加载路径
        with wx.FileDialog(
            self,
            "加载模型",
            wildcard="PyTorch模型 (*.pth)|*.pth",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            model_path = fileDialog.GetPath()
            metadata_path = model_path.replace('.pth', '_metadata.json')
            
            try:
                self.classifier = GestureClassifier(model_path=model_path, weight_data=metadata_path)
                self.update_status_display()
                wx.MessageBox(f"模型已加载:\n{model_path}", "成功", wx.OK | wx.ICON_INFORMATION)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                wx.MessageBox(f"加载模型失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
    
    def update_status_display(self, custom_status: Optional[str] = None):
        """更新状态信息显示"""
        if custom_status:
            self.status_label.SetLabel(f"训练状态: {custom_status}")
        else:
            if self.is_training:
                self.status_label.SetLabel("训练状态: 训练中...")
            elif len(self.classifier.gesture_map) > 0:
                gesture_list = ", ".join(self.classifier.gesture_map.values())
                self.status_label.SetLabel(f"训练状态: 已训练 ({len(self.classifier.gesture_map)} 个手势)")
            else:
                self.status_label.SetLabel("训练状态: 未训练")
        
        self.sample_count_label.SetLabel(f"已录制样本数: {len(self.training_samples)}")
        
        if self.current_gesture_name:
            self.gesture_label.SetLabel(f"当前手势类别: {self.current_gesture_name}")
        elif len(self.classifier.gesture_map) > 0:
            gesture_list = ", ".join(self.classifier.gesture_map.values())
            self.gesture_label.SetLabel(f"已训练手势: {gesture_list}")
        else:
            self.gesture_label.SetLabel("当前手势类别: 无")
    
    def on_key_press(self, event):
        """键盘事件处理"""
        keycode = event.GetKeyCode()
        
        # Space = 推理
        if keycode == wx.WXK_SPACE:
            self.on_inference(None)
        # R = 录制
        elif keycode == ord('R') or keycode == ord('r'):
            self.on_record_train(None)
        # I = 推理
        elif keycode == ord('I') or keycode == ord('i'):
            self.on_inference(None)
        # C = 清除
        elif keycode == ord('C') or keycode == ord('c'):
            self.on_clear_model(None)
        else:
            event.Skip()
    
    def on_close(self, event):
        """窗口关闭事件，释放资源"""
        # 停止定时器
        if self.timer:
            self.timer.Stop()
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 释放 MediaPipe
        if self.hands:
            self.hands.close()
        
        logger.info("Application closed, resources released")
        self.Destroy()


def main():
    """主函数：启动应用"""
    app = wx.App()
    frame = GestureRecognitionApp()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()

