import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import math
import threading
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from streamlit_webrtc import WebRtcMode
import logging

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# WebRTC配置
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VolumeController:
    def __init__(self):
        CoInitialize()  # COM初始化
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            self.vol_range = self.volume.GetVolumeRange()
            self.min_vol, self.max_vol = self.vol_range[0], self.vol_range[1]
            logging.info("音量控制器初始化成功")
        except Exception as e:
            logging.error(f"音量控制器初始化失败: {str(e)}")
            CoUninitialize()
            raise

    def set_volume(self, vol_percent):
        try:
            vol_percent = max(0, min(100, vol_percent))
            vol_scalar = vol_percent / 100.0
            self.volume.SetMasterVolumeLevelScalar(vol_scalar, None)
            return vol_percent
        except Exception as e:
            logging.error(f"音量设置失败: {str(e)}")
            return -1

    def __del__(self):
        CoUninitialize()  # COM清理


class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.volume_controller = VolumeController()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,  # 简化模型提高性能
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.vol_history = []
        self.smooth_factor = 5
        self.min_dist = 30
        self.max_dist = 300

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))  # 降低分辨率提高性能

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 获取关键点坐标
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])

                    # 计算拇指和食指距离
                    if len(lm_list) >= 9:
                        x1, y1 = lm_list[4][1], lm_list[4][2]
                        x2, y2 = lm_list[8][1], lm_list[8][2]
                        length = math.hypot(x2 - x1, y2 - y1)

                        # 平滑处理
                        vol = np.interp(length, [self.min_dist, self.max_dist], [0, 100])
                        self.vol_history.append(vol)
                        if len(self.vol_history) > self.smooth_factor:
                            self.vol_history.pop(0)
                        smooth_vol = sum(self.vol_history) / len(self.vol_history)

                        # 异步设置音量
                        threading.Thread(
                            target=self.volume_controller.set_volume,
                            args=(smooth_vol,)
                        ).start()

                        # 绘制UI
                        self._draw_ui(img, smooth_vol, x1, y1, x2, y2)

            return img
        except Exception as e:
            logging.error(f"视频处理错误: {str(e)}")
            return frame

    def _draw_ui(self, img, vol_percent, x1, y1, x2, y2):
        # 绘制连接线和点
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)

        # 显示音量百分比
        cv2.putText(img, f'Volume: {int(vol_percent)}%', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制音量条
        bar_width, bar_height = 25, 150
        bar_x, bar_y = 20, 60
        vol_fill_height = int(np.interp(vol_percent, [0, 100], [0, bar_height]))

        cv2.rectangle(img, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (0, 255, 0), 1)
        cv2.rectangle(img, (bar_x, bar_y + bar_height - vol_fill_height),
                      (bar_x + bar_width, bar_y + bar_height),
                      (0, 255, 0), cv2.FILLED)


def main():
    st.title("手势音量控制")
    st.markdown("""
        使用拇指和食指的距离调整系统音量
        - 两指分开：音量增大
        - 两指靠近：音量减小
        - 两指接触：静音
    """)

    webrtc_ctx = webrtc_streamer(
        key="hand-gesture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HandGestureTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.button("停止应用"):
        st.experimental_rerun()


if __name__ == "__main__":
    main()