import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
import numpy as np

st.title("DeepFace Web体験 (Experimental)")
st.write("※ 初回起動時はAIモデルのダウンロードで数分かかります。")
st.write("※ 動作が重い場合は、DeepFaceの処理時間を待っている状態です。")

# 映像処理を行うクラス
class EmotionProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_emotion = "Analyzing..." # 前回の結果を保存

    def transform(self, frame):
        # 1. 画像を取得
        img = frame.to_ndarray(format="bgr24")
        
        # 2. 軽量化のため、10フレームに1回だけAI分析する
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            try:
                # DeepFaceで感情分析
                # actions=['emotion']のみ指定して軽量化
                objs = DeepFace.analyze(
                    img_path=img, 
                    actions=['emotion'], 
                    enforce_detection=False, # 顔が見つからなくてもエラーにしない
                    detector_backend='opencv' # 軽量な検出器を使用
                )
                self.last_emotion = objs[0]['dominant_emotion']
            except Exception as e:
                # エラー（顔が見つからないなど）の時は無視
                pass

        # 3. 画面に文字を書く
        # 顔認識の四角描写は重くなるので省略し、文字だけ表示します
        cv2.putText(
            img, 
            f"Emotion: {self.last_emotion}", 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        return img

# メイン処理：Webカメラの起動ボタン
webrtc_streamer(
    key="example", 
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}, # マイクはオフ
    rtc_configuration={  # Streamlit Cloudでカメラを安定させるおまじない
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
