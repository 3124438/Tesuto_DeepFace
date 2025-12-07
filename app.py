import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
import numpy as np

# ページ設定
st.set_page_config(page_title="DeepFace Emoji & Text", layout="centered")

st.title("DeepFace 完成版 (Tesuto)")
st.write("顔文字とローマ字で、今の表情を表示します！")

# --- 辞書定義： (顔文字, ローマ字) のセット ---
EMOTION_DATA = {
    "neutral":  (" . _ . ",     "MAGAO"),      # 真顔
    "happy":    ("^ v ^",       "URESHII"),    # 嬉しい
    "surprise": ("O . O !",     "BIKKURI"),    # びっくり
    "sad":      ("T . T",       "KANASHII"),   # 悲しい
    "angry":    ("> _ < #",     "OKOTTERU"),   # 怒り
    "fear":     ("; O O ;",     "KOWAI"),      # 怖い
    "disgust":  ("...",         "IYA"),        # 嫌悪
}

# 映像処理クラス
class EmotionProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_emotion_key = "neutral"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 10フレームに1回分析（軽量化）
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            try:
                objs = DeepFace.analyze(
                    img_path=img, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                self.last_emotion_key = objs[0]['dominant_emotion']
            except Exception:
                pass

        # 辞書からデータを取得
        # データがない場合は "?" を表示
        data = EMOTION_DATA.get(self.last_emotion_key, ("?", "?"))
        kaomoji = data[0] # 顔文字
        romaji  = data[1] # ローマ字

        # --- 描画パート ---
        
        # 1行目：顔文字（大きく）
        # 黒いフチ
        cv2.putText(img, kaomoji, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    2.0, (0, 0, 0), 8)
        # 白い文字
        cv2.putText(img, kaomoji, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    2.0, (255, 255, 255), 3)

        # 2行目：ローマ字（少し小さく、下に配置）
        # 黒いフチ
        cv2.putText(img, romaji, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 0, 0), 6)
        # 黄色の文字（目立つように！）
        cv2.putText(img, romaji, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 255), 2)

        return img

# メイン処理
webrtc_streamer(
    key="deepface-final", 
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
