import os
import cv2
import base64
import hashlib
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import onnxruntime as ort
from openai import OpenAI
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None
    find_dotenv = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Emotion Voice Assistant",
    page_icon="🙂",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# -----------------------------
# Robust .env loading
# -----------------------------
def manual_load_env_file(env_path: Path):
    if not env_path.exists():
        return False

    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except Exception:
        return False

    loaded = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and value and not os.getenv(key):
            os.environ[key] = value
            loaded = True
    return loaded


ENV_PATHS_CHECKED = [
    BASE_DIR / ".env",
    Path.cwd() / ".env",
]

loaded_env_paths = []

if load_dotenv is not None:
    for env_path in ENV_PATHS_CHECKED:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            loaded_env_paths.append(str(env_path))
    if not loaded_env_paths and find_dotenv is not None:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=True)
            loaded_env_paths.append(found)

for env_path in ENV_PATHS_CHECKED:
    if env_path.exists():
        manual_load_env_file(env_path)
        if str(env_path) not in loaded_env_paths:
            loaded_env_paths.append(str(env_path))

# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = "emotion-ferplus-8.onnx"
MAX_TURNS = 10

CLASS_NAMES = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

EMOTION_MAP = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprised",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "upset",
    "fear": "worried",
    "contempt": "neutral",
}

OPENING_BY_EMOTION = {
    "sad": "You seem a little sad. Are you okay? Do you want to talk?",
    "angry": "You seem a little upset. Are you okay? Do you want to talk?",
    "neutral": "You seem a little quiet. Are you okay? Do you want to talk?",
    "worried": "You seem a little worried. Are you okay? Do you want to talk?",
    "upset": "You seem a little upset. Are you okay? Do you want to talk?",
    "happy": "You seem cheerful today. Do you still want to talk for a moment?",
    "surprised": "You seem a little surprised. Are you okay? Do you want to talk?",
}

MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
}

# -----------------------------
# Mobile-friendly styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        max-width: 760px;
    }
    h1, h2, h3 {
        line-height: 1.15;
    }
    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        min-height: 3.1rem;
        border-radius: 14px;
        font-size: 1rem;
        font-weight: 600;
    }
    div[data-testid="stAudioInput"] button {
        min-height: 3.2rem !important;
        border-radius: 14px !important;
    }
    .mobile-card {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-bottom: 0.6rem;
    }
    .assistant-bubble {
        background: rgba(70, 130, 180, 0.08);
    }
    .user-bubble {
        background: rgba(100, 149, 237, 0.05);
    }
    .sticky-help {
        position: sticky;
        bottom: 0.5rem;
        z-index: 999;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.75rem;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        p, li, label {
            font-size: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------
def init_state():
    defaults = {
        "detected_emotion": None,
        "face_key": None,
        "conversation_started": False,
        "conversation_finished": False,
        "turn_count": 0,
        "chat": [],
        "last_spoken_text": None,
        "pending_opening": False,
        "status_message": "",
        "last_audio_signature": None,
        "typed_reply_input": "",
        "debug_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# -----------------------------
# OpenAI client from .env / secrets
# -----------------------------
api_key = (
    os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("API_KEY", "").strip()
    or os.getenv("openai_api_key", "").strip()
)

try:
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
except Exception:
    pass

client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_emotion_model():
    model_path = BASE_DIR / MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found in the app folder. "
            "Download it and place it next to app.py."
        )
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade face detector.")
    return detector


try:
    model = load_emotion_model()
    face_cascade = load_face_detector()
except Exception as exc:
    st.error(f"Startup error: {exc}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, np.newaxis, :, :]


def detect_largest_face(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict_emotion(image_bgr: np.ndarray):
    face_box = detect_largest_face(image_bgr)
    if face_box is None:
        return None, None, None

    x, y, w, h = face_box
    face = image_bgr[y:y + h, x:x + w]
    input_tensor = preprocess_face(face)
    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: input_tensor})[0]

    scores = output[0]
    probs = softmax(scores)
    pred_index = int(np.argmax(probs))
    raw_emotion = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    emotion = EMOTION_MAP.get(raw_emotion, raw_emotion)
    return emotion, confidence, (x, y, w, h)


def draw_face_box(image_bgr: np.ndarray, box, emotion: str, confidence: float):
    x, y, w, h = box
    out = image_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{emotion} ({confidence:.2f})"
    cv2.putText(
        out,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    return out


def autoplay_tts(text: str):
    if not text or st.session_state.get("last_spoken_text") == text:
        return

    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            with open(temp_audio.name, "rb") as f:
                audio_bytes = f.read()

        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay playsinline>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/mp3")
        st.session_state["last_spoken_text"] = text
    except Exception as exc:
        st.warning(f"Assistant voice failed: {exc}")


def fallback_reply(user_text: str, emotion: str, turn_num: int) -> str:
    emotion = emotion or "sad"

    if turn_num >= MAX_TURNS:
        return (
            "Thank you for talking with me. I'm really glad you opened up. "
            "You are doing better than you think, and you have the strength to handle this. "
            "Keep going, one step at a time. Take care, bye."
        )

    if emotion in {"sad", "worried", "upset"}:
        lines = [
            "I'm here with you. Do you want to tell me a little more?",
            "That sounds heavy. What part is feeling hardest right now?",
            "It makes sense to feel this way. What would help you feel a little lighter today?",
            "Thank you for sharing that with me. Is there one small next step you can take today?",
            "You're doing a brave thing by talking about it. What has helped you before?",
            "Would it help to slow down and take one thing at a time?",
            "You matter, and your feelings matter too. What would support look like right now?",
            "I'm glad you're still talking with me. What is one good thing you can hold on to today?",
            "You're getting through this moment. What is one small action you can take after this?",
        ]
    elif emotion == "angry":
        lines = [
            "I hear you. Do you want to tell me what happened?",
            "That sounds really frustrating. What upset you the most?",
            "It's okay to pause for a second. What do you need right now?",
            "You're doing a good job talking it through. What would help calm things down a little?",
            "Your feelings are valid. What part feels most unfair?",
            "Let's slow it down together. What would make this easier to handle?",
            "You have more control than this moment makes it seem. What can you do next?",
            "You are handling this better than you think. What would help you reset?",
            "You're still showing up, and that matters. What do you want tomorrow to feel like?",
        ]
    else:
        lines = [
            "I'm listening. What's on your mind?",
            "Thanks for sharing that. Can you tell me a little more?",
            "I hear you. What feels most important right now?",
            "That makes sense. What would help most today?",
            "You're doing well opening up. What do you want to focus on next?",
            "I'm with you. What is one thing you wish someone understood?",
            "Thank you for staying with this conversation. What would feel helpful right now?",
            "You're making progress by speaking honestly. What do you want more of in your day?",
            "You have a lot of strength in you. What's one step you feel ready for?",
        ]

    return lines[min(turn_num - 1, len(lines) - 1)]


def get_ai_reply(user_text: str, emotion: str, turn_num: int) -> str:
    if turn_num >= MAX_TURNS:
        return (
            "Thank you for talking with me. I'm really glad you opened up. "
            "You are doing better than you think, and you have the strength to handle this. "
            "Keep going, one step at a time. Take care, bye."
        )

    if client is None:
        return fallback_reply(user_text, emotion, turn_num)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, calm, encouraging emotional support conversation partner. "
                "Keep responses short and natural for speech. "
                "Respond in 1 to 3 sentences. Ask only one gentle follow-up question. "
                "Do not mention AI, cameras, emotion detection, or diagnosis. "
                "On the tenth assistant turn, end with a positive affirmation and 'Take care, bye.'"
            ),
        }
    ]

    for item in st.session_state["chat"]:
        messages.append({"role": item["role"], "content": item["text"]})

    if emotion:
        messages.append({
            "role": "system",
            "content": f"The user initially appeared {emotion}. Be gentle and supportive.",
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        st.session_state["status_message"] = f"AI fallback used: {exc}"
        return fallback_reply(user_text, emotion, turn_num)


def get_audio_filename_and_mime(audio_value):
    original_name = getattr(audio_value, "name", "") or ""
    mime_type = getattr(audio_value, "type", "") or "application/octet-stream"

    suffix = Path(original_name).suffix.lower()
    if not suffix:
        suffix = MIME_TO_EXT.get(mime_type, ".wav")

    filename = f"voice_reply{suffix}"
    return filename, mime_type


def transcribe_audio_file(audio_bytes: bytes, filename: str, mime_type: str):
    if not audio_bytes:
        return "", "No audio data was captured."

    if client is None:
        return "", (
            "API key not found. Put OPENAI_API_KEY in a .env file next to app.py, "
            "then restart Streamlit."
        )

    try:
        buffer = BytesIO(audio_bytes)
        buffer.name = filename

        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=buffer,
        )

        text = getattr(transcript, "text", "").strip()
        if not text:
            return "", "The audio was received, but no speech was detected."
        return text, ""
    except Exception as exc:
        return "", f"Transcription failed: {exc}"


def build_pdf(chat_items):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Emotion Voice Assistant Conversation")
    y -= 30
    c.setFont("Helvetica", 11)

    for item in chat_items:
        speaker = "Assistant" if item["role"] == "assistant" else "User"
        text = f"{speaker}: {item['text']}"
        wrapped = wrap_text(text, 90)
        for line in wrapped:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            c.drawString(50, y, line)
            y -= 16
        y -= 6

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def wrap_text(text: str, max_len: int):
    words = text.split()
    lines = []
    current = []
    length = 0
    for word in words:
        if length + len(word) + len(current) <= max_len:
            current.append(word)
            length += len(word)
        else:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
    if current:
        lines.append(" ".join(current))
    return lines


def reset_for_new_face(face_key: str, emotion: str):
    st.session_state["face_key"] = face_key
    st.session_state["detected_emotion"] = emotion
    st.session_state["conversation_started"] = True
    st.session_state["conversation_finished"] = False
    st.session_state["turn_count"] = 0
    st.session_state["chat"] = []
    st.session_state["last_spoken_text"] = None
    st.session_state["pending_opening"] = True
    st.session_state["status_message"] = ""
    st.session_state["last_audio_signature"] = None
    st.session_state["typed_reply_input"] = ""
    st.session_state["debug_message"] = ""


def process_user_message(user_text: str):
    user_text = (user_text or "").strip()
    if not user_text or st.session_state["conversation_finished"]:
        return

    st.session_state["chat"].append({"role": "user", "text": user_text})
    st.session_state["turn_count"] += 1

    reply = get_ai_reply(
        user_text,
        st.session_state["detected_emotion"],
        st.session_state["turn_count"],
    )
    st.session_state["chat"].append({"role": "assistant", "text": reply})

    if st.session_state["turn_count"] >= MAX_TURNS:
        st.session_state["conversation_finished"] = True


def submit_typed_reply():
    text = st.session_state.get("typed_reply_input", "").strip()
    if text:
        process_user_message(text)
        st.session_state["typed_reply_input"] = ""


def audio_signature(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()

# -----------------------------
# Header
# -----------------------------
st.title("🙂 Emotion Voice Assistant")
st.caption("Take a photo, then tap record, speak, stop, and get an immediate reply.")

if api_key:
    st.success("API key detected.")
else:
    st.warning(
        "API key not detected. The app checked .env next to app.py, the current folder, and Streamlit secrets."
    )

with st.expander("Diagnostics", expanded=not bool(api_key)):
    st.write("App folder:", str(BASE_DIR))
    st.write("Current working folder:", str(Path.cwd()))
    st.write(".env paths checked:", ENV_PATHS_CHECKED)
    st.write(".env loaded from:", loaded_env_paths if loaded_env_paths else "none")
    st.write("Has OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("Has API_KEY:", bool(os.getenv("API_KEY")))
    st.write("Has openai_api_key:", bool(os.getenv("openai_api_key")))
    st.write("python-dotenv installed:", load_dotenv is not None)
    if st.session_state.get("debug_message"):
        st.write("Last audio debug:", st.session_state["debug_message"])

# -----------------------------
# Image input
# -----------------------------
st.subheader("1) Take or upload a face photo")

camera_tab, upload_tab = st.tabs(["📷 Camera", "🖼️ Upload"])
image_bgr = None

with camera_tab:
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        pil_img = Image.open(camera_photo).convert("RGB")
        image_np = np.array(pil_img)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

with upload_tab:
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and image_bgr is None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_img)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

if image_bgr is not None:
    emotion, confidence, face_box = predict_emotion(image_bgr)
    if emotion is None:
        st.warning("No face detected. Please try a clearer photo with good lighting.")
    else:
        boxed = draw_face_box(image_bgr, face_box, emotion, confidence)
        st.image(
            cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
            caption="Detected face and emotion",
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Emotion", emotion)
        c2.metric("Confidence", f"{confidence:.0%}")

        face_key = f"{emotion}_{round(confidence, 3)}_{face_box}"
        if st.session_state["face_key"] != face_key:
            reset_for_new_face(face_key, emotion)

# -----------------------------
# Conversation
# -----------------------------
if st.session_state["conversation_started"] and st.session_state["detected_emotion"]:
    opening = OPENING_BY_EMOTION.get(
        st.session_state["detected_emotion"],
        "You seem a little sad. Are you okay? Do you want to talk?",
    )

    if st.session_state["pending_opening"]:
        st.session_state["chat"].append({"role": "assistant", "text": opening})
        st.session_state["pending_opening"] = False

    st.subheader("2) Talk")

    if st.session_state["chat"]:
        autoplay_tts(st.session_state["chat"][-1]["text"])

    for item in st.session_state["chat"]:
        who = "🤖 Assistant" if item["role"] == "assistant" else "🧑 You"
        bubble_class = "assistant-bubble" if item["role"] == "assistant" else "user-bubble"
        st.markdown(
            f'<div class="mobile-card {bubble_class}"><strong>{who}</strong><br>{item["text"]}</div>',
            unsafe_allow_html=True,
        )

    turns_used = st.session_state["turn_count"]
    turns_left = max(0, MAX_TURNS - turns_used)
    m1, m2 = st.columns(2)
    m1.metric("Turns used", f"{turns_used}/{MAX_TURNS}")
    m2.metric("Turns left", turns_left)

    if not st.session_state["conversation_finished"]:
        st.markdown(
            """
            <div class="sticky-help">
                <strong>Voice mode</strong><br>
                Tap record below, speak, then stop. As soon as recording is available, the app transcribes and responds.
            </div>
            """,
            unsafe_allow_html=True,
        )

        audio_value = st.audio_input("🎙️ Tap to record your reply")
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            signature = audio_signature(audio_bytes)
            filename, mime_type = get_audio_filename_and_mime(audio_value)

            st.session_state["debug_message"] = (
                f"bytes={len(audio_bytes)}, name={getattr(audio_value, 'name', '')}, "
                f"type={getattr(audio_value, 'type', '')}, used_filename={filename}, mime={mime_type}"
            )

            if signature != st.session_state.get("last_audio_signature"):
                st.session_state["last_audio_signature"] = signature
                st.session_state["status_message"] = "Voice received. Transcribing now..."

                user_text, error_message = transcribe_audio_file(
                    audio_bytes,
                    filename,
                    mime_type,
                )

                if error_message:
                    st.session_state["status_message"] = error_message
                else:
                    st.session_state["status_message"] = f'You said: "{user_text}"'
                    process_user_message(user_text)
                    st.rerun()

        st.text_input(
            "Or type your reply and press Enter",
            key="typed_reply_input",
            placeholder="Type here and press Enter",
            on_change=submit_typed_reply,
        )

        if st.button("Send typed reply"):
            submit_typed_reply()
            st.rerun()

        if st.session_state.get("status_message"):
            st.caption(st.session_state["status_message"])

    if st.session_state["conversation_finished"]:
        st.success(
            "Conversation complete. Positive note: You are stronger than you think, "
            "and every small step forward matters."
        )

    pdf_bytes = build_pdf(st.session_state["chat"])
    st.download_button(
        "Download conversation as PDF",
        data=pdf_bytes,
        file_name="emotion_conversation.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# -----------------------------
# Reset
# -----------------------------
if st.button("Start over"):
    for key in [
        "detected_emotion",
        "face_key",
        "conversation_started",
        "conversation_finished",
        "turn_count",
        "chat",
        "last_spoken_text",
        "pending_opening",
        "status_message",
        "last_audio_signature",
        "typed_reply_input",
        "debug_message",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
