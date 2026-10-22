"""
TUNEX Utilities (Legacy compatibility bridge to ai_service & spotify_service)
"""
import logging
from .ai_service import (
    analyze_emotion,
    preprocess_roi as prepare,
    get_emotion_model,
    EMOTIONS,
    IMG_SIZE,
)
from .spotify_service import (
    get_recommendations_for_emotion,
    EMOTION_AUDIO_PROFILES,
)

logger = logging.getLogger(__name__)


def model():
    return get_emotion_model()


def egmap(emotion):
    """
    Returns mapped primary genre for emotion.
    """
    profile = EMOTION_AUDIO_PROFILES.get(emotion.lower() if emotion else "neutral", EMOTION_AUDIO_PROFILES["neutral"])
    return profile["genres"][0]


def predict_image(image_input, name_image="result.jpg"):
    """
    Legacy wrapper around analyze_emotion.
    """
    res = analyze_emotion(image_input)
    return res.get("annotated_image"), res.get("emotion")


def process_json(tracks):
    """
    Legacy parser for track tuples.
    """
    inde, artists_l, songs, urls, s_images = [], [], [], [], []
    for i, t in enumerate(tracks):
        inde.append(i)
        artists_l.append(t.get("artist", "Unknown"))
        songs.append(t.get("name", "Untitled"))
        urls.append(t.get("spotify_url", ""))
        s_images.append(t.get("album_art", ""))
    return zip(inde, artists_l, songs, urls, s_images)
