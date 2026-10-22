import base64
import json
import numpy as np
import cv2
from django.test import TestCase, Client
from django.urls import reverse

from .ai_service import analyze_emotion, EMOTIONS
from .spotify_service import EMOTION_AUDIO_PROFILES, get_recommendations_for_emotion


class EmotionModelTests(TestCase):
    def test_emotions_list(self):
        """Verify the 7 supported emotion categories exist."""
        self.assertEqual(len(EMOTIONS), 7)
        self.assertIn("happy", EMOTIONS)
        self.assertIn("sad", EMOTIONS)
        self.assertIn("angry", EMOTIONS)
        self.assertIn("neutral", EMOTIONS)

    def test_analyze_emotion_with_dummy_image(self):
        """Test the end-to-end emotion inference pipeline with a synthetic image."""
        dummy_img = np.zeros((150, 150, 3), dtype=np.uint8)
        # Draw a bright circle to simulate basic contrast
        cv2.circle(dummy_img, (75, 75), 40, (200, 200, 200), -1)

        result = analyze_emotion(dummy_img)
        self.assertTrue(result["success"])
        self.assertIn(result["emotion"], EMOTIONS)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(len(result["probabilities"]), 7)
        self.assertTrue(result["annotated_image"].startswith("data:image/jpeg;base64,"))

    def test_spotify_mood_profiles(self):
        """Verify audio attributes exist and are in valid ranges for all emotions."""
        for emo in EMOTIONS:
            self.assertIn(emo, EMOTION_AUDIO_PROFILES)
            profile = EMOTION_AUDIO_PROFILES[emo]
            self.assertGreaterEqual(profile["valence"], 0.0)
            self.assertLessEqual(profile["valence"], 1.0)
            self.assertGreaterEqual(profile["energy"], 0.0)
            self.assertLessEqual(profile["energy"], 1.0)
            self.assertTrue(len(profile["genres"]) > 0)


class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_page_status(self):
        """GET / should render the modern home page with HTTP 200."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "TUNEX")
        self.assertContains(response, "Listen to What Your")

    def test_live_scanner_page_status(self):
        """GET /live/ should render the live webcam scanner page with HTTP 200."""
        response = self.client.get(reverse('showlive'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Live Facial Emotion Scanner")
        self.assertContains(response, "webcamVideo")

    def test_upload_form_page_status(self):
        """GET /upload/ should render the static photo upload page with HTTP 200."""
        response = self.client.get(reverse('form'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload Portrait Photo")

    def test_recommendations_page_without_spotify(self):
        """GET /recommendations/ without Spotify connection should prompt for linking."""
        response = self.client.get(reverse('liveres'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Connect Your Spotify Account")

    def test_api_detect_emotion_endpoint(self):
        """POST /api/detect-emotion/ with base64 image should return JSON emotion predictions."""
        # Create a small 64x64 test image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        payload = {"image": b64_str}
        response = self.client.post(
            reverse('api_detect_emotion'),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn(data["emotion"], EMOTIONS)
        self.assertIn("probabilities", data)

    def test_api_recommendations_without_spotify_returns_401(self):
        """Calling /api/recommendations/ without linked Spotify account must return 401."""
        response = self.client.get(reverse('api_recommendations') + '?emotion=happy')
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "spotify_required")

    def test_spotify_disconnect_clears_session(self):
        """GET /spotify/disconnect/ should clear auth from session and redirect home."""
        session = self.client.session
        session['spotify_auth'] = {'access_token': 'fake_token'}
        session.save()

        response = self.client.get(reverse('spotify_disconnect'))
        self.assertRedirects(response, reverse('home'))
        self.assertNotIn('spotify_auth', self.client.session)
