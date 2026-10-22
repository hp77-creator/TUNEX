import base64
import logging
import random
import secrets
import time
import urllib.parse
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

# Curated mood mapping from 7 emotions to audio profiles & fallback search queries
EMOTION_AUDIO_PROFILES = {
    "happy": {
        "valence": 0.85,
        "energy": 0.80,
        "genres": ["pop", "dance", "disco", "happy"],
        "queries": ["happy upbeat hits", "feel good pop", "summer dance anthems"],
        "color": "#10B981", # Emerald
        "tagline": "Uplifting, high-energy anthems to match your smile",
    },
    "sad": {
        "valence": 0.20,
        "energy": 0.30,
        "genres": ["sad", "acoustic", "piano", "indie"],
        "queries": ["melancholy indie acoustic", "sad piano ballad", "heartbreak songs"],
        "color": "#3B82F6", # Azure
        "tagline": "Gentle, soul-soothing melodies for contemplative moments",
    },
    "angry": {
        "valence": 0.30,
        "energy": 0.90,
        "genres": ["rock", "metal", "punk", "alt-rock"],
        "queries": ["heavy rock workout", "raw punk rage", "hard metal"],
        "color": "#EF4444", # Crimson
        "tagline": "High-octane rock and metal to channel that fiery intensity",
    },
    "surprised": {
        "valence": 0.75,
        "energy": 0.85,
        "genres": ["electronic", "dance", "indie-pop", "synth-pop"],
        "queries": ["energetic electronic", "synthwave thrill", "hyperpop surprises"],
        "color": "#F59E0B", # Amber
        "tagline": "Electrifying, unexpected beats to match your excitement",
    },
    "afraid": {
        "valence": 0.35,
        "energy": 0.45,
        "genres": ["ambient", "soundtracks", "chill", "classical"],
        "queries": ["calming ambient anxiety relief", "peaceful piano chill", "lo-fi calm"],
        "color": "#8B5CF6", # Purple
        "tagline": "Calming, grounding ambient tracks to ease tension and breathe",
    },
    "disgust": {
        "valence": 0.40,
        "energy": 0.65,
        "genres": ["grunge", "alt-rock", "blues", "funk"],
        "queries": ["90s alternative grunge", "funky blues groove", "raw indie"],
        "color": "#14B8A6", # Teal
        "tagline": "Gritty, offbeat alternative grooves to reset your vibe",
    },
    "neutral": {
        "valence": 0.50,
        "energy": 0.50,
        "genres": ["chill", "ambient", "acoustic", "lo-fi"],
        "queries": ["lo-fi study beats", "coffeehouse acoustic chill", "chillout vibes"],
        "color": "#6B7280", # Slate
        "tagline": "Balanced, mellow tunes to keep your flow steady and relaxed",
    },
}


def _get_auth_headers():
    client_id = getattr(settings, 'SPOTIFY_CLIENT_ID', '').strip()
    client_secret = getattr(settings, 'SPOTIFY_CLIENT_SECRET', '').strip()
    if not client_id or not client_secret:
        return None
    creds = f"{client_id}:{client_secret}".encode("utf-8")
    b64_creds = base64.b64encode(creds).decode("utf-8")
    return {"Authorization": f"Basic {b64_creds}"}


def get_auth_url(request):
    """
    Generates Spotify OAuth2 Authorization URL with PKCE state protection.
    """
    client_id = getattr(settings, 'SPOTIFY_CLIENT_ID', '').strip()
    redirect_uri = getattr(settings, 'SPOTIFY_REDIRECT_URI', '').strip()
    scope = getattr(settings, 'SPOTIFY_SCOPE', '')

    if not client_id:
        return None

    state = secrets.token_urlsafe(16)
    request.session['spotify_oauth_state'] = state

    params = {
        'client_id': client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
        'scope': scope,
        'state': state,
        'show_dialog': 'true',
    }
    return f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"


def handle_callback(request, code, state):
    """
    Exchanges Spotify authorization code for access and refresh tokens.
    """
    saved_state = request.session.get('spotify_oauth_state')
    if not saved_state or saved_state != state:
        logger.error("State mismatch in Spotify OAuth callback.")
        return False, "OAuth state mismatch. Security verification failed."

    headers = _get_auth_headers()
    if not headers:
        return False, "Spotify Client ID or Secret is not configured."

    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': settings.SPOTIFY_REDIRECT_URI,
    }

    try:
        response = requests.post(SPOTIFY_TOKEN_URL, data=data, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.error(f"Failed to fetch tokens from Spotify: {response.status_code} {response.text}")
            return False, f"Spotify authorization error: {response.text}"

        token_data = response.json()
        access_token = token_data.get('access_token')
        refresh_token = token_data.get('refresh_token')
        expires_in = token_data.get('expires_in', 3600)
        expires_at = time.time() + expires_in - 60

        # Fetch user profile
        user_profile = get_user_profile(access_token)

        request.session['spotify_auth'] = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': expires_at,
            'user': user_profile,
        }
        request.session.modified = True
        return True, user_profile
    except Exception as e:
        logger.exception("Exception during Spotify token exchange")
        return False, str(e)


def get_user_profile(access_token):
    """
    Fetches user's Spotify profile details.
    """
    try:
        res = requests.get(
            f"{SPOTIFY_API_BASE}/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=8,
        )
        if res.status_code == 200:
            data = res.json()
            images = data.get('images', [])
            avatar = images[0].get('url') if images else None
            return {
                'id': data.get('id'),
                'display_name': data.get('display_name') or data.get('id'),
                'avatar': avatar,
                'email': data.get('email'),
                'product': data.get('product', 'free'),
                'profile_url': data.get('external_urls', {}).get('spotify'),
            }
    except Exception as e:
        logger.warning(f"Failed to fetch user profile: {e}")
    return {'id': 'user', 'display_name': 'Spotify User', 'avatar': None, 'product': 'free'}


def get_active_token(request):
    """
    Returns a valid access token from session, refreshing if expired.
    Returns None if user is not authenticated.
    """
    auth = request.session.get('spotify_auth')
    if not auth or 'access_token' not in auth:
        return None

    now = time.time()
    if now < auth.get('expires_at', 0):
        return auth['access_token']

    # Token expired, refresh it
    refresh_token = auth.get('refresh_token')
    if not refresh_token:
        request.session.pop('spotify_auth', None)
        return None

    new_token = refresh_access_token(request, refresh_token)
    return new_token


def refresh_access_token(request, refresh_token):
    """
    Refreshes an expired access token using the stored refresh_token.
    """
    headers = _get_auth_headers()
    if not headers:
        return None

    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
    }

    try:
        response = requests.post(SPOTIFY_TOKEN_URL, data=data, headers=headers, timeout=10)
        if response.status_code == 200:
            token_data = response.json()
            auth = request.session.get('spotify_auth', {})
            auth['access_token'] = token_data.get('access_token')
            auth['expires_at'] = time.time() + token_data.get('expires_in', 3600) - 60
            if 'refresh_token' in token_data:
                auth['refresh_token'] = token_data['refresh_token']
            request.session['spotify_auth'] = auth
            request.session.modified = True
            return auth['access_token']
        else:
            logger.warning(f"Refresh token error: {response.status_code} {response.text}")
            request.session.pop('spotify_auth', None)
            return None
    except Exception as e:
        logger.exception("Failed to refresh Spotify token")
        return None


def disconnect_spotify(request):
    """
    Clears Spotify credentials from user session.
    """
    request.session.pop('spotify_auth', None)
    request.session.pop('spotify_oauth_state', None)
    request.session.modified = True


def get_recommendations_for_emotion(access_token, emotion, limit=10):
    """
    Fetches personalized song recommendations based on detected emotion.
    Uses Spotify Web API with intelligent fallback to Search if Recommendations endpoint is restricted.
    """
    emotion_key = emotion.lower().strip() if emotion else "neutral"
    profile = EMOTION_AUDIO_PROFILES.get(emotion_key, EMOTION_AUDIO_PROFILES["neutral"])
    headers = {"Authorization": f"Bearer {access_token}"}

    # Step 1: Try recommendations endpoint
    rec_url = f"{SPOTIFY_API_BASE}/recommendations"
    genres = profile["genres"]
    seed_genres = ",".join(genres[:2])

    params = {
        "limit": limit,
        "seed_genres": seed_genres,
        "target_valence": profile["valence"],
        "target_energy": profile["energy"],
    }

    tracks = []
    try:
        res = requests.get(rec_url, headers=headers, params=params, timeout=8)
        if res.status_code == 200:
            data = res.json()
            raw_tracks = data.get("tracks", [])
            tracks = _parse_tracks(raw_tracks, emotion_key, profile)
    except Exception as e:
        logger.warning(f"Error fetching from recommendations endpoint: {e}")

    # Step 2: Fallback to Search API (Spotify 2024+ restricted recommendations for unverified apps)
    if not tracks:
        logger.info(f"Using Search API fallback for emotion '{emotion_key}'")
        query = random.choice(profile["queries"])
        search_url = f"{SPOTIFY_API_BASE}/search"
        search_params = {
            "q": query,
            "type": "track",
            "limit": limit,
            "market": "from_token",
        }
        try:
            res = requests.get(search_url, headers=headers, params=search_params, timeout=8)
            if res.status_code == 200:
                data = res.json()
                raw_tracks = data.get("tracks", {}).get("items", [])
                tracks = _parse_tracks(raw_tracks, emotion_key, profile)
        except Exception as e:
            logger.error(f"Search API fallback error: {e}")

    return {
        "emotion": emotion_key,
        "profile": profile,
        "tracks": tracks,
    }


def _parse_tracks(raw_tracks, emotion_key, profile):
    parsed = []
    for t in raw_tracks:
        if not t or not t.get("id"):
            continue
        track_id = t["id"]
        artists = ", ".join([a.get("name", "Unknown") for a in t.get("artists", [])])
        album = t.get("album", {})
        images = album.get("images", [])
        album_art = images[0].get("url") if images else "/static/images/default_album.png"

        parsed.append({
            "id": track_id,
            "name": t.get("name", "Untitled"),
            "artist": artists,
            "album": album.get("name", "Unknown Album"),
            "album_art": album_art,
            "spotify_url": t.get("external_urls", {}).get("spotify", f"https://open.spotify.com/track/{track_id}"),
            "uri": t.get("uri", f"spotify:track:{track_id}"),
            "embed_url": f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0",
            "emotion": emotion_key,
            "color": profile["color"],
        })
    return parsed
