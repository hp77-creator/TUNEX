import base64
import logging
import os
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
        "color": "#94A3B8", # Slate 400
        "tagline": "Balanced, mellow tunes to keep your flow steady and relaxed",
    },
}

# Rich curated demo tracks for instant evaluation without requiring API credentials
DEMO_TRACKS = {
    "happy": [
        {"id": "0VjIjW4GlULA7GV4y9l4pT", "name": "Happy", "artist": "Pharrell Williams", "album": "G I R L", "album_art": "https://i.scdn.co/image/ab67616d0000b2734105264b73b5c6e8e8ffc013"},
        {"id": "3n3Ppam7vgaVa1iaRUc9Lp", "name": "Mr. Blue Sky", "artist": "Electric Light Orchestra", "album": "Out of the Blue", "album_art": "https://i.scdn.co/image/ab67616d0000b27376c757c3d9b04f6e3c0b1156"},
        {"id": "2b8vdOqOEALNAePw0aCy1I", "name": "Good as Hell", "artist": "Lizzo", "album": "Cuz I Love You", "album_art": "https://i.scdn.co/image/ab67616d0000b273c5649add07ed3720be9d5526"},
        {"id": "50kpGaPAhYJ3sGmk6vplg0", "name": "Walking On Sunshine", "artist": "Katrina & The Waves", "album": "Walking On Sunshine", "album_art": "https://i.scdn.co/image/ab67616d0000b273a3a4c0eb322961d15582f348"},
        {"id": "1mea3bSkSGXuIRvnydlB5b", "name": "Viva La Vida", "artist": "Coldplay", "album": "Death and All His Friends", "album_art": "https://i.scdn.co/image/ab67616d0000b273e2e352d89826aef6dbd5ff8f"},
        {"id": "60nZcImufyMA1MKQY3dcCH", "name": "Happy Now", "artist": "Kygo, Sandro Cavazza", "album": "Happy Now", "album_art": "https://i.scdn.co/image/ab67616d0000b2734c38d2b9015c7e14fb1552a4"},
    ],
    "sad": [
        {"id": "4kflIG9jdZJW4ot2ioixTB", "name": "Someone Like You", "artist": "Adele", "album": "21", "album_art": "https://i.scdn.co/image/ab67616d0000b2732118bf91167c170669c45e6d"},
        {"id": "7LVHVU3tWfcxj5aiP2VWZs", "name": "Fix You", "artist": "Coldplay", "album": "X&Y", "album_art": "https://i.scdn.co/image/ab67616d0000b2738605c48b28cf9797d022b724"},
        {"id": "3U4isOIWM3VvDubwSI3y7a", "name": "All of Me", "artist": "John Legend", "album": "Love in the Future", "album_art": "https://i.scdn.co/image/ab67616d0000b2731c3aa71b000be0876793ceb1"},
        {"id": "75JFxkI2RXiU7L9VXzMkle", "name": "The Scientist", "artist": "Coldplay", "album": "A Rush of Blood to the Head", "album_art": "https://i.scdn.co/image/ab67616d0000b27390820409a805c31766b826de"},
        {"id": "0GjEhRrFj17V01H3s3vG7h", "name": "Creep", "artist": "Radiohead", "album": "Pablo Honey", "album_art": "https://i.scdn.co/image/ab67616d0000b27328198f6d89269d05be159518"},
        {"id": "3UmaczJpikHgJFy3GjydGq", "name": "Stay With Me", "artist": "Sam Smith", "album": "In The Lonely Hour", "album_art": "https://i.scdn.co/image/ab67616d0000b273ec6f8f533bf6f3a7638ad3a6"},
    ],
    "angry": [
        {"id": "2bLq0q9Ki5llIxdq07OFIO", "name": "Enter Sandman", "artist": "Metallica", "album": "Metallica", "album_art": "https://i.scdn.co/image/ab67616d0000b2730a846c26881c9a44fc476be1"},
        {"id": "254bXAAtmQIijb6nhmwCpV", "name": "Chop Suey!", "artist": "System Of A Down", "album": "Toxicity", "album_art": "https://i.scdn.co/image/ab67616d0000b27339d91f46755fb34ea07914e6"},
        {"id": "1BxfuPKGuaTgP7aM0fbdvf", "name": "Killing In The Name", "artist": "Rage Against The Machine", "album": "Rage Against The Machine", "album_art": "https://i.scdn.co/image/ab67616d0000b2734e565759ff856c8021c33be8"},
        {"id": "7ouMYWpwJ422jRcDASZB7P", "name": "Knights of Cydonia", "artist": "Muse", "album": "Black Holes and Revelations", "album_art": "https://i.scdn.co/image/ab67616d0000b27328933b808bfb4cbbd0385400"},
        {"id": "08mG3Y1vljYA6bvDt4Wqkj", "name": "Back In Black", "artist": "AC/DC", "album": "Back In Black", "album_art": "https://i.scdn.co/image/ab67616d0000b2730b51f8d91f3a21e8426361ae"},
    ],
    "surprised": [
        {"id": "4uLU6hMCjMI75M1A2tKUQC", "name": "Never Gonna Give You Up", "artist": "Rick Astley", "album": "Whenever You Need Somebody", "album_art": "https://i.scdn.co/image/ab67616d0000b2735755e164993798e0c9ef7d7a"},
        {"id": "2WfaOiMkCvy7Z5vo2vrF1o", "name": "Feel Good Inc.", "artist": "Gorillaz", "album": "Demon Days", "album_art": "https://i.scdn.co/image/ab67616d0000b273dfddf245e0ff956272fb5152"},
        {"id": "5CQ30WqJwcep0pYcV4AMNc", "name": "Starlight", "artist": "Muse", "album": "Black Holes and Revelations", "album_art": "https://i.scdn.co/image/ab67616d0000b27328933b808bfb4cbbd0385400"},
        {"id": "7qiZfU4dY1lWllzX7mPBI3", "name": "Shape of You", "artist": "Ed Sheeran", "album": "÷ (Divide)", "album_art": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96"},
    ],
    "afraid": [
        {"id": "3AJwUDP919kvQ9QcozQPxg", "name": "Weightless", "artist": "Marconi Union", "album": "Weightless", "album_art": "https://i.scdn.co/image/ab67616d0000b273b4007b065a71398e03e42f53"},
        {"id": "1q8XZQnQ08Q4t3a992zP2K", "name": "Experience", "artist": "Ludovico Einaudi", "album": "In A Time Lapse", "album_art": "https://i.scdn.co/image/ab67616d0000b27376c757c3d9b04f6e3c0b1156"},
        {"id": "3BQHpFgAp4l80e1XslIjNI", "name": "Spiegel im Spiegel", "artist": "Arvo Pärt", "album": "Alina", "album_art": "https://i.scdn.co/image/ab67616d0000b273c5649add07ed3720be9d5526"},
    ],
    "disgust": [
        {"id": "1hnWB2t6o2eRkP398x1XbM", "name": "Smells Like Teen Spirit", "artist": "Nirvana", "album": "Nevermind", "album_art": "https://i.scdn.co/image/ab67616d0000b273e175a19e530c898d167d39bf"},
        {"id": "4P5KoOXQll46HG8GSpHo9O", "name": "Come As You Are", "artist": "Nirvana", "album": "Nevermind", "album_art": "https://i.scdn.co/image/ab67616d0000b273e175a19e530c898d167d39bf"},
        {"id": "6dGnYIeXmYdcikdzNNDMm2", "name": "Here Comes The Sun", "artist": "The Beatles", "album": "Abbey Road", "album_art": "https://i.scdn.co/image/ab67616d0000b273dc30583ba717007b00cceb25"},
    ],
    "neutral": [
        {"id": "3KkXRQHbMCARz0aVfEt68P", "name": "Sunflower", "artist": "Post Malone, Swae Lee", "album": "Spider-Man: Into the Spider-Verse", "album_art": "https://i.scdn.co/image/ab67616d0000b273e2e352d89826aef6dbd5ff8f"},
        {"id": "0pqnGHJpmpxLKifKRmU6WP", "name": "Believer", "artist": "Imagine Dragons", "album": "Evolve", "album_art": "https://i.scdn.co/image/ab67616d0000b2735675e83f707f1d7271e5cf8a"},
        {"id": "7dt6x5M1jzdnqII0x6E17S", "name": "Sunday Best", "artist": "Surfaces", "album": "Where the Light Is", "album_art": "https://i.scdn.co/image/ab67616d0000b2731c3aa71b000be0876793ceb1"},
    ],
}


def _get_auth_headers():
    client_id = getattr(settings, 'SPOTIFY_CLIENT_ID', '').strip()
    client_secret = getattr(settings, 'SPOTIFY_CLIENT_SECRET', '').strip()
    if not client_id or not client_secret:
        return None
    creds = f"{client_id}:{client_secret}".encode("utf-8")
    b64_creds = base64.b64encode(creds).decode("utf-8")
    return {"Authorization": f"Basic {b64_creds}"}


def is_spotify_configured():
    client_id = getattr(settings, 'SPOTIFY_CLIENT_ID', '').strip()
    client_secret = getattr(settings, 'SPOTIFY_CLIENT_SECRET', '').strip()
    return bool(client_id and client_secret)


def configure_spotify_credentials(client_id, client_secret):
    """
    Saves client credentials to .env and updates runtime Django settings.
    """
    client_id = client_id.strip()
    client_secret = client_secret.strip()
    env_path = os.path.join(settings.BASE_DIR, '.env')

    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

    updated_id = False
    updated_secret = False
    new_lines = []
    for line in lines:
        if line.startswith('SPOTIFY_CLIENT_ID='):
            new_lines.append(f"SPOTIFY_CLIENT_ID={client_id}\n")
            updated_id = True
        elif line.startswith('SPOTIFY_CLIENT_SECRET='):
            new_lines.append(f"SPOTIFY_CLIENT_SECRET={client_secret}\n")
            updated_secret = True
        else:
            new_lines.append(line)

    if not updated_id:
        new_lines.append(f"SPOTIFY_CLIENT_ID={client_id}\n")
    if not updated_secret:
        new_lines.append(f"SPOTIFY_CLIENT_SECRET={client_secret}\n")

    with open(env_path, 'w') as f:
        f.writelines(new_lines)

    settings.SPOTIFY_CLIENT_ID = client_id
    settings.SPOTIFY_CLIENT_SECRET = client_secret
    return True


def login_demo_user(request):
    """
    Instantly links a demonstration Spotify account so users can test the
    complete emotion-to-music pipeline without creating developer credentials.
    """
    demo_profile = {
        'id': 'tunex_demo_user',
        'display_name': 'TUNEX Demo Listener',
        'avatar': 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80',
        'email': 'demo@tunex.ai',
        'product': 'premium',
        'profile_url': 'https://open.spotify.com',
    }
    request.session['spotify_auth'] = {
        'access_token': 'tunex_demo_active_token',
        'refresh_token': 'tunex_demo_refresh_token',
        'expires_at': time.time() + 86400 * 30,
        'is_demo': True,
        'user': demo_profile,
    }
    request.session.modified = True
    return demo_profile


def get_auth_url(request):
    """
    Generates Spotify OAuth2 Authorization URL with state verification.
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

        user_profile = get_user_profile(access_token)

        request.session['spotify_auth'] = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': expires_at,
            'is_demo': False,
            'user': user_profile,
        }
        request.session.modified = True
        return True, user_profile
    except Exception as e:
        logger.exception("Exception during Spotify token exchange")
        return False, str(e)


def get_user_profile(access_token):
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
    auth = request.session.get('spotify_auth')
    if not auth or 'access_token' not in auth:
        return None

    if auth.get('is_demo'):
        return auth['access_token']

    now = time.time()
    if now < auth.get('expires_at', 0):
        return auth['access_token']

    refresh_token = auth.get('refresh_token')
    if not refresh_token:
        request.session.pop('spotify_auth', None)
        return None

    return refresh_access_token(request, refresh_token)


def refresh_access_token(request, refresh_token):
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
            logger.warning(f"Refresh token error: {response.status_code}")
            request.session.pop('spotify_auth', None)
            return None
    except Exception as e:
        logger.exception("Failed to refresh Spotify token")
        return None


def disconnect_spotify(request):
    request.session.pop('spotify_auth', None)
    request.session.pop('spotify_oauth_state', None)
    request.session.modified = True


def get_recommendations_for_emotion(access_token, emotion, limit=10):
    """
    Fetches personalized song recommendations based on detected emotion.
    Seamlessly supports both live Spotify API and Demo Mode.
    """
    emotion_key = emotion.lower().strip() if emotion else "neutral"
    profile = EMOTION_AUDIO_PROFILES.get(emotion_key, EMOTION_AUDIO_PROFILES["neutral"])

    # If demo mode token, serve curated tracks with real Spotify IDs
    if access_token == 'tunex_demo_active_token':
        demo_list = DEMO_TRACKS.get(emotion_key, DEMO_TRACKS["neutral"])
        tracks = []
        for t in demo_list[:limit]:
            tracks.append({
                "id": t["id"],
                "name": t["name"],
                "artist": t["artist"],
                "album": t.get("album", "Original Soundtrack"),
                "album_art": t.get("album_art", "/static/images/default_album.png"),
                "spotify_url": f"https://open.spotify.com/track/{t['id']}",
                "uri": f"spotify:track:{t['id']}",
                "embed_url": f"https://open.spotify.com/embed/track/{t['id']}?utm_source=generator&theme=0",
                "emotion": emotion_key,
                "color": profile["color"],
            })
        return {
            "emotion": emotion_key,
            "profile": profile,
            "tracks": tracks,
        }

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
        logger.warning(f"Recommendations endpoint error: {e}")

    # Step 2: Fallback to Search API (Spotify 2024+ restricted recommendations for unverified apps)
    if not tracks:
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

    # Step 3: If API yielded nothing (e.g. rate limit), fallback to curated demo tracks
    if not tracks:
        demo_list = DEMO_TRACKS.get(emotion_key, DEMO_TRACKS["neutral"])
        for t in demo_list[:limit]:
            tracks.append({
                "id": t["id"],
                "name": t["name"],
                "artist": t["artist"],
                "album": t.get("album", "Original Soundtrack"),
                "album_art": t.get("album_art", "/static/images/default_album.png"),
                "spotify_url": f"https://open.spotify.com/track/{t['id']}",
                "uri": f"spotify:track:{t['id']}",
                "embed_url": f"https://open.spotify.com/embed/track/{t['id']}?utm_source=generator&theme=0",
                "emotion": emotion_key,
                "color": profile["color"],
            })

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
