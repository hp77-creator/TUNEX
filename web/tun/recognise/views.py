import json
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.contrib import messages

from .form import ImageForm
from .models import Image
from .ai_service import analyze_emotion, EMOTIONS
from .spotify_service import (
    get_auth_url,
    handle_callback,
    get_active_token,
    disconnect_spotify,
    get_recommendations_for_emotion,
    EMOTION_AUDIO_PROFILES,
)

logger = logging.getLogger(__name__)


def home(request):
    """
    Landing page showcasing TUNEX, mood categories, and connection status.
    """
    token = get_active_token(request)
    context = {
        'emotions': EMOTION_AUDIO_PROFILES,
        'spotify_linked': bool(token),
    }
    return render(request, 'home.html', context)


@ensure_csrf_cookie
def showlive(request):
    """
    Interactive live webcam page for real-time facial expression analysis.
    """
    token = get_active_token(request)
    auth_url = get_auth_url(request) if not token else None
    context = {
        'spotify_linked': bool(token),
        'spotify_auth_url': auth_url,
        'emotions': EMOTION_AUDIO_PROFILES,
    }
    return render(request, 'live.html', context)


def form(request):
    """
    Static photo upload and analysis view.
    """
    token = get_active_token(request)
    result_data = None
    tracks_data = None
    emotion = None
    auth_url = get_auth_url(request) if not token else None

    if request.method == "POST":
        upload_form = ImageForm(data=request.POST, files=request.FILES)
        if upload_form.is_valid():
            image_obj = upload_form.save()
            try:
                # Analyze image directly using AI service
                with open(image_obj.image.path, 'rb') as f:
                    result_data = analyze_emotion(f.read())

                emotion = result_data.get('emotion')
                request.session['last_detected_emotion'] = emotion

                # If Spotify is linked, fetch songs automatically
                if token and emotion:
                    tracks_data = get_recommendations_for_emotion(token, emotion, limit=8)
            except Exception as e:
                logger.exception("Error processing uploaded image")
                messages.error(request, f"Could not analyze image: {e}")
        else:
            messages.error(request, "Please upload a valid image file.")
    else:
        upload_form = ImageForm()

    context = {
        'form': upload_form,
        'result': result_data,
        'tracks_data': tracks_data,
        'emotion': emotion,
        'spotify_linked': bool(token),
        'spotify_auth_url': auth_url,
    }
    return render(request, 'predict.html', context)


@ensure_csrf_cookie
def recommendations_view(request):
    """
    Full recommendations view displaying songs tuned to the user's emotion.
    Requires linked Spotify account.
    """
    token = get_active_token(request)
    emotion = request.GET.get('emotion') or request.session.get('last_detected_emotion') or 'happy'

    if not token:
        # Prompt user to connect Spotify
        auth_url = get_auth_url(request)
        context = {
            'spotify_linked': False,
            'spotify_auth_url': auth_url,
            'emotion': emotion,
        }
        return render(request, 'liveres.html', context)

    # Fetch recommendations from Spotify
    rec_result = get_recommendations_for_emotion(token, emotion, limit=12)
    profile = EMOTION_AUDIO_PROFILES.get(emotion.lower(), EMOTION_AUDIO_PROFILES['neutral'])

    context = {
        'spotify_linked': True,
        'emotion': emotion,
        'profile': profile,
        'tracks': rec_result.get('tracks', []),
    }
    return render(request, 'liveres.html', context)


# ==========================================
# JSON API ENDPOINTS
# ==========================================

@ensure_csrf_cookie
def api_detect_emotion(request):
    """
    AJAX endpoint called by live browser webcam to analyze a frame.
    Accepts JSON payload: {"image": "data:image/jpeg;base64,..."}
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    try:
        data = json.loads(request.body)
        image_data = data.get("image")
        if not image_data:
            return JsonResponse({"success": False, "error": "No image provided"}, status=400)

        analysis = analyze_emotion(image_data)
        emotion = analysis.get("emotion")
        request.session['last_detected_emotion'] = emotion

        token = get_active_token(request)
        analysis['spotify_linked'] = bool(token)
        analysis['spotify_auth_url'] = get_auth_url(request) if not token else None

        return JsonResponse(analysis)
    except Exception as e:
        logger.exception("Error in api_detect_emotion")
        return JsonResponse({"success": False, "error": str(e)}, status=500)


def api_get_recommendations(request):
    """
    AJAX endpoint to fetch Spotify recommendations for an emotion.
    Requires linked Spotify account.
    """
    token = get_active_token(request)
    if not token:
        auth_url = get_auth_url(request)
        return JsonResponse({
            "success": False,
            "error": "spotify_required",
            "message": "Spotify account linking is required to get song recommendations.",
            "auth_url": auth_url,
        }, status=401)

    emotion = request.GET.get('emotion') or request.session.get('last_detected_emotion') or 'neutral'
    limit = int(request.GET.get('limit', 10))

    try:
        rec_data = get_recommendations_for_emotion(token, emotion, limit=limit)
        return JsonResponse({
            "success": True,
            "emotion": emotion,
            "profile": rec_data.get("profile"),
            "tracks": rec_data.get("tracks", []),
        })
    except Exception as e:
        logger.exception("Error in api_get_recommendations")
        return JsonResponse({"success": False, "error": str(e)}, status=500)


# ==========================================
# SPOTIFY OAUTH2 VIEWS
# ==========================================

def spotify_login(request):
    """
    Redirects user to Spotify Authorization page.
    """
    auth_url = get_auth_url(request)
    if not auth_url:
        messages.error(
            request,
            "Spotify credentials not configured. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env"
        )
        return redirect('home')
    return redirect(auth_url)


def spotify_callback(request):
    """
    Handles redirect callback from Spotify OAuth2.
    """
    code = request.GET.get('code')
    state = request.GET.get('state')
    error = request.GET.get('error')

    if error:
        messages.error(request, f"Spotify authorization declined: {error}")
        return redirect('home')

    if not code:
        messages.error(request, "Authorization code not returned by Spotify.")
        return redirect('home')

    success, result = handle_callback(request, code, state)
    if success:
        user_name = result.get('display_name', 'User')
        messages.success(request, f"Successfully connected to Spotify as {user_name}!")
        # If user had a detected emotion, redirect directly to recommendations
        last_emotion = request.session.get('last_detected_emotion')
        if last_emotion:
            return redirect(f"/recommendations/?emotion={last_emotion}")
        return redirect('showlive')
    else:
        messages.error(request, f"Failed to connect Spotify account: {result}")
        return redirect('home')


def spotify_disconnect(request):
    """
    Disconnects the user's Spotify account from session.
    """
    disconnect_spotify(request)
    messages.info(request, "Disconnected from Spotify.")
    return redirect('home')
