from django.conf import settings
from .spotify_service import get_active_token

def spotify_context(request):
    """
    Exposes Spotify authorization status and user profile to all templates.
    """
    auth_data = request.session.get('spotify_auth')
    spotify_linked = False
    spotify_user = None

    if auth_data:
        # Check / refresh token if valid
        token = get_active_token(request)
        if token:
            spotify_linked = True
            spotify_user = auth_data.get('user', {})

    return {
        'spotify_linked': spotify_linked,
        'spotify_user': spotify_user,
        'spotify_configured': bool(getattr(settings, 'SPOTIFY_CLIENT_ID', '') and getattr(settings, 'SPOTIFY_CLIENT_SECRET', '')),
    }
