from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path('', views.home, name="home"),
    path('home/', views.home, name="home_alt"),
    path('live/', views.showlive, name="showlive"),
    path('showlive/', views.showlive, name="showlive_alt"),
    path('upload/', views.form, name="form"),
    path('form/', views.form, name="form_alt"),
    path('recommendations/', views.recommendations_view, name="liveres"),
    path('liveres/', views.recommendations_view, name="liveres_alt"),

    # REST APIs
    path('api/detect-emotion/', views.api_detect_emotion, name="api_detect_emotion"),
    path('api/recommendations/', views.api_get_recommendations, name="api_recommendations"),

    # Spotify OAuth2
    path('spotify/login/', views.spotify_login, name="spotify_login"),
    path('spotify/demo/', views.spotify_demo, name="spotify_demo"),
    path('spotify/configure/', views.spotify_configure, name="spotify_configure"),
    path('spotify/callback/', views.spotify_callback, name="spotify_callback"),
    path('spotify/disconnect/', views.spotify_disconnect, name="spotify_disconnect"),
]
