# TUNEX Installation & Setup Guide

## Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Spotify Developer Account**: Free at [developer.spotify.com](https://developer.spotify.com/dashboard)
- **Webcam**: Any built-in or USB webcam (for live streaming mode)

---

## 1. Local Setup with Virtual Environment

### Clone the Repository
```bash
git clone https://github.com/hp77-creator/TUNEX.git tunex
cd tunex
```

### Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Configure Spotify Developer API

To enable user account linking and song suggestions:

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and log in.
2. Click **Create App**:
   - **App Name**: `TUNEX`
   - **App Description**: `Facial Emotion Music Recommender`
   - **Redirect URI**: `http://127.0.0.1:8000/spotify/callback/`
   - **APIs Used**: Select **Web API**
3. Open your app settings to find your **Client ID** and **Client Secret**.
4. Create a `.env` file in `web/tun/.env` (or copy `.env.example`):
   ```bash
   cp .env.example web/tun/.env
   ```
5. Edit `web/tun/.env` and paste your Spotify credentials:
   ```env
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   SPOTIFY_CLIENT_ID=your_actual_client_id
   SPOTIFY_CLIENT_SECRET=your_actual_client_secret
   SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/spotify/callback/
   ```

---

## 3. Run Database Migrations & Start Server

```bash
cd web/tun
python manage.py migrate
python manage.py runserver
```

Now open your browser at **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**.

---

## 4. Run Automated Tests

To run the complete test suite verifying AI models, views, and Spotify authentication gates:

```bash
python manage.py test recognise
```

---

## 5. Docker Setup (Optional)

You can also run TUNEX in Docker:

```bash
cd web/tun
docker build -t tunex_app .
docker run -p 8000:8000 --env-file .env tunex_app
```
