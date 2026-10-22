"""
TUNEX Camera Module (Deprecated)

Video capture has been moved to client-side HTML5 navigator.mediaDevices.getUserMedia
with frame transmission via the /api/detect-emotion/ REST endpoint.
This file remains as an architectural reference.
"""

# Client-side streaming replaces server-side cv2.VideoCapture(0)