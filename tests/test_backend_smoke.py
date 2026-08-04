import importlib.util
import os
import sys

import cv2
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from backend_app import app


def test_imports():
    required = ['cv2', 'numpy', 'tensorflow']
    for mod in required:
        assert importlib.util.find_spec(mod) is not None, f'{mod} missing'


def test_model_exists():
    model_path = os.path.join(ROOT, 'models', 'best_model.keras')
    assert os.path.exists(model_path), 'model missing'


def test_opencv_has_haar_cascade():
    assert hasattr(cv2, 'CascadeClassifier'), 'opencv headless missing CascadeClassifier'


def test_realtime_only_backend_routes():
    client = TestClient(app)
    routes = {route.path for route in app.routes}

    assert '/ws/emotion' in routes
    assert '/predict/image' not in routes

    health = client.get('/health')
    assert health.status_code == 200
