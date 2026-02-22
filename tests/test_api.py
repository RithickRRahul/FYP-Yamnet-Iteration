import os
import sys

# Set recursion limit for TensorFlow
sys.setrecursionlimit(10000)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    print(f"Health Response: {response.status_code}")
    print(response.json())
    assert response.status_code == 200

def test_root():
    response = client.get("/")
    print(f"Root Response: {response.status_code}")
    print(response.json())
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing FastAPI Client...")
    try:
        test_health()
        test_root()
        print("API TESTS PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
