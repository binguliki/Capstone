"""
Test suite for Leaf Disease Detection API
Run with: pytest test_api.py -v
Or for specific test: pytest test_api.py::test_health -v
"""

import pytest
import requests
import json
from pathlib import Path
from PIL import Image
import numpy as np
import io

# Configuration
API_URL = "http://localhost:8000"
TIMEOUT = 30


class TestAPIEndpoints:
    """Test API endpoints and functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.api_url = API_URL
    
    @pytest.mark.smoke
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = requests.get(f"{self.api_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "endpoints" in data
        assert data["service"] == "Leaf Disease Detection API"
    
    @pytest.mark.smoke
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.api_url}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert data["status"] in ["healthy", "degraded"]
        assert isinstance(data["model_loaded"], bool)
    
    @pytest.mark.smoke
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = requests.get(f"{self.api_url}/metrics", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "total_requests", "successful_predictions", "failed_predictions",
            "success_rate_percent", "average_inference_time_ms",
            "p95_inference_time_ms", "p99_inference_time_ms", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        assert data["success_rate_percent"] >= 0
        assert data["success_rate_percent"] <= 100
        assert data["average_inference_time_ms"] >= 0
    
    @staticmethod
    def create_test_image(width=256, height=256, format='JPEG'):
        """Create a synthetic test image."""
        # Create a green-tinted image to simulate leaf
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        img_array[:, :, 1] = 100  # Green channel
        img_array[:, :, 0] = 50   # Red channel
        img_array[:, :, 2] = 50   # Blue channel
        
        # Add some random spots to simulate disease
        for _ in range(5):
            y, x = np.random.randint(50, height-50, 2)
            img_array[y:y+20, x:x+20] = [150, 100, 100]
        
        img = Image.fromarray(img_array, 'RGB')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes
    
    @pytest.mark.core
    def test_predict_valid_image(self):
        """Test prediction with valid image."""
        test_image = self.create_test_image()
        
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(
            f"{self.api_url}/predict",
            files=files,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "classification" in data
        assert "confidence" in data
        assert "bounding_boxes" in data
        assert "inference_time_ms" in data
        assert "timestamp" in data
        assert "model_version" in data
        
        # Validate data types
        assert isinstance(data["classification"], str)
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["bounding_boxes"], list)
        assert data["inference_time_ms"] > 0
    
    @pytest.mark.core
    def test_predict_bounding_box_format(self):
        """Test bounding box format validation."""
        test_image = self.create_test_image()
        
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(f"{self.api_url}/predict", files=files, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        
        for bbox in data["bounding_boxes"]:
            assert "x1" in bbox
            assert "y1" in bbox
            assert "x2" in bbox
            assert "y2" in bbox
            assert "label" in bbox
            assert "confidence" in bbox
            
            # Validate normalized coordinates
            assert 0 <= bbox["x1"] <= 1, "x1 out of range"
            assert 0 <= bbox["y1"] <= 1, "y1 out of range"
            assert 0 <= bbox["x2"] <= 1, "x2 out of range"
            assert 0 <= bbox["y2"] <= 1, "y2 out of range"
            assert bbox["x1"] < bbox["x2"], "Invalid x coordinates"
            assert bbox["y1"] < bbox["y2"], "Invalid y coordinates"
            assert 0 <= bbox["confidence"] <= 1
    
    @pytest.mark.error_handling
    def test_predict_missing_file(self):
        """Test prediction without file."""
        response = requests.post(f"{self.api_url}/predict", timeout=TIMEOUT)
        assert response.status_code != 200
        # Should be 422 (validation error) or 400
        assert response.status_code in [400, 422]
    
    @pytest.mark.error_handling
    def test_predict_invalid_format(self):
        """Test prediction with invalid image format."""
        # Create invalid image
        invalid_content = b"This is not an image"
        files = {'file': ('test.txt', invalid_content, 'text/plain')}
        
        response = requests.post(f"{self.api_url}/predict", files=files, timeout=TIMEOUT)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.error_handling
    def test_predict_oversized_image(self):
        """Test prediction with oversized image."""
        # Create large image (>25MB)
        large_image = np.random.randint(0, 256, (4000, 4000, 3), dtype=np.uint8)
        img = Image.fromarray(large_image, 'RGB')
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        # Only test if image is actually > 25MB
        if len(img_bytes.getvalue()) > 25 * 1024 * 1024:
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=TIMEOUT)
            assert response.status_code == 413  # Request entity too large
    
    @pytest.mark.load
    def test_multiple_sequential_predictions(self):
        """Test multiple sequential predictions."""
        success_count = 0
        
        for i in range(5):
            test_image = self.create_test_image()
            files = {'file': ('test.jpg', test_image, 'image/jpeg')}
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=TIMEOUT)
            
            if response.status_code == 200:
                success_count += 1
        
        assert success_count == 5, f"Only {success_count}/5 predictions succeeded"
    
    @pytest.mark.performance
    def test_prediction_latency(self):
        """Test prediction latency."""
        test_image = self.create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        
        response = requests.post(f"{self.api_url}/predict", files=files, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        inference_time = data["inference_time_ms"]
        
        # Inference should typically be < 1000ms
        assert inference_time < 1000, f"Inference took {inference_time}ms, expected < 1000ms"
        print(f"\nInference time: {inference_time:.2f}ms")


class TestMetricsTracking:
    """Test metrics tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_url = API_URL
    
    def test_metrics_increment_on_success(self):
        """Test that metrics increment on successful prediction."""
        # Get initial metrics
        response = requests.get(f"{self.api_url}/metrics", timeout=TIMEOUT)
        initial_metrics = response.json()
        initial_total = initial_metrics["total_requests"]
        
        # Make a prediction
        test_image = TestAPIEndpoints.create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        requests.post(f"{self.api_url}/predict", files=files, timeout=TIMEOUT)
        
        # Check updated metrics
        response = requests.get(f"{self.api_url}/metrics", timeout=TIMEOUT)
        updated_metrics = response.json()
        updated_total = updated_metrics["total_requests"]
        
        assert updated_total > initial_total, "Metrics not updated"


class TestIntegration:
    """Integration tests combining multiple endpoints."""
    
    def test_full_workflow(self):
        """Test full workflow: health check → predict → check metrics."""
        api_url = API_URL
        
        # 1. Health check
        health = requests.get(f"{api_url}/health", timeout=TIMEOUT)
        assert health.status_code == 200
        assert health.json()["model_loaded"]
        
        # 2. Make prediction
        test_image = TestAPIEndpoints.create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        prediction = requests.post(f"{api_url}/predict", files=files, timeout=TIMEOUT)
        assert prediction.status_code == 200
        
        # 3. Check metrics updated
        metrics = requests.get(f"{api_url}/metrics", timeout=TIMEOUT)
        assert metrics.status_code == 200
        assert metrics.json()["total_requests"] > 0


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "smoke: Smoke tests")
    config.addinivalue_line("markers", "core: Core functionality tests")
    config.addinivalue_line("markers", "error_handling: Error handling tests")
    config.addinivalue_line("markers", "load: Load tests")
    config.addinivalue_line("markers", "performance: Performance tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
