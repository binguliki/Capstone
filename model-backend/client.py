"""
Python client for Leaf Disease Detection API
Example usage and helper functions for API integration
"""

import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time


class LeafDiseaseClient:
    """Client for interacting with Leaf Disease Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.
        
        Args:
            base_url: API endpoint URL (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def predict(self, image_path: str, timeout: int = 30) -> Dict:
        """
        Predict disease classification and bounding boxes for a leaf image.
        
        Args:
            image_path: Path to image file (JPEG/PNG/WebP)
            timeout: Request timeout in seconds
        
        Returns:
            Dictionary with predictions or error
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {'error': f'Image not found: {image_path}'}
        
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            return {'error': 'Unsupported image format. Use JPEG, PNG, or WebP.'}
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(
                    f"{self.base_url}/predict",
                    files=files,
                    timeout=timeout
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'API error {response.status_code}',
                    'detail': response.json().get('detail', response.text)
                }
        except requests.exceptions.Timeout:
            return {'error': f'Request timeout after {timeout}s'}
        except requests.exceptions.ConnectionError:
            return {'error': f'Cannot connect to {self.base_url}'}
        except Exception as e:
            return {'error': str(e)}
    
    def health(self) -> Dict:
        """
        Get service health status.
        
        Returns:
            Health status dictionary
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {'status': 'error', 'code': response.status_code}
        except Exception as e:
            return {'status': 'unreachable', 'error': str(e)}
    
    def metrics(self) -> Dict:
        """
        Get service metrics.
        
        Returns:
            Metrics dictionary
        """
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'API error {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def batch_predict(self, image_dir: str, pattern: str = "*.jpg") -> List[Dict]:
        """
        Predict on multiple images in a directory.
        
        Args:
            image_dir: Directory containing images
            pattern: File pattern to match (default: *.jpg)
        
        Returns:
            List of prediction results
        """
        image_dir = Path(image_dir)
        results = []
        
        for image_path in sorted(image_dir.glob(pattern)):
            result = self.predict(str(image_path))
            result['image'] = image_path.name
            results.append(result)
        
        return results


def print_prediction(prediction: Dict, verbose: bool = True):
    """Pretty print prediction results."""
    if 'error' in prediction:
        print(f"❌ Error: {prediction['error']}")
        if 'detail' in prediction:
            print(f"   Detail: {prediction['detail']}")
        return
    
    print(f"✅ Classification: {prediction['classification']}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    print(f"   Inference: {prediction['inference_time_ms']:.0f}ms")
    
    if prediction['bounding_boxes']:
        print(f"   Detected {len(prediction['bounding_boxes'])} region(s):")
        for i, box in enumerate(prediction['bounding_boxes'], 1):
            print(f"     {i}. {box['label']} ({box['confidence']:.0%}) "
                  f"[{box['x1']:.2f}, {box['y1']:.2f}, {box['x2']:.2f}, {box['y2']:.2f}]")
    else:
        print("   No disease regions detected")


def batch_predict_with_summary(client: LeafDiseaseClient, image_dir: str) -> Dict:
    """Run batch predictions and return summary statistics."""
    results = client.batch_predict(image_dir)
    
    summary = {
        'total_images': len(results),
        'successful': 0,
        'failed': 0,
        'classifications': {},
        'avg_inference_time_ms': 0,
        'total_inference_time_ms': 0,
    }
    
    for result in results:
        if 'error' in result:
            summary['failed'] += 1
        else:
            summary['successful'] += 1
            
            cls = result['classification']
            summary['classifications'][cls] = summary['classifications'].get(cls, 0) + 1
            summary['total_inference_time_ms'] += result['inference_time_ms']
    
    if summary['successful'] > 0:
        summary['avg_inference_time_ms'] = (
            summary['total_inference_time_ms'] / summary['successful']
        )
    
    return summary, results


# Example usage
if __name__ == "__main__":
    client = LeafDiseaseClient()
    
    # Check health
    print("Checking API health...")
    health = client.health()
    print(json.dumps(health, indent=2))
    print()
    
    # Get metrics
    print("Getting metrics...")
    metrics = client.metrics()
    print(json.dumps(metrics, indent=2))
    print()
    
    # Example prediction (uncomment to test with actual image)
    # print("Making prediction...")
    # result = client.predict("path/to/leaf_image.jpg")
    # print_prediction(result)
