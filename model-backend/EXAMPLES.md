# API Usage Examples & Patterns

Quick copy-paste examples for common use cases.

## 🔗 cURL Examples

### Make a Prediction
```bash
# Basic prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/leaf.jpg"

# Pretty print response
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/leaf.jpg" | python -m json.tool

# Save response to file
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/leaf.jpg" > prediction.json
```

### Health Check
```bash
curl http://localhost:8000/health | python -m json.tool
```

### Get Metrics
```bash
curl http://localhost:8000/metrics | python -m json.tool
```

---

## 🐍 Python Examples

### Basic Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("leaf.jpg", "rb")}
)
result = response.json()

print(f"Disease: {result['classification']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Inference time: {result['inference_time_ms']:.0f}ms")
```

### Using the Client Library
```python
from client import LeafDiseaseClient

client = LeafDiseaseClient(base_url="http://localhost:8000")
result = client.predict("leaf.jpg")

if 'error' not in result:
    print(f"✓ {result['classification']} ({result['confidence']:.1%})")
    for bbox in result['bounding_boxes']:
        print(f"  - {bbox['label']}: {bbox['confidence']:.0%}")
else:
    print(f"✗ {result['error']}")
```

### Batch Processing
```python
from client import LeafDiseaseClient
import json

client = LeafDiseaseClient()
results = client.batch_predict("path/to/images/", pattern="*.jpg")

# Save results
with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
successful = sum(1 for r in results if 'error' not in r)
print(f"Processed: {len(results)} | Success: {successful}")
```

### Health Monitoring (Circuit Breaker)
```python
from client import LeafDiseaseClient
from datetime import datetime, timedelta

client = LeafDiseaseClient()

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = None
        self.is_open = False
    
    def check(self):
        """Check if we should allow requests"""
        health = client.health()
        
        if health.get('status') == 'healthy':
            self.failure_count = 0
            self.is_open = False
            return True
        
        self.failure_count += 1
        self.last_failure = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            return False
        
        return True
    
    def should_retry(self):
        """Check if we should retry after timeout"""
        if not self.is_open:
            return True
        
        if datetime.now() - self.last_failure > timedelta(seconds=self.timeout):
            print("Circuit breaker: Attempting recovery...")
            self.failure_count = 0
            self.is_open = False
            return True
        
        return False

# Usage
breaker = CircuitBreaker()

try:
    if breaker.check():
        result = client.predict("leaf.jpg")
    else:
        print("Service unavailable - circuit open")
except Exception as e:
    print(f"Request failed: {e}")
    if breaker.should_retry():
        print("Retrying...")
```

### Metrics Monitoring
```python
from client import LeafDiseaseClient
import time

client = LeafDiseaseClient()

# Check metrics every 10 seconds
for i in range(5):
    metrics = client.metrics()
    
    print(f"\n--- Metrics Check {i+1} ---")
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Success rate: {metrics['success_rate_percent']:.1f}%")
    print(f"Avg latency: {metrics['average_inference_time_ms']:.0f}ms")
    print(f"P95 latency: {metrics['p95_inference_time_ms']:.0f}ms")
    print(f"P99 latency: {metrics['p99_inference_time_ms']:.0f}ms")
    
    time.sleep(10)
```

---

## 📊 JavaScript/Node.js Examples

### Fetch Prediction
```javascript
async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
const imageInput = document.getElementById('imageInput');
imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  
  try {
    const result = await predictDisease(file);
    console.log(`Disease: ${result.classification}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    
    result.bounding_boxes.forEach((bbox, i) => {
      console.log(`  ${i+1}. ${bbox.label} (${(bbox.confidence * 100).toFixed(0)}%)`);
    });
  } catch (error) {
    console.error('Prediction failed:', error);
  }
});
```

### Health Check & Circuit Breaker
```javascript
class APICircuitBreaker {
  constructor(failureThreshold = 5, timeout = 60000) {
    this.failureThreshold = failureThreshold;
    this.timeout = timeout;
    this.failureCount = 0;
    this.lastFailure = null;
    this.isOpen = false;
  }
  
  async check() {
    try {
      const response = await fetch('http://localhost:8000/health');
      const health = await response.json();
      
      if (health.status === 'healthy') {
        this.failureCount = 0;
        this.isOpen = false;
        return true;
      }
    } catch (error) {
      this.failureCount++;
      this.lastFailure = Date.now();
      
      if (this.failureCount >= this.failureThreshold) {
        this.isOpen = true;
      }
    }
    
    return !this.isOpen;
  }
  
  shouldRetry() {
    if (!this.isOpen) return true;
    
    const timeSinceFailure = Date.now() - this.lastFailure;
    if (timeSinceFailure > this.timeout) {
      console.log('Circuit breaker: Attempting recovery...');
      this.failureCount = 0;
      this.isOpen = false;
      return true;
    }
    
    return false;
  }
}

// Usage
const breaker = new APICircuitBreaker();

async function predictSafely(imageFile) {
  if (!await breaker.check()) {
    throw new Error('Service unavailable - circuit breaker open');
  }
  
  return await predictDisease(imageFile);
}
```

### React Hook Example
```javascript
import React, { useState } from 'react';

function LeafDiseasePredictor() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="predictor">
      <h2>Leaf Disease Detector</h2>
      
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        disabled={loading}
      />
      
      {loading && <p>Analyzing image...</p>}
      
      {error && <p style={{color: 'red'}}>Error: {error}</p>}
      
      {result && (
        <div className="results">
          <h3>{result.classification}</h3>
          <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          <p>Inference time: {result.inference_time_ms.toFixed(0)}ms</p>
          
          {result.bounding_boxes.length > 0 && (
            <div>
              <h4>Detected regions:</h4>
              <ul>
                {result.bounding_boxes.map((bbox, i) => (
                  <li key={i}>
                    {bbox.label} ({(bbox.confidence * 100).toFixed(0)}%)
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default LeafDiseasePredictor;
```

---

## 🧪 Integration Testing

### Test with Multiple Images
```python
from client import LeafDiseaseClient
from pathlib import Path
import json

client = LeafDiseaseClient()
image_dir = Path("test_images/")

# Predict on all images
results = client.batch_predict(str(image_dir), pattern="*.jpg")

# Analyze results
diseases = {}
for r in results:
    if 'error' not in r:
        disease = r['classification']
        diseases[disease] = diseases.get(disease, 0) + 1

# Print summary
print("\nDisease Distribution:")
for disease, count in sorted(diseases.items(), key=lambda x: x[1], reverse=True):
    print(f"  {disease}: {count}")

print(f"\nTotal processed: {len(results)}")
print(f"Successful: {len([r for r in results if 'error' not in r])}")
```

### Performance Profiling
```python
from client import LeafDiseaseClient
import time
import statistics

client = LeafDiseaseClient()
image_path = "test_image.jpg"

# Warm up
client.predict(image_path)

# Run multiple predictions and measure
times = []
for _ in range(10):
    start = time.time()
    client.predict(image_path)
    times.append(time.time() - start)

print(f"Min: {min(times)*1000:.0f}ms")
print(f"Max: {max(times)*1000:.0f}ms")
print(f"Mean: {statistics.mean(times)*1000:.0f}ms")
print(f"Median: {statistics.median(times)*1000:.0f}ms")
print(f"StdDev: {statistics.stdev(times)*1000:.0f}ms")
```

---

## 🚀 Deployment Integration

### Docker Compose with Custom Network
```yaml
version: '3.8'
services:
  api:
    container_name: leaf-api
    build: .
    ports:
      - "8000:8000"
    networks:
      - api-network
  
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    networks:
      - api-network
    depends_on:
      - api

networks:
  api-network:
    driver: bridge
```

### Kubernetes ConfigMap Mount
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: leaf-disease-config
data:
  model.h5: |
    # Binary model data would go here
    # Or reference external storage

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: leaf-disease-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: leaf-disease-api:latest
        env:
        - name: MODEL_PATH
          value: /models/modelv1.h5
        volumeMounts:
        - name: model
          mountPath: /models
      volumes:
      - name: model
        configMap:
          name: leaf-disease-config
```

---

## 📋 Common Patterns

### Pattern 1: Async Batch Processing
```python
import asyncio
import concurrent.futures
from client import LeafDiseaseClient

async def process_images_batch(image_files, batch_size=5):
    client = LeafDiseaseClient()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, client.predict, img)
            for img in image_files
        ]
        results = await asyncio.gather(*tasks)
    
    return results

# Usage
image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = asyncio.run(process_images_batch(image_files))
```

### Pattern 2: Retry Logic
```python
from client import LeafDiseaseClient
import time

def predict_with_retry(image_path, max_retries=3, backoff=0.5):
    client = LeafDiseaseClient()
    
    for attempt in range(max_retries):
        try:
            result = client.predict(image_path)
            if 'error' not in result:
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff * (2 ** attempt)
                print(f"Retry {attempt+1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    return {'error': 'Max retries exceeded'}
```

### Pattern 3: Caching Results
```python
from functools import lru_cache
from pathlib import Path
import hashlib
from client import LeafDiseaseClient

class CachedClient:
    def __init__(self):
        self.client = LeafDiseaseClient()
        self.cache = {}
    
    def get_image_hash(self, image_path):
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def predict(self, image_path):
        img_hash = self.get_image_hash(image_path)
        
        if img_hash in self.cache:
            print(f"Cache hit for {image_path}")
            return self.cache[img_hash]
        
        result = self.client.predict(image_path)
        if 'error' not in result:
            self.cache[img_hash] = result
        
        return result

# Usage
cached = CachedClient()
result1 = cached.predict("leaf.jpg")
result2 = cached.predict("leaf.jpg")  # FromCache
```

---

## 📞 Troubleshooting Examples

### Check Why Predictions Are Failing
```python
from client import LeafDiseaseClient

client = LeafDiseaseClient()

# 1. Check health
health = client.health()
print(f"Service status: {health['status']}")
print(f"Model loaded: {health['model_loaded']}")

# 2. Check metrics
metrics = client.metrics()
print(f"Failed predictions: {metrics['failed_predictions']}")
print(f"Success rate: {metrics['success_rate_percent']:.1f}%")

# 3. Try a prediction
result = client.predict("test.jpg")
if 'error' in result:
    print(f"Error: {result['error']}")
    if 'detail' in result:
        print(f"Detail: {result['detail']}")
```

---

All examples are copy-paste ready! Choose the language/framework that works best for your integration.
