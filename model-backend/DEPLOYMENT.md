# Deployment Guide

Comprehensive deployment instructions for Leaf Disease Detection API in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production with Gunicorn](#production-with-gunicorn)
4. [Cloud Deployment](#cloud-deployment)
5. [Kubernetes](#kubernetes)
6. [Monitoring & Logging](#monitoring--logging)
7. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Clone/navigate to project directory
cd model-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and set MODEL_PATH
# MODEL_PATH=./modelv1.h5
```

### Run Server

```bash
python main.py
```

Server accessible at: `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

### Development with Auto-Reload

```bash
pip install uvicorn[standard]
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

### Quick Start

```bash
# Build and run with Docker Compose
docker-compose up --build

# In another terminal, test the API
curl http://localhost:8000/health
```

### Manual Docker Build

```bash
# Build image
docker build -t leaf-disease-api:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/modelv1.h5:/app/modelv1.h5:ro \
  -e MODEL_PATH=/app/modelv1.h5 \
  --name leaf-disease-api \
  leaf-disease-api:latest

# View logs
docker logs leaf-disease-api

# Stop container
docker stop leaf-disease-api
```

### Docker Compose with Custom Configuration

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    environment:
      - LOG_LEVEL=debug
      - WORKERS=8
    ports:
      - "8000:8000"
    healthcheck:
      start_period: 20s
```

---

## Production with Gunicorn

Gunicorn provides multi-worker support for production deployments.

### Installation

```bash
pip install gunicorn uvicorn[standard]
```

### Run with Gunicorn

```bash
# Basic
gunicorn -c gunicorn_config.py main:app

# With specific worker count
WORKERS=8 gunicorn -c gunicorn_config.py main:app

# With logging
gunicorn -c gunicorn_config.py main:app \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

### Systemd Service (Linux)

Create `/etc/systemd/system/leaf-disease-api.service`:

```ini
[Unit]
Description=Leaf Disease Detection API
After=network.target

[Service]
Type=notify
User=appuser
WorkingDirectory=/opt/leaf-disease-api
ExecStart=/opt/leaf-disease-api/venv/bin/gunicorn \
  -c gunicorn_config.py \
  --bind 0.0.0.0:8000 \
  main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable leaf-disease-api
sudo systemctl start leaf-disease-api
sudo systemctl status leaf-disease-api
```

---

## Cloud Deployment

### AWS EC2

```bash
# 1. SSH into instance
ssh -i key.pem ec2-user@<instance-ip>

# 2. Install dependencies
sudo yum update -y
sudo yum install -y python3 python3-pip git

# 3. Clone repository
git clone <repo-url> /opt/leaf-disease-api
cd /opt/leaf-disease-api

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install packages
pip install -r requirements.txt

# 6. Copy model
# Ensure modelv1.h5 is available in /opt/leaf-disease-api/

# 7. Run with Gunicorn
gunicorn -c gunicorn_config.py main:app --bind 0.0.0.0:8000
```

#### Security Group Configuration

- Inbound: Allow port 8000 from your client IPs
- Outbound: Allow all

### Google Cloud Run

```bash
# 1. Create requirements.txt with gunicorn
echo "gunicorn" >> requirements.txt

# 2. Adjust main.py for Cloud Run (optional port binding)
# Cloud Run sets PORT environment variable

# 3. Deploy
gcloud run deploy leaf-disease-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=/workspace/modelv1.h5 \
  --max-instances 10 \
  --memory 4Gi
```

### Azure Container Instances

```bash
# 1. Build and push image
az acr build --registry <registry-name> --image leaf-disease-api:latest .

# 2. Deploy container
az container create \
  --resource-group <group-name> \
  --name leaf-disease-api \
  --image <registry-name>.azurecr.io/leaf-disease-api:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label leaf-disease-api
```

---

## Kubernetes

### Deployment Manifest

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: leaf-disease-api
  labels:
    app: leaf-disease-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: leaf-disease-api
  template:
    metadata:
      labels:
        app: leaf-disease-api
    spec:
      containers:
      - name: api
        image: leaf-disease-api:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        
        env:
        - name: MODEL_PATH
          value: /models/modelv1.h5
        - name: LOG_LEVEL
          value: info
        - name: WORKERS
          value: "4"
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        volumeMounts:
        - name: model
          mountPath: /models
          readOnly: true
      
      volumes:
      - name: model
        configMap:
          name: model-config
---
apiVersion: v1
kind: Service
metadata:
  name: leaf-disease-api-service
spec:
  selector:
    app: leaf-disease-api
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: leaf-disease-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: leaf-disease-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# 1. Load model into ConfigMap
kubectl create configmap model-config \
  --from-file=modelv1.h5=./modelv1.h5

# 2. Apply deployment
kubectl apply -f deployment.yaml

# 3. Check status
kubectl get pods
kubectl get svc

# 4. Access service
kubectl port-forward svc/leaf-disease-api-service 8000:80
```

### Helm Chart (Optional)

Create `Chart.yaml`:

```yaml
apiVersion: v2
name: leaf-disease-api
description: Leaf Disease Detection API Helm Chart
type: application
version: 1.0.0
appVersion: "1.0.0"
```

Create `values.yaml`:

```yaml
replicaCount: 3

image:
  repository: leaf-disease-api
  tag: latest
  pullPolicy: IfNotPresent

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

service:
  type: LoadBalancer
  port: 80

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
```

---

## Monitoring & Logging

### Prometheus Metrics Integration

```python
# Add to main.py for Prometheus integration
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.get("/metrics-prometheus")
async def metrics_prometheus():
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

Configure JSON logging for production:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

### ELK Stack (Elasticsearch, Logstash, Kibana)

Filebeat configuration (`filebeat.yml`):

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /opt/leaf-disease-api/logs/*.log
  json.message_key: message
  json.keys_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "leaf-disease-api-%{+yyyy.MM.dd}"

logging.level: info
```

---

## Performance Tuning

### Image Preprocessing Optimization

```python
# Use TensorFlow's tf.image APIs for faster preprocessing
@tf.function
def preprocess_batch(image_batch):
    """Optimized batch preprocessing with graph mode."""
    image_batch = tf.image.resize(image_batch, [256, 256])
    image_batch = (image_batch - NORM_MEAN) / NORM_STD
    return image_batch
```

### Model Quantization

```python
# For faster inference, consider quantized models
import tensorflow as tf

model = tf.keras.models.load_model("modelv1.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

### GPU Acceleration

```bash
# With NVIDIA GPU support
docker run --gpus all -p 8000:8000 leaf-disease-api:latest
```

Dockerfile update:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
```

---

## Troubleshooting

### High Memory Usage

**Problem**: Container consuming too much memory

**Solutions**:
1. Reduce batch size or request timeout
2. Enable memory limits in Docker/Kubernetes
3. Monitor with `docker stats` or `kubectl top`

```bash
# Check container memory
docker stats leaf-disease-api

# Limit memory in docker-compose.yml
services:
  api:
    mem_limit: 4g
```

### Slow Inference

**Problem**: Predictions taking > 1 second

**Solutions**:
1. Verify GPU is being used (check with `nvidia-smi`)
2. Profile with TensorFlow Profiler
3. Reduce image size temporarily to test
4. Check system load (`top`, `htop`)

### Model Loading Failures

**Problem**: "Model file not found" error

**Solutions**:
```bash
# Verify model path and permissions
ls -la modelv1.h5

# Check volume mounts in Docker
docker inspect leaf-disease-api | grep -A 5 Mounts

# Verify Kubernetes ConfigMap
kubectl get configmap model-config
```

### Out of Memory (OOM) Errors

**Problem**: "Resource exhausted" error during inference

**Solutions**:
1. Reduce `MAX_DETECTIONS` in `main.py`
2. Increase available memory
3. Use GPU with TensorFlow-GPU

```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS
wmic OS get TotalVisibleMemorySize  # Windows
```

### Connection Refused

**Problem**: Cannot connect to API

**Solutions**:
```bash
# Verify server is running
curl http://localhost:8000/health

# Check port is open
netstat -tlnp | grep 8000  # Linux
lsof -i :8000  # macOS

# Verify firewall rules
sudo ufw status
```

---

## Backup & Recovery

### Model Backup

```bash
# Create backup
cp modelv1.h5 modelv1.h5.backup.$(date +%Y%m%d)

# Restore from backup
cp modelv1.h5.backup.20240115 modelv1.h5
```

### Database Backup (if using metrics database)

```bash
# For PostgreSQL (example)
pg_dump leaf_disease_db > backup.sql
psql leaf_disease_db < backup.sql
```

---

## Contact & Support

For deployment assistance or issues, please contact the development team.
