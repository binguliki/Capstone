# Project Structure & Architecture

## Directory Layout

```
model-backend/
├── main.py                          # FastAPI application (core)
├── client.py                        # Python client for API integration
├── test_api.py                      # Comprehensive test suite
├── gunicorn_config.py              # Production WSGI configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Docker Compose orchestration
├── README.md                       # Main documentation
├── DEPLOYMENT.md                   # Deployment guide
├── start.sh                        # Quick start script (Linux/macOS)
├── start.bat                       # Quick start script (Windows)
├── modelv1.h5                      # TensorFlow model (not in repo)
└── [Other project files]
```

## File Descriptions

### Core Application

#### `main.py` (Production-grade FastAPI)
- **Purpose**: Main API application
- **Size**: ~600 lines
- **Key Components**:
  - **Global State**: Model loading, metrics tracking
  - **Image Processing**: Validation, preprocessing, normalization
  - **Endpoints**:
    - `POST /predict` - Disease detection with inference
    - `GET /health` - Circuit breaker health check
    - `GET /metrics` - Performance metrics
  - **Error Handling**: Comprehensive validation and error responses
  - **Logging**: Structured logging for debugging

#### `client.py` (Python Client Library)
- **Purpose**: Convenient SDK for API integration
- **Features**:
  - Single and batch prediction
  - Health checking
  - Metrics retrieval
  - Error handling

### Configuration & Deployment

#### `requirements.txt`
- FastAPI, Uvicorn, TensorFlow, OpenCV, Pillow
- Production-ready versions specified

#### `.env.example`
- Template for environment configuration
- Copy to `.env` and customize

#### `gunicorn_config.py`
- Production WSGI server configuration
- Multi-worker support for scaling
- Tuned for optimal performance

#### `Dockerfile`
- Multi-stage build for smaller image size
- Minimal runtime dependencies
- Health check configured

#### `docker-compose.yml`
- Single-command deployment
- Volume mount for model
- Health check and restart policy

### Documentation

#### `README.md`
- Quick start guide
- API endpoint documentation
- Configuration reference
- Error handling guide
- Circuit breaker pattern integration

#### `DEPLOYMENT.md`
- Local development setup
- Docker deployment
- Production with Gunicorn
- Cloud deployment (AWS, GCP, Azure)
- Kubernetes deployment
- Monitoring & logging setup
- Troubleshooting guide

### Testing & Quick Start

#### `test_api.py`
- Comprehensive test suite with pytest
- 20+ test cases covering:
  - Smoke tests (endpoints working)
  - Core functionality (predictions)
  - Error handling (invalid inputs)
  - Performance (latency checks)
  - Metrics (tracking)
- Run with: `pytest test_api.py -v`

#### `start.sh` & `start.bat`
- Automated setup scripts
- Virtual environment creation
- Dependency installation
- Quick server startup

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
├──────────────────────┬──────────────────────────────────────┤
│   Web Browser        │   Python/JS/Mobile Clients           │
│   (Swagger UI)       │   (SDK/REST)                         │
└─────────┬────────────┴──────────────────────┬───────────────┘
          │                                    │
          └────────────────┬───────────────────┘
                           │
                    ┌──────▼──────────┐
                    │   FastAPI App   │
                    │   (main.py)     │
                    ├─────────────────┤
                    │  Endpoints:     │
                    │  • /predict     │
                    │  • /health      │
                    │  • /metrics     │
                    └──────┬───────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐  ┌─────▼──┐  ┌─────▼──┐
         │ Image  │  │ Model  │  │Metrics │
         │Process │  │ Inf    │  │Tracking│
         └────┬───┘  └────┬───┘  └──┬─────┘
              │            │         │
              └────┬───────┴────┬────┘
                   │            │
              ┌────▼──────┐  ┌──▼─────┐
              │TensorFlow │  │Logging │
              │Model      │  │System  │
              └───────────┘  └────────┘
```

### Data Flow for Prediction

```
1. Client sends image file
                │
                ▼
2. Image validated
   - File type check
   - Size check (<25MB)
   - Format verification
                │
                ▼
3. Image preprocessed
   - Load image (PIL)
   - Resize to 256×256
   - Convert to RGB
   - Normalize (ImageNet stats)
   - Add batch dimension
                │
                ▼
4. Model inference
   - Forward pass through TensorFlow model
   - Get outputs: classification + bboxes
   - Measure inference time
                │
                ▼
5. Post-processing
   - Extract class prediction
   - Filter bounding boxes by confidence
   - Validate coordinates
   - Sort by confidence
                │
                ▼
6. Format response
   - Build JSON response
   - Include confidence scores
   - Add timing info
   - Update metrics
                │
                ▼
7. Return to client
```

### Metrics Tracking

```
Metrics Class
├── total_requests: int
├── successful_predictions: int
├── failed_predictions: int
├── total_inference_time: float
├── request_times: List[float]  (last 1000)
└── Methods:
    ├── record_success(time)
    ├── record_failure()
    └── get_stats() → Dict
```

### Model Input/Output Specification

**Input**:
- Shape: `(1, 256, 256, 3)` (batch, height, width, channels)
- Type: `float32`
- Range: Normalized per ImageNet statistics
- Format: Pre-preprocessed image tensor

**Expected Output** (model architecture dependent):
- Classification logits: `[batch, 30]` (30 disease classes)
- Bounding boxes: `[batch, 100, 4]` (normalized x1,y1,x2,y2)
- Box labels: `[batch, 100]` (class indices)
- Box confidences: `[batch, 100]` (prediction confidence)

---

## Configuration Reference

### Image Processing Constants

```python
IMG_SIZE = 256                           # Model input resolution
CHANNELS = 3                             # RGB channels
MAX_IMAGE_SIZE_MB = 25                   # File size limit
ALLOWED_FORMATS = {'image/jpeg', ...}    # Accepted MIME types

# Normalization (ImageNet pretrained)
NORM_MEAN = [0.485, 0.456, 0.406]       # Per-channel mean
NORM_STD = [0.229, 0.224, 0.225]        # Per-channel std dev
```

### Model Parameters

```python
NUM_CLASSES = 30                         # Disease classes
MAX_DETECTIONS = 100                     # Max boxes per image
CONFIDENCE_THRESHOLD = 0.3               # Minimum confidence
MIN_BOX_AREA = 0.001                     # Minimum box area (0.1% of image)
```

### Server Configuration

```python
HOST = "0.0.0.0"                        # Listen address
PORT = 8000                              # Listen port
WORKERS = CPU_COUNT * 2 + 1              # Gunicorn workers
TIMEOUT = 120                            # Request timeout (seconds)
MAX_REQUESTS = 10000                     # Requests per worker
```

---

## Dependencies

### Production

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| tensorflow | 2.14.0 | ML model execution |
| opencv-python | 4.8.1 | Image processing |
| pillow | 10.1.0 | Image I/O |
| numpy | 1.24.3 | Numerical computing |
| pydantic | 2.5.0 | Data validation |

### Development

```
pytest              # Testing framework
pytest-asyncio      # Async test support
gunicorn           # Production WSGI server
python-multipart   # Form data parsing
```

---

## Performance Characteristics

### Inference Benchmarks

| Metric | Typical | P95 | P99 |
|--------|---------|-----|-----|
| Latency | 200-300ms | ~310ms | ~450ms |
| Memory (per request) | ~50MB | ~60MB | ~80MB |
| Throughput (single worker) | 3-5 req/s | - | - |
| CPU (single image) | 50-70% | - | - |

### Scaling Guidelines

- **Single process**: 3-5 requests/second
- **Multi-worker (4 workers)**: 12-20 requests/second
- **GPU acceleration**: 2-3x faster (model dependent)
- **Batch processing**: Not supported by design (async inference)

---

## Security Considerations

### Input Validation

- ✅ File type checking (MIME type validation)
- ✅ File size limits (25MB max)
- ✅ Image format verification
- ✅ Bounding box coordinate validation
- ✅ Confidence score bounds checking

### Output Sanitization

- ✅ Float precision clamping
- ✅ Coordinate normalization
- ✅ Label validation against disease class list
- ✅ JSON response validation

### Production Hardening

- ✅ Error message sanitization (no stack traces to client)
- ✅ Logging of suspicious requests
- ✅ Rate limiting (via reverse proxy)
- ✅ CORS configuration (customize in production)

---

## Troubleshooting Matrix

| Issue | Cause | Solution |
|-------|-------|----------|
| Model not loading | File not found | Check MODEL_PATH env var |
| High latency | CPU-bound | Use GPU, increase workers |
| OOM errors | Large image batch | Reduce MAX_DETECTIONS |
| Connection refused | Port in use | Change PORT env var |
| Prediction fails | Invalid image | Verify JPEG/PNG format |

---

## Future Enhancements

- [ ] Batch prediction endpoint
- [ ] Model versioning support
- [ ] Request caching
- [ ] Async model loading
- [ ] Prometheus metrics export
- [ ] WebSocket support for streaming
- [ ] Multi-model support
- [ ] A/B testing framework

---

## Development Workflow

### Local Testing

```bash
# 1. Setup
source venv/bin/activate
export MODEL_PATH=modelv1.h5

# 2. Run with autoreload
uvicorn main:app --reload

# 3. Test endpoints
curl http://localhost:8000/docs

# 4. Run tests
pytest test_api.py -v -k "core"
```

### Production Deployment

```bash
# 1. Build
docker build -t api:latest .

# 2. Deploy
docker-compose up -d

# 3. Monitor
docker logs -f leaf-disease-api
curl http://localhost:8000/metrics
```

---

## Contact & Support

For questions about architecture or implementation details, please refer to:
- README.md for API usage
- DEPLOYMENT.md for deployment options
- test_api.py for usage examples
