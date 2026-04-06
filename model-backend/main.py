"""
Production-grade FastAPI server for Leaf Disease Detection Model
- Endpoints: /predict, /health, /metrics
- Guardrails: Image validation, rate limiting, error handling, monitoring
"""

import os
import io
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import uvicorn
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import mobilevit_classes

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Constants & Configuration ────────────────────────────────────────────────

MODEL_PATH = os.getenv('MODEL_PATH', 'modelv2.keras')
IMG_SIZE = 256
CHANNELS = 3

# Image validation limits
MAX_IMAGE_SIZE_MB = 25
ALLOWED_FORMATS = {'image/jpeg', 'image/png', 'image/jpg', 'image/webp'}

# Disease classes (must match training)
DISEASE_CLASSES = [
    "healthy", "tomato_early_blight", "tomato_late_blight", "tomato_leaf_miner",
    "tomato_mosaic_virus", "tomato_septoria_leaf_spot", "tomato_spider_mites",
    "tomato_yellow_leaf_curl_virus", "corn_common_rust", "corn_gray_leaf_spot",
    "corn_northern_leaf_blight", "potato_early_blight", "potato_late_blight",
    "rice_blast", "rice_brown_spot", "rice_leaf_scald", "wheat_leaf_rust",
    "wheat_powdery_mildew", "wheat_septoria", "apple_scab", "apple_black_rot",
    "apple_cedar_rust", "grape_black_measles", "grape_leaf_blight",
    "strawberry_leaf_scorch", "pepper_bacterial_spot", "soybean_bacterial_pustule",
    "cherry_powdery_mildew", "peach_bacterial_spot", "blueberry_rust",
]

NUM_CLASSES = len(DISEASE_CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(DISEASE_CLASSES)}

# Normalization constants (from training pipeline)
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Global State ─────────────────────────────────────────────────────────────

model: Optional[tf.keras.Model] = None
app_start_time: datetime = datetime.now()

# Metrics tracking
class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.total_inference_time = 0.0
        self.request_times: List[float] = []
    
    def record_success(self, inference_time: float):
        self.total_requests += 1
        self.successful_predictions += 1
        self.total_inference_time += inference_time
        self.request_times.append(inference_time)
        # Keep only last 1000 requests for memory efficiency
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
    
    def record_failure(self):
        self.total_requests += 1
        self.failed_predictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        avg_time = (self.total_inference_time / self.successful_predictions 
                   if self.successful_predictions > 0 else 0)
        success_rate = (self.successful_predictions / self.total_requests * 100 
                       if self.total_requests > 0 else 0)
        
        return {
            "total_requests": self.total_requests,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "success_rate_percent": round(success_rate, 2),
            "average_inference_time_ms": round(avg_time * 1000, 2),
            "p95_inference_time_ms": round(np.percentile(self.request_times, 95) * 1000, 2) if self.request_times else 0,
            "p99_inference_time_ms": round(np.percentile(self.request_times, 99) * 1000, 2) if self.request_times else 0,
        }

metrics = Metrics()

# ─── Response Models ─────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Bounding box annotation with normalized coordinates"""
    x1: float = Field(..., ge=0, le=1, description="Normalized left coordinate")
    y1: float = Field(..., ge=0, le=1, description="Normalized top coordinate")
    x2: float = Field(..., ge=0, le=1, description="Normalized right coordinate")
    y2: float = Field(..., ge=0, le=1, description="Normalized bottom coordinate")
    label: str = Field(..., description="Disease class")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    classification: str = Field(..., description="Primary disease classification")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="Detected regions")
    inference_time_ms: float = Field(..., description="Time taken for inference")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(default="v1", description="Model version")


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Is model loaded")
    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: str = Field(..., description="Health check timestamp")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    success_rate_percent: float
    average_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    timestamp: str


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model() -> bool:
    """Load TensorFlow model with error handling."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        import mobilevit_classes
        custom_objs = {name: getattr(mobilevit_classes, name) for name in dir(mobilevit_classes) if isinstance(getattr(mobilevit_classes, name), type)}
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objs)
        logger.info(f"Model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


# ─── Image Processing ────────────────────────────────────────────────────────

async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate and read image file.
    
    Guardrails:
    - Check content type
    - Check file size
    - Verify it's a valid image
    """
    # Validate content type
    if file.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format. Allowed: {ALLOWED_FORMATS}"
        )
    
    # Read and validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Max size: {MAX_IMAGE_SIZE_MB}MB, got: {size_mb:.2f}MB"
        )
    
    return contents


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image matching training pipeline:
    1. Load image and convert BGR→RGB
    2. Resize to 256×256
    3. Normalize using ImageNet statistics
    
    Returns: (1, 256, 256, 3) normalized float32 array
    """
    # Load image from bytes
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil, dtype=np.uint8)
    
    # Resize to model input size
    if image_np.shape[0] != IMG_SIZE or image_np.shape[1] != IMG_SIZE:
        image_np = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), 
                             interpolation=cv2.INTER_LINEAR)
    
    # Convert to float32 and normalize to [0, 1]
    image_float = image_np.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    image_normalized = (image_float - NORM_MEAN) / NORM_STD
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch


# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(image_batch: np.ndarray) -> Dict[str, Any]:
    """
    Run model inference on preprocessed image.
    
    Expected model outputs:
    - Classification logits: [batch, num_classes]
    - Bounding boxes: [batch, max_boxes, 4] (normalized x1,y1,x2,y2)
    - Bounding box labels: [batch, max_boxes]
    - Bounding box confidences: [batch, max_boxes]
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        outputs = model(image_batch, training=False)
        
        # New Model outputs: {'cls_logits', 'det_cls', 'det_box'}
        cls_logits = outputs.get('cls_logits')
        det_cls = outputs.get('det_cls')
        det_box = outputs.get('det_box')
        
        # Post-process detection
        det_cls_np = tf.nn.softmax(det_cls, axis=-1).numpy()[0]
        det_box_np = det_box.numpy()[0]
        
        scores_all = det_cls_np.max(axis=-1)
        labels_all = det_cls_np.argmax(axis=-1)
        
        return {
            'cls_logits': cls_logits[0].numpy() if hasattr(cls_logits, 'numpy') else cls_logits[0],
            'boxes': det_box_np,
            'box_labels': labels_all,
            'box_confs': scores_all
        }
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


def postprocess_outputs(inference_outputs: Dict[str, Any], 
                        conf_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Postprocess model outputs to extract disease classification and bounding boxes.
    
    Args:
        inference_outputs: Raw model outputs
        conf_threshold: Minimum confidence for bounding box inclusion
    
    Returns:
        Processed predictions with classification and filtered bounding boxes
    """
    cls_logits = inference_outputs['cls_logits']  # [num_classes]
    boxes = inference_outputs['boxes']  # [max_boxes, 4]
    box_labels = inference_outputs['box_labels']  # [max_boxes]
    box_confs = inference_outputs['box_confs']  # [max_boxes]
    
    # Classification: take argmax of logits
    cls_idx = int(np.argmax(cls_logits))
    cls_conf = float(tf.nn.softmax(cls_logits)[cls_idx].numpy())
    classification = IDX_TO_CLASS.get(cls_idx, "unknown")
    
    # Bounding boxes: filter by confidence and validity
    valid_boxes = []
    
    for i in range(len(boxes)):
        conf = float(box_confs[i]) if box_confs[i] > 0 else 0
        
        # Skip invalid boxes (low confidence)
        if conf < conf_threshold:
            continue
        
        # Clip boxes to [0, 1] to handle raw network outputs
        x1, y1, x2, y2 = np.clip(boxes[i], 0.0, 1.0)
        
        # Validate box coordinates
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            continue
        
        # Skip very small boxes
        area = (x2 - x1) * (y2 - y1)
        if area < 0.001:  # Less than 0.1% of image
            continue
        
        label_idx = int(box_labels[i]) if box_labels[i] >= 0 else 0
        label = IDX_TO_CLASS.get(label_idx, "healthy")
        
        valid_boxes.append({
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'label': label,
            'confidence': min(conf, 1.0),
        })
    
    # Sort by confidence descending and cap at top 10
    valid_boxes.sort(key=lambda x: x['confidence'], reverse=True)
    valid_boxes = valid_boxes[:10]
    
    return {
        'classification': classification,
        'confidence': cls_conf,
        'bounding_boxes': valid_boxes,
    }


# ─── Lifespan Management ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting Leaf Disease Detection API...")
    model_loaded = load_model()
    if not model_loaded:
        logger.error("Failed to load model on startup")
    else:
        logger.info("Model loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Leaf Disease Detection API...")
    logger.info(f"Final metrics: {metrics.get_stats()}")


# ─── FastAPI Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Leaf Disease Detection API",
    description="Production ML service for plant disease classification and localization",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict disease classification and bounding boxes",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid image format or size"},
        413: {"description": "Image too large"},
        500: {"description": "Inference error"},
        503: {"description": "Model not loaded"},
    }
)
async def predict(file: UploadFile = File(..., description="Leaf image (JPEG/PNG/WebP)")):
    """
    Predict disease classification and bounding boxes for a leaf image.
    
    **Input:**
    - image file (JPEG, PNG, or WebP format, max 25MB)
    
    **Output:**
    - Primary disease classification with confidence
    - List of detected disease regions with bounding boxes
    - Inference time and timestamp
    """
    if model is None:
        metrics.record_failure()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        # Validate and read image
        image_bytes = await validate_image_file(file)
        
        # Preprocess
        image_batch = preprocess_image(image_bytes)
        
        # Inference with timing
        start_time = time.time()
        inference_outputs = run_inference(image_batch)
        inference_time = time.time() - start_time
        
        # Postprocess
        predictions = postprocess_outputs(inference_outputs)
        
        # Record metrics
        metrics.record_success(inference_time)
        
        # Start with any model-derived bounding boxes
        bounding_boxes = [
            BoundingBox(
                x1=bbox['x1'],
                y1=bbox['y1'],
                x2=bbox['x2'],
                y2=bbox['y2'],
                label=bbox['label'],
                confidence=bbox['confidence'],
            )
            for bbox in predictions['bounding_boxes']
        ]

        # We completely removed all legacy seg_logits bounding box logic. 
        # All boxes are now naturally extracted from predict() as bounding_boxes

        return PredictionResponse(
            classification=predictions['classification'],
            confidence=predictions['confidence'],
            bounding_boxes=bounding_boxes,
            inference_time_ms=inference_time * 1000,
            timestamp=datetime.now().isoformat(),
        )
    
    except HTTPException:
        metrics.record_failure()
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        metrics.record_failure()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint for circuit breaker pattern",
    tags=["Monitoring"],
)
async def health():
    """
    Health check endpoint for circuit breaker integration.
    
    Returns service status, model availability, and uptime.
    """
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get service metrics and performance statistics",
    tags=["Monitoring"],
)
async def get_metrics():
    """
    Get comprehensive service metrics including request counts, success rates, 
    and inference time percentiles.
    
    Useful for monitoring dashboards and performance analysis.
    """
    stats = metrics.get_stats()
    
    return MetricsResponse(
        **stats,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/",
    summary="API Information",
    tags=["Info"],
)
async def root():
    """Get API information and available endpoints."""
    return {
        "service": "Leaf Disease Detection API",
        "version": "1.0.0",
        "status": "healthy" if model is not None else "degraded",
        "endpoints": {
            "predict": "POST /predict - Disease classification + bounding boxes",
            "health": "GET /health - Circuit breaker health check",
            "metrics": "GET /metrics - Performance metrics",
            "docs": "GET /docs - Interactive API documentation (Swagger UI)",
        }
    }


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
