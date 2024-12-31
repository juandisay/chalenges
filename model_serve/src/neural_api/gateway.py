import json
import torch
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from tempfile import NamedTemporaryFile
from enum import Enum

from neural_api.config import Config
from neural_network.config import Config as neural_network_config
from neural_network.predictor import (
    DogPredictor,
    PredictionError,
    ImageProcessingError
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageFormat(Enum):
    """Supported image formats"""
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    filename: str
    predictions: List[dict]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str

class ModelManager:
    def __init__(self):
        self.predictor: Optional[DogPredictor] = None
        self.config: Optional[Config] = None
    
    def initialize(self):
        """Initialize model and predictor - synchronous version"""
        try:
            # Load configuration
            self.config = neural_network_config()
            
            # Initialize predictor with config
            self.predictor = DogPredictor(self.config)
            
            logger.info(
                f"Model initialized successfully (Device: {self.config.get_device()})"
            )
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def get_predictor(self) -> DogPredictor:
        """Get initialized predictor"""
        if not self.predictor:
            raise RuntimeError("Predictor not initialized")
        return self.predictor

class APIRouter:
    """Handles API route definitions and logic"""
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    async def predict_image(
        self,
        file: UploadFile = File(...),
        predictor: DogPredictor = Depends(lambda: model_manager.get_predictor())
    ) -> PredictionResponse:
        """Handle image prediction request"""
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Validate extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in {fmt.value for fmt in ImageFormat}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported: {[fmt.value for fmt in ImageFormat]}"
            )
        
        try:
            # Save temporary file
            with NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = Path(temp_file.name)
            
            try:
                # Make prediction
                result = predictor.predict(temp_file_path, visualize=False)
                return PredictionResponse(**result)
                
            finally:
                # Clean up
                if temp_file_path.exists():
                    temp_file_path.unlink()
                    
        except (PredictionError, ImageProcessingError) as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def health_check(self) -> HealthResponse:
        """Handle health check request"""
        predictor = self.model_manager.predictor
        return HealthResponse(
            status="healthy",
            model_loaded=predictor is not None,
            device=str(self.model_manager.config.get_device())
        )

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Initialize FastAPI
    app = FastAPI(
        title="Neural API",
        description="API for classifying Dog breeds from images",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Initialize router
    router = APIRouter(model_manager)
    
    @app.on_event("startup")
    def startup_event():
        """Initialize model on startup - changed to sync"""
        model_manager.initialize()
    
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint"""
        return {
            "status": "ok",
            "message": "Neural API is running"
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        predictor: DogPredictor = Depends(lambda: model_manager.get_predictor())
    ):
        return await router.predict_image(file, predictor)
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return await router.health_check()
    
    return app
