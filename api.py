from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from pathlib import Path
import shutil
import json
from dotenv import load_dotenv
from main import process_image
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphMaster API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Bubble.io domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/extract-graphs")
async def extract_graphs(
    pdf_file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Extract graphs and figures from a PDF file
    Returns JSON with extracted data and metadata
    """
    try:
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded PDF
            pdf_path = os.path.join(temp_dir, pdf_file.filename)
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf_file.file, buffer)
            
            # Process the PDF and extract images
            # This will need to be implemented using pdffigures2 or similar
            # For now, we'll just return a placeholder response
            result = {
                "status": "success",
                "message": "PDF processed successfully",
                "data": {
                    "graphs": [],  # List of extracted graphs with their data
                    "metadata": {
                        "filename": pdf_file.filename,
                        "page_count": 0,  # Add actual page count
                    }
                }
            }
            
            return result

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-image")
async def process_single_image(
    image_file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Process a single image to extract graph data
    Returns JSON with extracted data points and metadata
    """
    try:
        # Validate file type
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be an image (PNG, JPG, or JPEG)")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            image_path = os.path.join(temp_dir, image_file.filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            
            # Process the image using existing GraphMaster functionality
            result = process_image(image_path)
            
            return result

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 