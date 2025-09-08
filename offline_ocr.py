#!/usr/bin/env python3
"""
OFFLINE OCR SERVICE - NO EXTERNAL APIS
Enterprise-grade document identification system for on-premise deployment
Compliant with Botswana DPA requirements and offline operation constraints
"""

import io
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
import logging
import time
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineOCRService:
    def __init__(self):
        """Initialize offline OCR engines"""
        logger.info("🚀 Initializing Offline OCR Service...")
        
        # Initialize EasyOCR (primary engine)
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU only for stability
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"❌ EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
        
        # Tesseract as backup
        try:
            # Test tesseract availability
            pytesseract.image_to_string(Image.new('RGB', (100, 100), 'white'))
            logger.info("✅ Tesseract OCR available as backup")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract not available: {e}")
    
    def preprocess_image(self, image_path: str, enhance_quality=True):
        """
        Advanced image preprocessing for better OCR accuracy
        Includes: denoising, deskewing, contrast enhancement, sharpening
        """
        start_time = time.time()
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = image_path
            
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to RGB for PIL processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            if enhance_quality:
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(2.0)
                
                # Convert back to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # Adaptive thresholding for better text contrast
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Deskew if needed (basic rotation correction)
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                # Only correct significant skew
                if abs(angle) > 0.5:
                    (h, w) = binary.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            processing_time = time.time() - start_time
            logger.info(f"🔧 Image preprocessing completed in {processing_time:.3f}s")
            
            return binary
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            # Return original image if preprocessing fails
            try:
                return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            except:
                return None
    
    def extract_text_easyocr(self, image):
        """Extract text using EasyOCR (primary method)"""
        try:
            if self.easyocr_reader is None:
                return None
            
            start_time = time.time()
            
            # EasyOCR expects RGB format
            if len(image.shape) == 2:  # Grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text with confidence scores
            results = self.easyocr_reader.readtext(image_rgb, detail=1)
            
            # Combine all text with confidence filtering
            text_parts = []
            total_confidence = 0
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low-confidence text
                    text_parts.append(text)
                    total_confidence += confidence
            
            extracted_text = ' '.join(text_parts)
            avg_confidence = total_confidence / len(results) if results else 0
            
            processing_time = time.time() - start_time
            logger.info(f"✅ EasyOCR extraction completed in {processing_time:.3f}s (confidence: {avg_confidence:.2f})")
            
            return {
                'text': extracted_text,
                'confidence': avg_confidence,
                'method': 'EasyOCR'
            }
            
        except Exception as e:
            logger.error(f"❌ EasyOCR extraction failed: {e}")
            return None
    
    def extract_text_tesseract(self, image):
        """Extract text using Tesseract (backup method)"""
        try:
            start_time = time.time()
            
            # Convert to PIL Image for tesseract
            if len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image, mode='L')
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text with confidence data
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ '
            
            # Get text
            extracted_text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # Get confidence data
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Tesseract extraction completed in {processing_time:.3f}s (confidence: {avg_confidence:.2f})")
            
            return {
                'text': extracted_text,
                'confidence': avg_confidence / 100,  # Normalize to 0-1
                'method': 'Tesseract'
            }
            
        except Exception as e:
            logger.error(f"❌ Tesseract extraction failed: {e}")
            return None
    
    def extract_text(self, image_path: str):
        """
        Main text extraction method with fallback strategy
        Uses EasyOCR first, falls back to Tesseract if needed
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            raise Exception("Failed to preprocess image")
        
        # Try EasyOCR first
        easyocr_result = self.extract_text_easyocr(processed_image)
        
        # Use EasyOCR if confidence is good
        if easyocr_result and easyocr_result['confidence'] > 0.6:
            total_time = time.time() - start_time
            logger.info(f"🎯 Using EasyOCR result (total time: {total_time:.3f}s)")
            return self.clean_text(easyocr_result['text'])
        
        # Fall back to Tesseract
        logger.info("🔄 Falling back to Tesseract OCR")
        tesseract_result = self.extract_text_tesseract(processed_image)
        
        if tesseract_result:
            total_time = time.time() - start_time
            logger.info(f"🎯 Using Tesseract result (total time: {total_time:.3f}s)")
            return self.clean_text(tesseract_result['text'])
        
        # If both fail, try EasyOCR result anyway
        if easyocr_result:
            total_time = time.time() - start_time
            logger.warning(f"⚠️ Using low-confidence EasyOCR result (total time: {total_time:.3f}s)")
            return self.clean_text(easyocr_result['text'])
        
        raise Exception("All OCR methods failed")
    
    def clean_text(self, text: str):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^A-Z0-9/\-\.\s\n]+", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

# Initialize global OCR service
ocr_service = OfflineOCRService()

# Export main function for compatibility
def extract_text_offline(image_path: str):
    """Main function for offline text extraction"""
    return ocr_service.extract_text(image_path)
