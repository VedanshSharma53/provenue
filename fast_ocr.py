#!/usr/bin/env python3
"""
FAST ENHANCED OCR SERVICE - SPEED OPTIMIZED
Reduced processing strategies for faster results while maintaining accuracy
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
import pytesseract
import logging
import time
import os
import re
from typing import Dict, List, Tuple, Optional

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastOCRService:
    def __init__(self):
        """Initialize fast OCR engines"""
        logger.info("🚀 Initializing Fast OCR Service...")
        
        # Initialize EasyOCR
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✅ EasyOCR initialized")
        except Exception as e:
            logger.error(f"❌ EasyOCR failed: {e}")
            self.easyocr_reader = None
        
        # Test Tesseract
        try:
            pytesseract.image_to_string(Image.new('RGB', (100, 100), 'white'))
            logger.info("✅ Tesseract available")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract not available: {e}")
    
    def fast_preprocessing(self, image_path: str) -> List[np.ndarray]:
        """
        Fast preprocessing - only the most effective strategies
        """
        try:
            # Load image
            original = cv2.imread(image_path)
            if original is None:
                raise ValueError("Could not load image")
            
            processed_images = []
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            # Strategy 1: CLAHE (most effective from tests)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(gray)
            processed_images.append(("clahe", clahe_image))
            
            # Strategy 2: Histogram equalization (second most effective)
            equalized = cv2.equalizeHist(gray)
            processed_images.append(("histogram_equalized", equalized))
            
            # Strategy 3: Enhanced contrast (good for low quality images)
            pil_image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            high_contrast = contrast_enhancer.enhance(2.0)
            enhanced_array = cv2.cvtColor(np.array(high_contrast), cv2.COLOR_RGB2BGR)
            enhanced_gray = cv2.cvtColor(enhanced_array, cv2.COLOR_BGR2GRAY)
            processed_images.append(("enhanced_contrast", enhanced_gray))
            
            return processed_images
            
        except Exception as e:
            logger.error(f"❌ Fast preprocessing failed: {e}")
            return [("original", cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))]
    
    def fast_tesseract_extract(self, image: np.ndarray) -> Optional[Dict]:
        """Fast Tesseract extraction with only best configs"""
        try:
            pil_image = Image.fromarray(image) if len(image.shape) == 2 else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Only use the most effective config
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/.- '
            
            text = pytesseract.image_to_string(pil_image, config=config)
            
            if text.strip():
                return {
                    'text': self.clean_text(text),
                    'confidence': 0.8,  # Assume good confidence for speed
                    'method': 'tesseract_fast',
                    'length': len(text.strip())
                }
            return None
            
        except Exception:
            return None
    
    def fast_easyocr_extract(self, image: np.ndarray) -> Optional[Dict]:
        """Fast EasyOCR extraction"""
        if self.easyocr_reader is None:
            return None
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Single EasyOCR pass
            ocr_result = self.easyocr_reader.readtext(image_rgb, detail=1)
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in ocr_result:
                if confidence > 0.1:
                    text_parts.append(text)
                    confidences.append(confidence)
            
            if text_parts:
                combined_text = ' '.join(text_parts)
                avg_confidence = sum(confidences) / len(confidences)
                return {
                    'text': self.clean_text(combined_text),
                    'confidence': avg_confidence,
                    'method': 'easyocr_fast',
                    'length': len(combined_text)
                }
            
            return None
            
        except Exception:
            return None
    
    def extract_text_fast(self, image_path: str) -> Dict:
        """
        Fast text extraction using reduced strategies
        """
        start_time = time.time()
        
        # Get only the most effective preprocessed versions
        processed_images = self.fast_preprocessing(image_path)
        
        all_results = []
        
        # Test each preprocessing strategy with both OCR engines
        for strategy_name, processed_image in processed_images:
            if processed_image is None:
                continue
            
            # Try EasyOCR first (usually faster)
            easyocr_result = self.fast_easyocr_extract(processed_image)
            if easyocr_result:
                easyocr_result['preprocessing'] = strategy_name
                all_results.append(easyocr_result)
            
            # Try Tesseract
            tesseract_result = self.fast_tesseract_extract(processed_image)
            if tesseract_result:
                tesseract_result['preprocessing'] = strategy_name
                all_results.append(tesseract_result)
            
            # If we have a good result, stop early for speed
            if all_results and max(r['confidence'] for r in all_results) > 0.7:
                break
        
        # Return best result or empty if none found
        if not all_results:
            return {
                'text': '',
                'confidence': 0.0,
                'method': 'none',
                'preprocessing': 'failed',
                'total_methods_tried': 0,
                'processing_time': time.time() - start_time,
                'all_results': []
            }
        
        # Get best result
        best_result = max(all_results, key=lambda x: (x['confidence'], x['length']))
        
        return {
            'text': best_result['text'],
            'confidence': best_result['confidence'],
            'method': best_result['method'],
            'preprocessing': best_result['preprocessing'],
            'total_methods_tried': len(all_results),
            'processing_time': time.time() - start_time,
            'all_results': all_results[:3]  # Top 3 for debugging
        }
    
    def clean_text(self, text: str) -> str:
        """Fast text cleaning"""
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^A-Z0-9/\-\.\s\n:()]+", " ", text)
        
        # Fix spacing
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

# Global fast OCR service
fast_ocr_service = FastOCRService()

def extract_text_fast(image_path: str) -> str:
    """Main function for fast text extraction"""
    result = fast_ocr_service.extract_text_fast(image_path)
    return result['text']

def get_fast_ocr_results(image_path: str) -> Dict:
    """Get fast OCR results"""
    return fast_ocr_service.extract_text_fast(image_path)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = get_fast_ocr_results(image_path)
            print(f"Text: {result['text']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Time: {result['processing_time']:.2f}s")
        else:
            print(f"Error: File {image_path} not found")
    else:
        print("Usage: python fast_ocr.py <image_path>")
