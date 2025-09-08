#!/usr/bin/env python3
"""
ENHANCED OFFLINE OCR SERVICE - IMPROVED ACCURACY
Advanced document identification system with multiple OCR strategies
Optimized for Botswana Omang and International Passports
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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

class EnhancedOCRService:
    def __init__(self):
        """Initialize enhanced OCR engines with multiple strategies"""
        logger.info("🚀 Initializing Enhanced OCR Service...")
        
        # Initialize EasyOCR with multiple readers
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"❌ EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
        
        # Test Tesseract availability
        try:
            pytesseract.image_to_string(Image.new('RGB', (100, 100), 'white'))
            logger.info("✅ Tesseract OCR available")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract not available: {e}")
    
    def advanced_image_preprocessing(self, image_path: str) -> List[np.ndarray]:
        """
        Create multiple preprocessed versions of the image for OCR
        Returns a list of processed images with different enhancement strategies
        """
        start_time = time.time()
        
        try:
            # Load original image
            original = cv2.imread(image_path)
            if original is None:
                raise ValueError("Could not load image")
            
            processed_images = []
            
            # Strategy 1: High contrast binary
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(("adaptive_threshold", adaptive_thresh))
            
            # Strategy 2: Otsu's thresholding
            _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("otsu_threshold", otsu_thresh))
            
            # Strategy 3: Morphological operations for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            processed_images.append(("morphological", morph))
            
            # Strategy 4: Enhanced contrast and sharpening
            # Convert to PIL for advanced enhancement
            pil_image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            high_contrast = contrast_enhancer.enhance(2.5)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(high_contrast)
            sharp_image = sharpness_enhancer.enhance(3.0)
            
            # Convert back to OpenCV format
            enhanced_array = cv2.cvtColor(np.array(sharp_image), cv2.COLOR_RGB2BGR)
            enhanced_gray = cv2.cvtColor(enhanced_array, cv2.COLOR_BGR2GRAY)
            processed_images.append(("enhanced_contrast", enhanced_gray))
            
            # Strategy 5: Histogram equalization
            equalized = cv2.equalizeHist(gray)
            processed_images.append(("histogram_equalized", equalized))
            
            # Strategy 6: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(gray)
            processed_images.append(("clahe", clahe_image))
            
            # Strategy 7: Bilateral filter for noise reduction while preserving edges
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("bilateral_filtered", bilateral_thresh))
            
            # Strategy 8: Resize for better OCR if image is too small
            height, width = gray.shape
            if min(height, width) < 600:
                scale_factor = 600 / min(height, width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                _, resized_thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(("resized_enhanced", resized_thresh))
            
            processing_time = time.time() - start_time
            logger.info(f"🔧 Advanced preprocessing created {len(processed_images)} variants in {processing_time:.3f}s")
            
            return processed_images
            
        except Exception as e:
            logger.error(f"❌ Advanced preprocessing failed: {e}")
            return [("original", cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))]
    
    def extract_text_with_multiple_configs(self, image: np.ndarray, method: str = "tesseract") -> List[Dict]:
        """
        Extract text using multiple Tesseract configurations
        """
        results = []
        
        if method == "tesseract":
            # Configuration 1: Standard
            config1 = r'--oem 3 --psm 6'
            
            # Configuration 2: Assume a single uniform block of text
            config2 = r'--oem 3 --psm 8'
            
            # Configuration 3: Treat image as a single text line
            config3 = r'--oem 3 --psm 7'
            
            # Configuration 4: Sparse text
            config4 = r'--oem 3 --psm 11'
            
            # Configuration 5: Single character
            config5 = r'--oem 3 --psm 10'
            
            # Configuration 6: With character whitelist for documents
            config6 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/.- '
            
            configs = [
                ("standard", config1),
                ("single_block", config2),
                ("single_line", config3),
                ("sparse_text", config4),
                ("single_char", config5),
                ("whitelist", config6)
            ]
            
            pil_image = Image.fromarray(image) if len(image.shape) == 2 else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            for config_name, config in configs:
                try:
                    text = pytesseract.image_to_string(pil_image, config=config)
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if text.strip():
                        results.append({
                            'text': self.clean_text(text),
                            'confidence': avg_confidence / 100,
                            'method': f'tesseract_{config_name}',
                            'length': len(text.strip())
                        })
                except Exception as e:
                    logger.warning(f"Config {config_name} failed: {e}")
        
        return results
    
    def extract_text_easyocr_enhanced(self, image: np.ndarray) -> Optional[Dict]:
        """Enhanced EasyOCR extraction with different parameters"""
        if self.easyocr_reader is None:
            return None
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Try different EasyOCR configurations
            results = []
            
            # Configuration 1: Standard
            try:
                ocr_result = self.easyocr_reader.readtext(image_rgb, detail=1)
                text_parts = []
                confidences = []
                
                for (bbox, text, confidence) in ocr_result:
                    if confidence > 0.1:  # Lower threshold for better recall
                        text_parts.append(text)
                        confidences.append(confidence)
                
                if text_parts:
                    combined_text = ' '.join(text_parts)
                    avg_confidence = sum(confidences) / len(confidences)
                    results.append({
                        'text': self.clean_text(combined_text),
                        'confidence': avg_confidence,
                        'method': 'easyocr_standard',
                        'length': len(combined_text)
                    })
            except Exception as e:
                logger.warning(f"EasyOCR standard config failed: {e}")
            
            # Configuration 2: Different width/height ratios
            try:
                # Resize image for better OCR
                height, width = image_rgb.shape[:2]
                if height > 0 and width > 0:
                    # Try different scales
                    for scale in [1.5, 2.0, 0.75]:
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        
                        ocr_result = self.easyocr_reader.readtext(resized, detail=1)
                        text_parts = []
                        confidences = []
                        
                        for (bbox, text, confidence) in ocr_result:
                            if confidence > 0.1:
                                text_parts.append(text)
                                confidences.append(confidence)
                        
                        if text_parts:
                            combined_text = ' '.join(text_parts)
                            avg_confidence = sum(confidences) / len(confidences)
                            results.append({
                                'text': self.clean_text(combined_text),
                                'confidence': avg_confidence,
                                'method': f'easyocr_scaled_{scale}',
                                'length': len(combined_text)
                            })
            except Exception as e:
                logger.warning(f"EasyOCR scaling failed: {e}")
            
            # Return best result
            if results:
                best_result = max(results, key=lambda x: x['confidence'] * x['length'])
                return best_result
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced EasyOCR failed: {e}")
            return None
    
    def extract_text_comprehensive(self, image_path: str) -> Dict:
        """
        Comprehensive text extraction using all strategies
        """
        start_time = time.time()
        logger.info(f"🔍 Starting comprehensive OCR on {os.path.basename(image_path)}")
        
        try:
            # Get multiple preprocessed versions
            processed_images = self.advanced_image_preprocessing(image_path)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Fallback to simple grayscale
            try:
                simple_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                processed_images = [("simple_grayscale", simple_image)]
            except:
                raise Exception(f"Cannot load image: {image_path}")
        
        all_results = []
        
        # Test each preprocessing strategy
        for strategy_name, processed_image in processed_images:
            if processed_image is None:
                continue
                
            logger.info(f"📝 Testing strategy: {strategy_name}")
            
            # Try EasyOCR on this variant
            try:
                easyocr_result = self.extract_text_easyocr_enhanced(processed_image)
                if easyocr_result:
                    easyocr_result['preprocessing'] = strategy_name
                    all_results.append(easyocr_result)
            except Exception as e:
                logger.warning(f"EasyOCR failed on {strategy_name}: {e}")
            
            # Try multiple Tesseract configs on this variant
            try:
                tesseract_results = self.extract_text_with_multiple_configs(processed_image, "tesseract")
                for result in tesseract_results:
                    result['preprocessing'] = strategy_name
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Tesseract failed on {strategy_name}: {e}")
        
        # Analyze and combine results
        if not all_results:
            logger.warning("No results from any OCR method - returning empty result")
            return {
                'text': '',
                'confidence': 0.0,
                'method': 'none',
                'preprocessing': 'failed',
                'total_methods_tried': 0,
                'processing_time': time.time() - start_time,
                'all_results': []
            }
        
        # Sort by confidence and text length
        all_results.sort(key=lambda x: (x['confidence'], x['length']), reverse=True)
        
        # Get the best result
        best_result = all_results[0]
        
        # Try to combine multiple good results for better coverage
        high_confidence_results = [r for r in all_results if r['confidence'] > 0.5]
        if len(high_confidence_results) > 1:
            # Combine texts from high-confidence results
            combined_texts = []
            for result in high_confidence_results[:3]:  # Top 3 results
                if result['text'] and result['text'] not in combined_texts:
                    combined_texts.append(result['text'])
            
            if combined_texts:
                combined_text = ' '.join(combined_texts)
                best_result['text'] = self.clean_text(combined_text)
                best_result['method'] = 'combined_best'
        
        total_time = time.time() - start_time
        
        return {
            'text': best_result['text'],
            'confidence': best_result['confidence'],
            'method': best_result['method'],
            'preprocessing': best_result['preprocessing'],
            'total_methods_tried': len(all_results),
            'processing_time': total_time,
            'all_results': all_results[:5]  # Top 5 for debugging
        }
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Replace common OCR errors
        replacements = {
            '0': 'O',  # Sometimes O is read as 0
            '1': 'I',  # Sometimes I is read as 1
            '5': 'S',  # Sometimes S is read as 5
            '8': 'B',  # Sometimes B is read as 8
        }
        
        # Apply replacements only if it makes sense in context
        # For now, keep original to avoid false corrections
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^A-Z0-9/\-\.\s\n:()]+", " ", text)
        
        # Fix common spacing issues
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove very short fragments
        words = text.split()
        words = [word for word in words if len(word) > 1 or word.isdigit()]
        
        return ' '.join(words)

# Global enhanced OCR service
enhanced_ocr_service = EnhancedOCRService()

def extract_text_enhanced(image_path: str) -> str:
    """Main function for enhanced offline text extraction"""
    result = enhanced_ocr_service.extract_text_comprehensive(image_path)
    return result['text']

def get_detailed_ocr_results(image_path: str) -> Dict:
    """Get detailed OCR results including all methods tried"""
    return enhanced_ocr_service.extract_text_comprehensive(image_path)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = get_detailed_ocr_results(image_path)
            print(f"Best text extracted: {result['text']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Method: {result['method']}")
            print(f"Preprocessing: {result['preprocessing']}")
            print(f"Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"Error: File {image_path} not found")
    else:
        print("Usage: python enhanced_ocr.py <image_path>")
