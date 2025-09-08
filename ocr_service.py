#!/usr/bin/env python3
"""
OFFLINE DOCUMENT IDENTIFICATION SERVICE - NO EXTERNAL APIS
Enterprise-grade document processing for EKYC verification
Supports Botswana Omang and International Passports
Compliant with Botswana DPA requirements and offline operation constraints
Target: Maintain 95% accuracy with <60ms response time
"""

import re
import json
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple
import logging
import time
import os
from fast_ocr import get_fast_ocr_results
# Keep enhanced OCR as fallback
from enhanced_ocr import get_detailed_ocr_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== DOCUMENT QUALITY ASSESSMENT =====
def assess_document_quality(image_path: str) -> Dict:
    """
    Assess document image quality for processing confidence
    Returns quality metrics and recommendations
    """
    try:
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"quality_score": 0, "issues": ["Cannot load image"], "processing_time": 0}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        quality_metrics = {}
        issues = []
        
        # 1. Brightness assessment
        mean_brightness = np.mean(gray)
        quality_metrics['brightness'] = mean_brightness
        if mean_brightness < 50:
            issues.append("Image too dark")
        elif mean_brightness > 220:
            issues.append("Image overexposed")
        
        # 2. Contrast assessment
        contrast = gray.std()
        quality_metrics['contrast'] = contrast
        if contrast < 30:
            issues.append("Low contrast")
        
        # 3. Sharpness assessment (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        quality_metrics['sharpness'] = sharpness
        if sharpness < 100:
            issues.append("Blurry image")
        
        # 4. Resolution check
        height, width = gray.shape
        resolution_score = min(width, height)
        quality_metrics['resolution'] = resolution_score
        if resolution_score < 500:
            issues.append("Low resolution")
        
        # 5. Noise assessment
        noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        quality_metrics['noise_level'] = noise_level
        if noise_level > 20:
            issues.append("High noise level")
        
        # Calculate overall quality score (0-100)
        brightness_score = max(0, 100 - abs(mean_brightness - 128) / 128 * 100)
        contrast_score = min(100, contrast / 50 * 100)
        sharpness_score = min(100, sharpness / 500 * 100)
        resolution_score = min(100, resolution_score / 1000 * 100)
        noise_score = max(0, 100 - noise_level / 30 * 100)
        
        quality_score = (brightness_score + contrast_score + sharpness_score + 
                        resolution_score + noise_score) / 5
        
        processing_time = time.time() - start_time
        
        return {
            "quality_score": round(quality_score, 2),
            "metrics": quality_metrics,
            "issues": issues,
            "processing_time": round(processing_time * 1000, 2)  # ms
        }
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return {"quality_score": 0, "issues": [f"Assessment error: {str(e)}"], "processing_time": 0}

# ===== TEXT PROCESSING =====
def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    text = text.upper().strip()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters except basic punctuation
    text = re.sub(r'[^\w\s\-\.\/\(\)]', '', text)
    
    return text

def extract_dates(text: str) -> List[str]:
    """Extract all possible dates from text using multiple patterns"""
    dates = []
    
    # Date patterns for Botswana documents
    patterns = [
        r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b',  # DD/MM/YY or DD/MM/YYYY
        r'\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b',    # YYYY/MM/DD
        r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{2,4})\b',  # DD MON YY/YYYY
        r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2}),?\s+(\d{2,4})\b',  # MON DD, YY/YYYY
        r'\b(\d{2})/(\d{2})/(\d{2})\b',  # DD/MM/YY format common in Botswana
        r'\b(\d{6,8})\b',  # DDMMYYYY or DDMMYY format
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            date_str = match.group().strip()
            # Additional validation for date-like numbers
            if pattern == r'\b(\d{6,8})\b':
                # Only include if it looks like a date (not random 6-8 digit numbers)
                if len(date_str) in [6, 8]:
                    # Basic validation: first two digits should be valid day (01-31)
                    day = date_str[:2]
                    if day.isdigit() and 1 <= int(day) <= 31:
                        dates.append(date_str)
            else:
                dates.append(date_str)
    
    return list(set(dates))  # Remove duplicates

def extract_omang_numbers(text: str) -> List[str]:
    """Extract Botswana Omang numbers (format: 9+ digits)"""
    # Look for various digit sequences that could be Omang numbers
    patterns = [
        r'\b(\d{9})\b',  # Standard 9 digits
        r'\b(\d{17})\b',  # Long format like 08261659949226911
        r'\b(\d{11,17})\b',  # Variable length 11-17 digits
        r'(\d{2}\s*\d{2}\s*\d{2}\s*\d{3,11})',  # With possible spaces
        r'ACBWAD(\d{11,17})',  # After ACBWAD prefix
        r'(\d{8,17})',  # General 8-17 digit sequences
    ]
    
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text)
        for match in found:
            # Clean up any spaces
            clean_number = re.sub(r'\s+', '', match)
            if len(clean_number) >= 8 and clean_number.isdigit():
                # Additional validation: not all same digits
                if len(set(clean_number)) > 1:
                    matches.append(clean_number)
    
    return list(set(matches))  # Remove duplicates

def extract_passport_numbers(text: str) -> List[str]:
    """Extract passport numbers using international patterns"""
    passport_numbers = []
    
    # Various passport number patterns
    patterns = [
        r'\b([A-Z]{1,2}\d{6,9})\b',          # 1-2 letters + 6-9 digits
        r'\b([A-Z]\d{7,8})\b',               # 1 letter + 7-8 digits  
        r'\b(\d{8,10})\b',                   # 8-10 digits only
        r'\b([A-Z]{2}\d{7,8})\b',            # 2 letters + 7-8 digits
        r'PASSPORT\s*NO?\s*[:\-]?\s*([A-Z0-9]{6,12})', # After "PASSPORT NO"
        r'P\s*[:\-]?\s*([A-Z0-9]{6,12})',    # After "P:"
        r'\b([0-9]{7,10})\b',                # 7-10 digit sequences (common in many countries)
        r'([A-Z]{3}\d{6,8})',               # Country code + digits
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        passport_numbers.extend(matches)
    
    # Filter out invalid matches
    valid_numbers = []
    for num in passport_numbers:
        num = num.upper().strip()
        if 6 <= len(num) <= 12:
            # Additional validation: not all same digit
            if not (num.isdigit() and len(set(num)) == 1):
                valid_numbers.append(num)
    
    return list(set(valid_numbers))  # Remove duplicates

def extract_names(text: str) -> List[str]:
    """Extract potential names from text with enhanced patterns"""
    names = []
    
    # Look for name patterns after common keywords
    name_keywords = [
        r'NAME[S]?\s*[:\-]?\s*([A-Z][A-Z\s]{2,50})',
        r'SURNAME[S]?\s*[:\-]?\s*([A-Z][A-Z\s]{2,30})',
        r'FIRST\s*NAME[S]?\s*[:\-]?\s*([A-Z][A-Z\s]{2,30})',
        r'GIVEN\s*NAME[S]?\s*[:\-]?\s*([A-Z][A-Z\s]{2,30})',
        r'FAMILY\s*NAME[S]?\s*[:\-]?\s*([A-Z][A-Z\s]{2,30})',
        r'FORENAMES\s*[:\-]?\s*([A-Z][A-Z\s]{2,40})',  # For Omang cards
    ]
    
    for pattern in name_keywords:
        matches = re.findall(pattern, text)
        for match in matches:
            clean_name = re.sub(r'\s+', ' ', match.strip())
            if len(clean_name) > 2:
                names.append(clean_name)
    
    # Enhanced pattern for Omang cards: Look for names after "IDENTITY CARD"
    omang_name_pattern = r'IDENTITY\s+CARD\s+([A-Z]+)\s+([A-Z\s]+?)(?=\s+\d{2}/\d{2}/\d{2}|\s+\d{6,8}|\s*$)'
    omang_match = re.search(omang_name_pattern, text)
    if omang_match:
        surname = omang_match.group(1).strip()
        forenames = omang_match.group(2).strip()
        full_name = f"{surname} {forenames}"
        names.extend([full_name, surname, forenames])
    
    # Look for capitalized word sequences that could be names
    name_pattern = r'\b([A-Z][A-Z]{2,15}\s+[A-Z][A-Z]{2,15}(?:\s+[A-Z][A-Z]{2,15})?)\b'
    potential_names = re.findall(name_pattern, text)
    
    for name in potential_names:
        # Filter out obvious non-names
        exclude_keywords = ['BOTSWANA', 'REPUBLIC', 'NATIONAL', 'IDENTITY', 'CARD', 'PASSPORT', 'DATE', 'BIRTH', 'PLACE', 'GABORONE']
        if not any(keyword in name for keyword in exclude_keywords):
            if len(name.split()) >= 2:  # At least first and last name
                names.append(name.strip())
    
    return list(set(names))  # Remove duplicates

# ===== DOCUMENT TYPE DETECTION =====
def detect_document_type(text: str, image_path: str = None) -> str:
    """
    Detect document type based on text content and image characteristics
    Returns: 'omang', 'passport', or 'unknown'
    """
    text_upper = text.upper()
    
    # Omang indicators (Botswana specific)
    omang_keywords = [
        'OMANG', 'BOTSWANA', 'REPUBLIC OF BOTSWANA', 'IDENTITY', 'CARD',
        'CITIZENSHIP', 'GABORONE', 'TRIBAL', 'OFFICE OF THE PRESIDENT',
        'NATIONAL IDENTITY', 'MOTSWANA'
    ]
    
    # Passport indicators (international)
    passport_keywords = [
        'PASSPORT', 'REPUBLIC', 'TRAVEL', 'DOCUMENT', 'DIPLOMATIC',
        'OFFICIAL', 'ORDINARY', 'MACHINE READABLE', 'TYPE P',
        'CESTOVNY', 'PAS', 'PASSEPORT', 'PASSAPORTO', 'REISEPASS',
        'SLOVAK', 'SLOVENSKA', 'REPUBLIKA', 'SVK', 'USA', 'UNITED STATES',
        'NATIONALITY', 'DATE OF BIRTH', 'PLACE OF BIRTH', 'SURNAME',
        'GIVEN NAME', 'SEX', 'ISSUED', 'EXPIRES', 'AUTHORITY'
    ]
    
    omang_score = sum(1 for keyword in omang_keywords if keyword in text_upper)
    passport_score = sum(1 for keyword in passport_keywords if keyword in text_upper)
    
    # Check for Omang number pattern (9 digits)
    if extract_omang_numbers(text):
        omang_score += 3
    
    # Check for passport number patterns
    if extract_passport_numbers(text):
        passport_score += 2
    
    # Additional pattern checks for passports
    # Look for country codes (3 letters)
    country_codes = re.findall(r'\b[A-Z]{3}\b', text_upper)
    if any(code in ['USA', 'SVK', 'GBR', 'DEU', 'FRA', 'CAN', 'AUS'] for code in country_codes):
        passport_score += 2
    
    # Look for MRZ-like patterns (P< followed by letters)
    if re.search(r'P<[A-Z]{2,3}', text_upper.replace(' ', '')):
        passport_score += 3
    
    # Look for date patterns common in passports
    if re.search(r'\b\d{2}[A-Z]{3}\d{2,4}\b', text_upper):  # 01JAN2025 format
        passport_score += 1
    
    # Additional image-based detection
    if image_path:
        try:
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                aspect_ratio = width / height
                
                # Omang cards typically have a different aspect ratio than passports
                if 1.4 < aspect_ratio < 1.8:  # Credit card-like ratio
                    omang_score += 1
                elif 0.6 < aspect_ratio < 1.4:  # Passport page ratio (more square-ish)
                    passport_score += 1
        except:
            pass
    
    # Enhanced decision logic
    logger.info(f"📊 Document type scoring - Omang: {omang_score}, Passport: {passport_score}")
    
    if omang_score > passport_score and omang_score >= 2:
        return 'omang'
    elif passport_score > omang_score and passport_score >= 1:  # Lower threshold for passports
        return 'passport'
    elif passport_score >= 1:  # Even if equal, prefer passport if we have any passport indicators
        return 'passport'
    else:
        return 'unknown'

# ===== DOCUMENT PROCESSING =====
def process_omang(text: str) -> Dict:
    """Process Botswana Omang document"""
    logger.info("🇧🇼 Processing Omang document")
    
    result = {
        "document_type": "omang",
        "country": "Botswana",
        "omang_number": None,
        "names": [],
        "dates": [],
        "place_of_birth": None,
        "tribal_territory": None,
        "validity": {},
    }
    
    # Extract Omang number
    omang_numbers = extract_omang_numbers(text)
    if omang_numbers:
        result["omang_number"] = omang_numbers[0]
        result["validity"]["omang_number"] = "valid" if len(omang_numbers[0]) == 9 else "invalid"
    
    # Extract names with enhanced patterns
    result["names"] = extract_names(text)
    
    # Try to extract specific name components from Omang structure
    # Pattern: IDENTITY CARD SURNAME FORENAMES DATE
    omang_structure = re.search(r'IDENTITY\s+CARD\s+([A-Z]+)\s+([A-Z\s]+?)(?=\s+\d{2}/\d{2}/\d{2})', text)
    if omang_structure:
        surname = omang_structure.group(1).strip()
        forenames = omang_structure.group(2).strip()
        result["surname"] = surname
        result["forenames"] = forenames
        result["full_name"] = f"{surname} {forenames}"
        if f"{surname} {forenames}" not in result["names"]:
            result["names"].append(f"{surname} {forenames}")
    
    # Extract dates
    result["dates"] = extract_dates(text)
    
    # Try to extract specific date fields
    dob_pattern = r'(?:DATE\s+OF\s+)?BIRTH[:\-]?\s*(\d{2}/\d{2}/\d{2,4})'
    dob_match = re.search(dob_pattern, text, re.IGNORECASE)
    if dob_match:
        result["date_of_birth"] = dob_match.group(1)
    elif result["dates"]:
        # If no specific DOB found, assume first date is DOB
        result["date_of_birth"] = result["dates"][0]
    
    # Extract specific Omang fields
    # Place of birth - look for common pattern
    birth_patterns = [
        r'PLACE\s*OF\s*BIRTH[:\-]?\s*([A-Z\s]{2,30})',
        r'BIRTH[:\-]?\s*([A-Z]{3,20})',  # Just the place name
    ]
    
    for pattern in birth_patterns:
        birth_match = re.search(pattern, text, re.IGNORECASE)
        if birth_match:
            place = birth_match.group(1).strip()
            # Filter out non-place words
            if place not in ['BIRTH', 'DATE', 'OF'] and len(place) > 2:
                result["place_of_birth"] = place
                break
    
    # For Omang, GABORONE is a common place
    if not result["place_of_birth"] and 'GABORONE' in text:
        result["place_of_birth"] = "GABORONE"
    
    # Tribal territory
    tribal_match = re.search(r'TRIBAL\s*TERRITORY[:\-]?\s*([A-Z\s]{2,30})', text, re.IGNORECASE)
    if tribal_match:
        result["tribal_territory"] = tribal_match.group(1).strip()
    
    # Validate completeness
    required_fields = ["omang_number", "names"]
    missing_fields = [field for field in required_fields if not result[field]]
    
    result["validity"]["completeness"] = "complete" if not missing_fields else "incomplete"
    result["validity"]["missing_fields"] = missing_fields
    
    return result

def process_passport(text: str) -> Dict:
    """Process international passport document"""
    logger.info("🌍 Processing Passport document")
    
    result = {
        "document_type": "passport",
        "passport_number": None,
        "country_code": None,
        "nationality": None,
        "names": [],
        "dates": [],
        "place_of_birth": None,
        "mrz_data": None,
        "document_info": {},
        "validity": {},
    }
    
    # Extract passport number
    passport_numbers = extract_passport_numbers(text)
    if passport_numbers:
        result["passport_number"] = passport_numbers[0]
        result["validity"]["passport_number"] = "valid"
    
    # Extract names with enhanced patterns
    result["names"] = extract_names(text)
    
    # Look for additional name patterns specific to passports
    additional_name_patterns = [
        r'SURNAME[S]?\s*[:\-]?\s*([A-Z\s]{2,40})',
        r'GIVEN\s*NAME[S]?\s*[:\-]?\s*([A-Z\s]{2,40})',
        r'FAMILY\s*NAME[S]?\s*[:\-]?\s*([A-Z\s]{2,40})',
        r'PRENOM[S]?\s*[:\-]?\s*([A-Z\s]{2,40})',  # French
        r'NOM[S]?\s*[:\-]?\s*([A-Z\s]{2,40})',     # French
    ]
    
    for pattern in additional_name_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_name = re.sub(r'\s+', ' ', match.strip())
            if len(clean_name) > 2 and clean_name not in result["names"]:
                result["names"].append(clean_name)
    
    # Extract dates
    result["dates"] = extract_dates(text)
    
    # Extract country information with enhanced patterns
    country_patterns = [
        r'REPUBLIC\s*OF\s*([A-Z\s]{2,30})',
        r'REPUBLIQUE\s*([A-Z\s]{2,30})',
        r'REPUBLIKA\s*([A-Z\s]{2,30})',
        r'SLOVENSKA\s*REPUBLIKA',
        r'SLOVAK\s*REPUBLIC',
        r'UNITED\s*STATES',
        r'\b(USA|SVK|GBR|DEU|FRA|CAN|AUS|ITA|ESP)\b',
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if pattern.endswith(r'\b'):  # Country code pattern
                result["country_code"] = match.group(1)
            else:
                country_name = match.group(1) if match.groups() else match.group()
                result["country_code"] = country_name.strip()
            break
    
    # Extract nationality with enhanced patterns
    nationality_patterns = [
        r'NATIONALITY[:\-]?\s*([A-Z\s]{2,30})',
        r'NATIONALITE[:\-]?\s*([A-Z\s]{2,30})',
        r'NARODNOST[:\-]?\s*([A-Z\s]{2,30})',
    ]
    
    for pattern in nationality_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["nationality"] = match.group(1).strip()
            break
    
    # Extract place of birth
    birth_place_patterns = [
        r'PLACE\s*OF\s*BIRTH[:\-]?\s*([A-Z\s,]{2,40})',
        r'LIEU\s*DE\s*NAISSANCE[:\-]?\s*([A-Z\s,]{2,40})',
        r'MIESTO\s*NARODENIA[:\-]?\s*([A-Z\s,]{2,40})',
    ]
    
    for pattern in birth_place_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["place_of_birth"] = match.group(1).strip()
            break
    
    # Look for MRZ (Machine Readable Zone) - typically at bottom
    # MRZ lines start with P< and contain <<<
    mrz_patterns = [
        r'P<[A-Z<]{1,3}[A-Z<\d]{39,44}',
        r'[A-Z0-9<]{28,44}',  # Second line of MRZ
    ]
    
    mrz_found = False
    for pattern in mrz_patterns:
        match = re.search(pattern, text.replace(' ', ''))
        if match:
            result["mrz_data"] = match.group()
            result["validity"]["mrz_present"] = True
            mrz_found = True
            break
    
    if not mrz_found:
        result["validity"]["mrz_present"] = False
    
    # Extract document information (issue date, expiry, etc.)
    doc_info_patterns = [
        (r'DATE\s*OF\s*ISSUE[:\-]?\s*([0-9\/\-\.]{6,12})', 'issue_date'),
        (r'ISSUED[:\-]?\s*([0-9\/\-\.]{6,12})', 'issue_date'),
        (r'EXPIR[YE][S]?[:\-]?\s*([0-9\/\-\.]{6,12})', 'expiry_date'),
        (r'VALID\s*UNTIL[:\-]?\s*([0-9\/\-\.]{6,12})', 'expiry_date'),
        (r'SEX[:\-]?\s*([MF])', 'sex'),
        (r'AUTORITE[:\-]?\s*([A-Z\s]{2,30})', 'issuing_authority'),
        (r'AUTHORITY[:\-]?\s*([A-Z\s]{2,30})', 'issuing_authority'),
    ]
    
    for pattern, field_name in doc_info_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["document_info"][field_name] = match.group(1).strip()
    
    # Validate completeness
    required_fields = ["names"]  # Relaxed requirements for passports
    optional_fields = ["passport_number", "country_code", "dates"]
    
    missing_fields = [field for field in required_fields if not result[field]]
    present_optional = [field for field in optional_fields if result[field]]
    
    # Consider complete if we have names and at least one optional field
    completeness = "complete" if not missing_fields and len(present_optional) > 0 else "incomplete"
    
    result["validity"]["completeness"] = completeness
    result["validity"]["missing_fields"] = missing_fields
    result["validity"]["fields_found"] = len([f for f in required_fields + optional_fields if result[f]])
    
    return result

# ===== MAIN IDENTIFICATION FUNCTION =====
def identify_document(image_path: str, enable_quality_check: bool = True) -> Dict:
    """
    Main document identification function - OFFLINE ONLY
    
    Args:
        image_path: Path to the document image
        enable_quality_check: Whether to perform quality assessment
    
    Returns:
        Comprehensive document analysis results
    """
    start_time = time.time()
    logger.info(f"🔍 Starting offline document identification: {os.path.basename(image_path)}")
    
    try:
        # Step 1: Quality assessment
        quality_result = {}
        if enable_quality_check:
            quality_result = assess_document_quality(image_path)
            logger.info(f"📊 Quality Score: {quality_result['quality_score']:.1f}/100")
            
            if quality_result['quality_score'] < 30:
                logger.warning("⚠️ Low quality image detected")
        
        # Step 2: Extract text using fast OCR (with enhanced fallback)
        logger.info("🔤 Extracting text with fast OCR...")
        
        # Try fast OCR first
        ocr_results = get_fast_ocr_results(image_path)
        extracted_text = ocr_results['text']
        ocr_confidence = ocr_results['confidence']
        ocr_method = ocr_results['method']
        
        # If fast OCR fails or gives poor results, fallback to enhanced OCR
        if not extracted_text or len(extracted_text.strip()) < 10 or ocr_confidence < 0.3:
            logger.info("🔄 Fast OCR insufficient, using enhanced OCR...")
            ocr_results = get_detailed_ocr_results(image_path)
            extracted_text = ocr_results['text']
            ocr_confidence = ocr_results['confidence']
            ocr_method = f"enhanced_{ocr_results['method']}"
        
        # If still no text extracted, create minimal result
        if not extracted_text or len(extracted_text.strip()) == 0:
            logger.warning("⚠️ No text extracted - creating minimal result")
            return {
                "success": True,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "image_file": os.path.basename(image_path),
                "file_size_kb": round(os.path.getsize(image_path) / 1024, 2),
                "ocr_method": "fast_offline",
                "ocr_details": {
                    "confidence": ocr_confidence,
                    "method_used": ocr_method,
                    "preprocessing": ocr_results.get('preprocessing', 'unknown'),
                    "total_methods_tried": ocr_results.get('total_methods_tried', 0)
                },
                "document_data": {
                    "document_type": "unknown",
                    "extracted_text": "",
                    "validity": {"status": "no_text_extracted"}
                },
                "quality_assessment": quality_result,
                "extracted_text_length": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_info": {
                    "offline_mode": True,
                    "apis_used": "none",
                    "compliance": "Botswana DPA compliant",
                    "fast_ocr": True
                }
            }
        
        logger.info(f"✅ Extracted {len(extracted_text)} characters with {ocr_confidence:.2f} confidence using {ocr_method}")
        
        # Step 3: Detect document type
        doc_type = detect_document_type(extracted_text, image_path)
        logger.info(f"📋 Detected document type: {doc_type}")
        
        # Step 4: Process based on document type
        if doc_type == 'omang':
            document_data = process_omang(extracted_text)
        elif doc_type == 'passport':
            document_data = process_passport(extracted_text)
        else:
            # Generic processing for unknown documents
            document_data = {
                "document_type": "unknown",
                "extracted_text": extracted_text[:500],  # First 500 chars
                "numbers_found": extract_omang_numbers(extracted_text) + extract_passport_numbers(extracted_text),
                "names_found": extract_names(extracted_text),
                "dates_found": extract_dates(extracted_text),
                "validity": {"status": "unrecognized_document_type"}
            }
        
        # Step 5: Compile final result
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "processing_time_ms": round(processing_time * 1000, 2),
            "image_file": os.path.basename(image_path),
            "file_size_kb": round(os.path.getsize(image_path) / 1024, 2),
            "ocr_method": "fast_offline",
            "ocr_details": {
                "confidence": ocr_confidence,
                "method_used": ocr_method,
                "preprocessing": ocr_results.get('preprocessing', 'unknown'),
                "total_methods_tried": ocr_results.get('total_methods_tried', 0)
            },
            "document_data": document_data,
            "quality_assessment": quality_result,
            "extracted_text_length": len(extracted_text),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "offline_mode": True,
                "apis_used": "none",
                "compliance": "Botswana DPA compliant",
                "fast_ocr": True
            }
        }
        
        # Performance check
        if processing_time * 1000 > 60:  # Target: <60ms
            logger.warning(f"⚠️ Processing took {processing_time*1000:.1f}ms (target: <60ms)")
        else:
            logger.info(f"🚀 Processing completed in {processing_time*1000:.1f}ms")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_result = {
            "success": False,
            "error": str(e),
            "processing_time_ms": round(processing_time * 1000, 2),
            "image_file": os.path.basename(image_path) if os.path.exists(image_path) else "file_not_found",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "offline_mode": True,
                "apis_used": "none",
                "compliance": "Botswana DPA compliant",
                "fast_ocr": True
            }
        }
        
        logger.error(f"❌ Document identification failed: {e}")
        return error_result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = identify_document(image_path)
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: File {image_path} not found")
    else:
        print("Usage: python ocr_service.py <image_path>")
        print("Example: python ocr_service.py omang1-front.png")
