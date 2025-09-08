#!/usr/bin/env python3
"""
OFFLINE OCR CONFIGURATION - NO EXTERNAL APIS
Configuration for on-premise document processing system
Compliant with Botswana DPA requirements
"""

import os
import logging

# ===== OFFLINE OCR SETTINGS =====
OFFLINE_OCR_CONFIG = {
    # EasyOCR Settings (Primary OCR Engine)
    "easyocr": {
        "languages": ["en"],  # English only for Botswana documents
        "gpu": False,         # Use CPU for stability and universal compatibility
        "model_storage_directory": "./models/easyocr",
        "download_enabled": False,  # Disable model downloads for offline operation
        "confidence_threshold": 0.3,
        "width_ths": 0.7,
        "height_ths": 0.7
    },
    
    # Tesseract Settings (Backup OCR Engine)
    "tesseract": {
        "cmd": "tesseract",
        "config": "--oem 3 --psm 6",
        "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ ",
        "timeout": 30,  # seconds
        "min_confidence": 30
    },
    
    # Image Preprocessing Settings
    "preprocessing": {
        "enable_denoising": True,
        "enable_deskewing": True,
        "enable_contrast_enhancement": True,
        "enable_sharpening": True,
        "gaussian_blur_kernel": (5, 5),
        "adaptive_threshold_block_size": 11,
        "adaptive_threshold_c": 2,
        "rotation_angle_threshold": 0.5  # degrees
    },
    
    # Quality Assessment Thresholds
    "quality_thresholds": {
        "minimum_brightness": 50,
        "maximum_brightness": 220,
        "minimum_contrast": 30,
        "minimum_sharpness": 100,
        "minimum_resolution": 500,
        "maximum_noise_level": 20,
        "minimum_quality_score": 30
    },
    
    # Performance Settings
    "performance": {
        "target_processing_time_ms": 60,
        "enable_parallel_processing": False,  # Single-threaded for stability
        "max_image_size_mb": 10,
        "enable_caching": False  # No caching for security
    }
}

# ===== DOCUMENT TYPE PATTERNS =====
DOCUMENT_PATTERNS = {
    "omang": {
        "keywords": [
            "OMANG", "BOTSWANA", "REPUBLIC OF BOTSWANA", "IDENTITY", "CARD",
            "CITIZENSHIP", "GABORONE", "TRIBAL", "OFFICE OF THE PRESIDENT"
        ],
        "number_pattern": r"\b(\d{9})\b",
        "aspect_ratio_range": (1.4, 1.8),
        "required_fields": ["omang_number", "names"]
    },
    
    "passport": {
        "keywords": [
            "PASSPORT", "REPUBLIC", "TRAVEL", "DOCUMENT", "DIPLOMATIC",
            "OFFICIAL", "ORDINARY", "MACHINE READABLE", "TYPE P"
        ],
        "number_patterns": [
            r"\b([A-Z]{1,2}\d{6,9})\b",
            r"\b([A-Z]\d{7})\b",
            r"\b(\d{8,9})\b",
            r"\b([A-Z]{2}\d{7})\b"
        ],
        "aspect_ratio_range": (0.6, 0.8),
        "required_fields": ["passport_number", "names"]
    }
}

# ===== DATE PATTERNS =====
DATE_PATTERNS = [
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b",  # DD/MM/YYYY
    r"\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b",  # YYYY/MM/DD
    r"\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b",  # DD MON YYYY
    r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2}),?\s+(\d{4})\b"  # MON DD, YYYY
]

# ===== NAME EXTRACTION PATTERNS =====
NAME_PATTERNS = [
    r"NAME[S]?\s*[:\-]?\s*([A-Z\s]{2,50})",
    r"SURNAME[S]?\s*[:\-]?\s*([A-Z\s]{2,30})",
    r"FIRST\s*NAME[S]?\s*[:\-]?\s*([A-Z\s]{2,30})",
    r"GIVEN\s*NAME[S]?\s*[:\-]?\s*([A-Z\s]{2,30})",
    r"FAMILY\s*NAME[S]?\s*[:\-]?\s*([A-Z\s]{2,30})"
]

# ===== SYSTEM CONFIGURATION =====
SYSTEM_CONFIG = {
    "mode": "offline",
    "external_apis_allowed": False,
    "internet_required": False,
    "compliance": "Botswana DPA",
    "data_retention": "session_only",
    "logging_level": "INFO",
    "security_mode": "high"
}

# ===== SUPPORTED FILE FORMATS =====
SUPPORTED_FORMATS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".pdf"  # PDF support through pdf2image
]

# ===== LOGGING CONFIGURATION =====
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console"],  # No file logging for security
    "sensitive_data_filtering": True
}

# ===== ERROR MESSAGES =====
ERROR_MESSAGES = {
    "file_not_found": "Document image file not found",
    "invalid_format": "Unsupported file format",
    "processing_failed": "Document processing failed",
    "low_quality": "Document image quality too low for processing",
    "no_text_extracted": "No text could be extracted from the document",
    "ocr_engine_error": "OCR engine encountered an error",
    "timeout": "Processing timeout exceeded",
    "unknown_document": "Document type could not be determined"
}

# ===== VALIDATION RULES =====
VALIDATION_RULES = {
    "omang_number": {
        "length": 9,
        "type": "numeric",
        "required": True
    },
    "passport_number": {
        "min_length": 6,
        "max_length": 12,
        "type": "alphanumeric",
        "required": True
    },
    "names": {
        "min_length": 2,
        "max_length": 50,
        "type": "alphabetic",
        "required": True
    },
    "dates": {
        "format": "various",
        "validation": "date_patterns",
        "required": False
    }
}

# ===== HELPER FUNCTIONS =====
def get_offline_config():
    """Get the complete offline configuration"""
    return {
        "ocr": OFFLINE_OCR_CONFIG,
        "documents": DOCUMENT_PATTERNS,
        "dates": DATE_PATTERNS,
        "names": NAME_PATTERNS,
        "system": SYSTEM_CONFIG,
        "formats": SUPPORTED_FORMATS,
        "logging": LOGGING_CONFIG,
        "errors": ERROR_MESSAGES,
        "validation": VALIDATION_RULES
    }

def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in SUPPORTED_FORMATS

def get_tesseract_config() -> str:
    """Get Tesseract configuration string"""
    return OFFLINE_OCR_CONFIG["tesseract"]["config"]

def get_quality_threshold(metric: str) -> float:
    """Get quality threshold for specific metric"""
    return OFFLINE_OCR_CONFIG["quality_thresholds"].get(metric, 0)

# ===== ENVIRONMENT VALIDATION =====
def validate_offline_environment():
    """Validate that the environment is properly configured for offline operation"""
    validation_results = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    # Check for required packages
    try:
        import easyocr
        validation_results["warnings"].append("EasyOCR available")
    except ImportError:
        validation_results["valid"] = False
        validation_results["issues"].append("EasyOCR not installed")
    
    try:
        import pytesseract
        validation_results["warnings"].append("Tesseract available")
    except ImportError:
        validation_results["issues"].append("Tesseract not available (backup OCR)")
    
    try:
        import cv2
        validation_results["warnings"].append("OpenCV available")
    except ImportError:
        validation_results["valid"] = False
        validation_results["issues"].append("OpenCV not installed")
    
    # Check internet connectivity (should be disabled)
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        validation_results["warnings"].append("Internet connection detected - ensure offline mode")
    except (socket.error, OSError):
        validation_results["warnings"].append("No internet connection - good for offline mode")
    
    return validation_results

if __name__ == "__main__":
    # Validate environment when run directly
    print("🔧 Validating offline OCR environment...")
    results = validate_offline_environment()
    
    if results["valid"]:
        print("✅ Environment validation passed")
    else:
        print("❌ Environment validation failed")
        for issue in results["issues"]:
            print(f"  - {issue}")
    
    for warning in results["warnings"]:
        print(f"ℹ️  {warning}")
    
    print("\n📊 Configuration Summary:")
    config = get_offline_config()
    print(f"  - Mode: {config['system']['mode']}")
    print(f"  - External APIs: {config['system']['external_apis_allowed']}")
    print(f"  - Compliance: {config['system']['compliance']}")
    print(f"  - Supported formats: {len(config['formats'])}")
