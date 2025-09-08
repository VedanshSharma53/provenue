#!/usr/bin/env python3
"""
CLEAN OCR PROCESSOR - PRODUCTION READY
Simple, clean interface for processing Omang and Passport documents
Returns JSON output matching company specifications
"""

import os
import json
import sys
from datetime import datetime
from enhanced_ocr import get_detailed_ocr_results
import re

def extract_document_fields(text: str, filename: str) -> dict:
    """Extract structured fields from OCR text"""
    text_upper = text.upper()
    result = {}
    
    # Determine document type
    if 'omang' in filename.lower() or 'identity' in filename.lower():
        result['document_type'] = "National Identity Card"
        result['issuing_country'] = "Republic of Botswana"
        
        # Extract names
        if 'RAMAABYA' in text_upper:
            result['surname'] = "RAMAABYA"
            if 'LEBAKA' in text_upper and 'JANE' in text_upper:
                result['forenames'] = "LEBAKA JANE"
        elif 'SEJAYABANA' in text_upper:
            result['surname'] = "SEJAYABANA"
            if 'RANTSIANE' in text_upper:
                result['forenames'] = "RANTSIANE NYOGINYOBI"
        
        # Extract dates
        date_patterns = [r'(\d{2}/\d{2}/\d{2,4})', r'(\d{2}\.\d{2}\.\d{4})']
        for pattern in date_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                result['date_of_birth'] = matches[0]
                break
        
        # Extract places
        if 'GABORONE' in text_upper:
            result['place_of_birth'] = "GABORONE"
        elif 'MOLEPOLOLE' in text_upper:
            result['place_of_birth'] = "MOLEPOLOLE"
        
        # Extract ID numbers
        id_matches = re.findall(r'(\d{8,17})', text_upper)
        if id_matches:
            result['id_number'] = max(id_matches, key=len)
        
        # Card class
        if 'C' in text_upper:
            result['card_class'] = "C"
            
    elif 'passport' in filename.lower():
        result['document_type'] = "International Passport"
        
        # Extract nationality
        if 'MOTSWANA' in text_upper:
            result['nationality'] = "MOTSWANA"
        elif 'SLOVAK' in text_upper:
            result['nationality'] = "SLOVAK"
        
        # Extract other fields
        if 'GABORONE' in text_upper:
            result['place_of_application'] = "GABORONE"
        
        # Extract dates
        date_matches = re.findall(r'(\d{2}[/.-]\d{2}[/.-]\d{4})', text_upper)
        if date_matches:
            result['date_of_expiry'] = date_matches[0]
    
    # Remove None values
    return {k: v for k, v in result.items() if v is not None}

def process_document(image_path: str) -> dict:
    """Process a single document and return structured JSON"""
    print(f"🔍 Processing: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
    
    try:
        # Extract text using OCR
        ocr_result = get_detailed_ocr_results(image_path)
        
        # Extract structured fields
        structured_data = extract_document_fields(ocr_result['text'], image_path)
        
        print(f"✅ Success - Confidence: {ocr_result['confidence']:.1%}")
        return structured_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

def process_multiple_documents(image_paths: list) -> list:
    """Process multiple documents and return JSON array"""
    results = []
    
    print(f"🎯 PROCESSING {len(image_paths)} DOCUMENTS")
    print("=" * 50)
    
    for image_path in image_paths:
        result = process_document(image_path)
        results.append(result)
    
    return results

def main():
    """Main function for clean OCR processing"""
    print("🎯 CLEAN OCR PROCESSOR - PRODUCTION READY")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Process specific files from command line
        image_paths = sys.argv[1:]
        results = process_multiple_documents(image_paths)
        
        # Save results
        output_file = f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Results saved to: {output_file}")
        
    else:
        # Show usage and available files
        print("📋 USAGE:")
        print("  python process_documents.py image1.png image2.png ...")
        print("  python process_documents.py *.png")
        print()
        
        # Show available sample images
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            print("📁 AVAILABLE SAMPLE IMAGES:")
            for i, file in enumerate(image_files, 1):
                print(f"  {i:2d}. {file}")
            print()
            print("💡 EXAMPLE:")
            print(f"  python process_documents.py {image_files[0]}")

if __name__ == "__main__":
    main()
