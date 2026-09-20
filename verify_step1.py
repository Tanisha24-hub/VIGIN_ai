from modules.plate_ocr import PlateRecognizer
import numpy as np
import cv2

print("==================================================")
print("       VIGIN_ai: Step 1 Verification Suite        ")
print("==================================================")

ocr = PlateRecognizer()

# Test 1: Regex & Character Disambiguation Engine
test_cases = [
    # (Raw OCR Input, Expected Output / Should Match)
    ("KA 01 AH 7253", "KA01AH7253"),
    ("GJ02CG3862", "GJ02CG3862"),
    ("DL-01-AB-1234", "DL01AB1234"),
    ("22BH1234AA", "22BH1234AA"),
    ("KA0IAH7253", "KA01AH7253"),      # 'I' swapped to '1' in district code
    ("KA01AH72S3", "KA01AH7253"),      # 'S' swapped to '5' in sequence
    ("VENTURES", None),                # YouTube watermark (must reject)
    ("2023112215352433", None),        # Dashboard timestamp (must reject)
    ("PEPSI", None),                   # Random text (must reject)
]

passed_tests = 0
for raw, expected in test_cases:
    # Test normalization and regex matching
    cleaned = ocr.normalize_plate_text(raw) if hasattr(ocr, 'normalize_plate_text') else raw.replace(" ", "").upper()
    is_valid = bool(ocr.is_valid_indian_plate(cleaned)) if hasattr(ocr, 'is_valid_indian_plate') else (len(cleaned) >= 8 and len(cleaned) <= 10)
    
    if expected is not None:
        if is_valid and (expected in cleaned or cleaned == expected):
            print(f"✅ PASS: Raw '{raw}' -> Validated: '{cleaned}'")
            passed_tests += 1
        else:
            print(f"❌ FAIL: Expected '{expected}', got '{cleaned}' (Valid: {is_valid})")
    else:
        if not is_valid:
            print(f"✅ PASS: Invalid string '{raw}' was correctly REJECTED.")
            passed_tests += 1
        else:
            print(f"❌ FAIL: Junk string '{raw}' was incorrectly ACCEPTED as '{cleaned}'")

print(f"\nResult: {passed_tests}/{len(test_cases)} verification tests passed.")
if passed_tests == len(test_cases):
    print("🎯 Step 1 is FULLY VERIFIED and ready for Step 2!")
else:
    print("⚠️ Some regex/disambiguation rules need adjustment in modules/plate_ocr.py")
