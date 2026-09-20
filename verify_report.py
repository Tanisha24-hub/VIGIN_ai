import os
import pandas as pd

CSV_PATH = os.path.join("data", "output", "report_test_compilation.csv")
JSON_PATH = os.path.join("data", "output", "report_test_compilation.json")

print("==================================================")
print("             VIGIN_ai Pipeline Verification        ")
print("==================================================")

# 1. Check if report files exist
if not os.path.exists(CSV_PATH):
    print(f"[ERROR] CSV Report file not found at '{CSV_PATH}'!")
    print("Please make sure you run: python main.py test_compilation.mp4")
    exit(1)
else:
    print(f"[OK] CSV Report file found: '{CSV_PATH}'")

if not os.path.exists(JSON_PATH):
    print(f"[ERROR] JSON Report file not found at '{JSON_PATH}'!")
    exit(1)
else:
    print(f"[OK] JSON Report file found: '{JSON_PATH}'")

print("\n--------------------------------------------------")
print("               Generated Report Contents          ")
print("--------------------------------------------------")

# 2. Read and print the report
df = pd.read_csv(CSV_PATH)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(df.to_string(index=False))
print("--------------------------------------------------\n")

# 3. Assertions / Checks
detected_plates = len(df)
unknown_plates = len(df[df["License Plate"].isna() | (df["License Plate"] == "UNKNOWN")])

print(f"• Total detected plates: {detected_plates}")
print(f"• Vehicles with UNKNOWN plates in output: {unknown_plates}")

if unknown_plates > 0:
    print("[FAIL] Verification Failed: The report still contains UNKNOWN plates.")
else:
    print("[SUCCESS] Verification Passed: The report contains zero UNKNOWN plates.")

# Check Vehicle #2 (Main Vehicle)
veh2 = df[df["Vehicle ID"] == 2]
if not veh2.empty:
    plate2 = str(veh2.iloc[0]["License Plate"]).strip()
    conf2 = veh2.iloc[0].get("OCR Confidence", veh2.iloc[0].get("Confidence", 0.0))
    if plate2 == "UNKNOWN":
        print("[FAIL] Verification Failed: Vehicle #2 license plate is still UNKNOWN.")
    elif plate2 == "1":
        print("[FAIL] Verification Failed: Vehicle #2 license plate read is incorrect ('1').")
    else:
        print(f"[SUCCESS] Verification Passed: Vehicle #2 license plate successfully read as '{plate2}' (Conf: {conf2:.2f})")
else:
    print("[WARNING] Vehicle #2 not found in the report.")

# Check other plates
if detected_plates > 1:
    print(f"[SUCCESS] Verification Passed: Successfully extracted multiple license plates across the video compilation!")
else:
    print("[FAIL] Verification Failed: No other license plates detected.")

print("\n==================================================")
