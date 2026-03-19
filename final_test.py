import requests
import json
import os

url = "http://localhost:8000/api/predict"
# Pick a random test image from the local dataset to verify
test_img_dir = r"c:\Users\Om Sai\Downloads\CNN-Based-Brain-MRI-Tumor-Classification-main\CNN-Based-Brain-MRI-Tumor-Classification-main\brain-tumor-mri-dataset\Testing\glioma"
test_img = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])

print(f"Testing with image: {test_img}")

with open(test_img, "rb") as f:
    files = {"image": (os.path.basename(test_img), f, "image/jpeg")}
    try:
        response = requests.post(url, files=files)
        print("Status:", response.status_code)
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Details:", response.text)
    except Exception as e:
        print("Request failed:", e)
