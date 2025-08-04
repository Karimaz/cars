import streamlit as st
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image
import os, io
import requests

subprocess.run([sys.executable, "-m", "pip", "show", "opencv-python"])

BASE = os.path.dirname(__file__)
CAR_IMG_DIR = os.path.join(BASE, "cars_imgs")
SHOWROOM_BG = os.path.join(BASE, "backgrounds", "showroom.png")


st.title("🚗 KSA Cars with Background") 
st.write(f"OpenCV version: {cv2.__version__}")


# Replace with your actual URL
CAR_IMG_URL = "https://doktorly.de/ksa_cars"

def list_remote_folders(url):
    try:
        resp = requests.get(url + "/folders.json")
        resp.raise_for_status()
        return sorted(resp.json())
    except Exception as e:
        st.error(f"Failed to fetch car folders: {e}")
        st.stop()

car_folders = list_remote_folders(CAR_IMG_URL)
selected_folder = st.selectbox("Select Car Folder", car_folders)

# Get list of images in the selected folder
try:
    car_files_resp = requests.get(f"{CAR_IMG_URL}/{selected_folder}/index.json")
    car_files_resp.raise_for_status()
    car_files = [f for f in car_files_resp.json()
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
except Exception as e:
    st.error(f"Failed to fetch car images: {e}")
    st.stop()

# Car image selection
car_names = [os.path.splitext(fn)[0] for fn in car_files]
selected_car_name = st.selectbox("Select Car Image", car_names)
selected_car = car_files[car_names.index(selected_car_name)]

# ✅ Correct way to load and convert remote image with OpenCV
image_url = f"{CAR_IMG_URL}/{selected_folder}/{selected_car}"
resp = requests.get(image_url)
img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
car = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

# Background selection
bg_files = sorted(
    fn for fn in os.listdir(os.path.join(BASE, "backgrounds"))
    if fn.lower().endswith((".png", ".jpg", ".jpeg")) and fn != "showroom.png"
)
bg_options = ["showroom.png"] + bg_files + ["[Upload Custom]"]
selected_bg = st.selectbox("Select Background", bg_options, index=0)

if selected_bg == "[Upload Custom]":
    uploaded_bg = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])
    if uploaded_bg is not None:
        bg_img = Image.open(uploaded_bg).convert("RGB")
        bg = np.array(bg_img)
    else:
        st.warning("Please upload a background image.")
        st.stop()
else:
    bg_path = os.path.join(BASE, "backgrounds", selected_bg)
    if not os.path.exists(bg_path):
        st.error(f"Missing background: {bg_path}")
        st.stop()
    bg = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
h, w = car.shape[:2]

#--- Buttom Text aka Watermark ---
do_brand = st.checkbox("Add Branding", True)
brand_text = st.text_input("Brand Text", "SmartiveMedia 2025")


canvas_w, canvas_h = w * 2, h * 2
canvas = cv2.resize(bg, (canvas_w, canvas_h))

#Positioning options
x_off = st.slider("Car X Position", 0, canvas_w - w, (canvas_w - w) // 2)
y_off = st.slider("Car Y Position", 0, canvas_h - h, 350)

# Put things together
roi = canvas[y_off:y_off+h, x_off:x_off+w]
gray = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)
_, mask_bg = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
mask_fg = cv2.bitwise_not(mask_bg)

if mask_bg.shape[:2] != roi.shape[:2]:
    mask_bg = cv2.resize(mask_bg, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_fg = cv2.resize(mask_fg, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

bg_part = cv2.bitwise_and(roi, roi, mask=mask_bg)
fg_part = cv2.bitwise_and(car, car, mask=mask_fg)
canvas[y_off:y_off+h, x_off:x_off+w] = cv2.add(bg_part, fg_part)

# Watermark add 
if do_brand:
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = cv2.getTextSize(brand_text, font, 1.2, 2)[0]
    bx = (canvas_w - ts[0]) // 2
    by = canvas_h - 30
    cv2.putText(canvas, brand_text, (bx+2, by+2), font, 1.2, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(canvas, brand_text, (bx, by), font, 1.2, (255,255,255), 2, cv2.LINE_AA)

#  Display and Download Options
st.image(canvas, caption="🚗 Positioned Car", use_container_width=True)
buf = io.BytesIO()
Image.fromarray(canvas).save(buf, format="PNG")
st.download_button("📥 Download Image", buf.getvalue(), "car_positioned.png", "image/png")
