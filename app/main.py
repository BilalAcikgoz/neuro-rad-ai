from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import sys
import os
import torch
import torch.serialization
import numpy as np
import nibabel as nib
import pydicom
import cv2
import io
import base64
from PIL import Image
import uuid
from io import BytesIO
import zipfile
import json

# Add the project root directory to the Python module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from scripts.train import LightUNet3D

app = Flask(__name__)

# Model path defining
MODEL_PATH = "models/best_brats_model.pt"

# First create the model using the Model class
model = LightUNet3D(in_channels=4, out_channels=2, base_filters=12)

# After load weights
with torch.serialization.safe_globals([LightUNet3D]):
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)

model.eval()

# In-memory storage for processing results
# Using a dictionary with session_id as keys
session_data = {}

def preprocess_3d_image(file_data, file_extension):
    """
    Preprocess a 3D medical image in memory
    
    Args:
        file_data: BytesIO object containing file data
        file_extension: File extension to determine processing method
    """
    if file_extension == '.nii' or file_extension == '.gz':
        # Load NIfTI file from memory
        fh = nib.FileHolder(fileobj=file_data)
        img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        image_data = img.get_fdata()
        
        # Normalize each slice
        normalized_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
        
    elif file_extension == '.dcm':
        # For a single DICOM file
        file_data.seek(0)
        ds = pydicom.dcmread(file_data)
        image_data = ds.pixel_array
        normalized_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    elif file_extension == '.zip':
        # For a directory of DICOM files in a ZIP
        dicom_files = []
        with zipfile.ZipFile(file_data) as zip_ref:
            # Extract file list
            file_list = [f for f in zip_ref.namelist() if f.endswith('.dcm')]
            file_list.sort()
            
            # Read the first file to get dimensions
            with zip_ref.open(file_list[0]) as first_file:
                first_ds = pydicom.dcmread(first_file)
                rows, cols = first_ds.Rows, first_ds.Columns
            
            # Allocate 3D array
            image_data = np.zeros((len(file_list), rows, cols), dtype=np.float32)
            
            # Read all DICOM files
            for i, dicom_file in enumerate(file_list):
                with zip_ref.open(dicom_file) as f:
                    ds = pydicom.dcmread(f)
                    image_data[i, :, :] = ds.pixel_array
            
            normalized_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Resize to match model input dimensions if needed
    depth, height, width = normalized_data.shape
    
    # Resize to match expected input size (keeping depth, resizing H and W)
    resized_data = np.zeros((depth, 128, 128), dtype=np.float32)
    for i in range(depth):
        resized_data[i] = cv2.resize(normalized_data[i], (128, 128))
    
    # The model expects 4 channels (flair, t1, t1ce, t2)
    # For testing, we'll duplicate the same volume across all 4 channels
    input_tensor = np.zeros((4, depth, 128, 128), dtype=np.float32)
    for c in range(4):
        input_tensor[c] = resized_data
    
    # Convert to PyTorch tensor and add batch dimension
    tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
    
    return tensor, normalized_data, normalized_data.shape

def generate_slice_images(volume_data, segmentation_data):
    """
    Generate original and segmentation overlay images for each slice in memory
    
    Returns:
        Dictionary of images stored in BytesIO objects
    """
    num_slices = volume_data.shape[0]
    images = {}
    
    for i in range(num_slices):
        # Original slice
        orig_slice = volume_data[i]
        orig_slice = (orig_slice * 255).astype(np.uint8)
        orig_slice_rgb = cv2.cvtColor(orig_slice, cv2.COLOR_GRAY2RGB)
        
        # Segmentation overlay
        seg_slice = segmentation_data[i]
        seg_overlay = orig_slice_rgb.copy()
        
        # Create red overlay for segmentation
        seg_overlay[seg_slice == 1, 0] = 255  # Red channel
        seg_overlay[seg_slice == 1, 1] = 0    # Green channel
        seg_overlay[seg_slice == 1, 2] = 0    # Blue channel
        
        # Save images to BytesIO objects
        orig_buffer = BytesIO()
        _, encoded_orig = cv2.imencode('.png', orig_slice)
        orig_buffer.write(encoded_orig)
        orig_buffer.seek(0)
        
        seg_buffer = BytesIO()
        _, encoded_seg = cv2.imencode('.png', seg_overlay)
        seg_buffer.write(encoded_seg)
        seg_buffer.seek(0)
        
        # Store in dictionary
        images[f"original_{i}"] = orig_buffer
        images[f"segmentation_{i}"] = seg_buffer
    
    return images, num_slices

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return render_template("index.html", error="Lütfen bir dosya yükleyin!")
    
    file = request.files["file"]
    
    if file.filename == "":
        return render_template("index.html", error="Dosya seçilmedi!")
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    valid_extensions = ['.nii', '.dcm']
    is_valid = any(file.filename.endswith(ext) for ext in valid_extensions) or file.filename.endswith('.nii.gz')
    
    if not is_valid:
        return render_template("index.html", error="Lütfen geçerli bir DICOM (.dcm), NIfTI (.nii/.nii.gz) veya ZIP dosyası yükleyin")
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    # Read file into memory
    file_data = BytesIO(file.read())
    
    try:
        # Preprocess the 3D image
        input_tensor, original_volume, original_shape = preprocess_3d_image(file_data, file_extension)
        
        # Model prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process segmentation result
        _, preds = torch.max(output, dim=1)
        
        # Get the predicted segmentation volume (remove batch dimension)
        pred_volume = preds[0].cpu().numpy()
        
        # Resize segmentation back to original dimensions if needed
        depth = original_shape[0]
        resized_segmentation = np.zeros(original_shape, dtype=np.uint8)
        
        for i in range(depth):
            if i < pred_volume.shape[0]:  # Make sure we don't exceed the prediction depth
                # Resize the segmentation slice to match the original dimensions
                resized_segmentation[i] = cv2.resize(
                    pred_volume[i].astype(np.uint8),
                    (original_shape[2], original_shape[1]),
                    interpolation=cv2.INTER_NEAREST
                )
        
        # Generate slice images for web viewing
        images, num_slices = generate_slice_images(original_volume, resized_segmentation)
        
        # Calculate tumor statistics
        tumor_percentage = np.mean(resized_segmentation) * 100
        tumor_volume_pixels = np.sum(resized_segmentation)
        
        # Store data in memory
        session_data[session_id] = {
            "images": images,
            "stats": {
                "tumor_percentage": float(tumor_percentage),
                "tumor_volume_pixels": int(tumor_volume_pixels),
                "num_slices": int(num_slices)
            }
        }
        
        # Redirect to results page
        return redirect(url_for('view_results', session_id=session_id))
        
    except Exception as e:
        # Clean up on error
        if session_id in session_data:
            del session_data[session_id]
        
        return render_template("index.html", error=f"İşlem sırasında bir hata oluştu: {str(e)}")
    

@app.route("/results/<session_id>")
def view_results(session_id):
    # Check if session exists
    if session_id not in session_data:
        return "Result not found", 404
    
    # Get session data
    images = session_data[session_id]["images"]
    stats = session_data[session_id]["stats"]
    
    # Get original and segmentation image names
    original_images = sorted([name for name in images.keys() if name.startswith("original_")])
    segmentation_images = sorted([name for name in images.keys() if name.startswith("segmentation_")])
    
    # Get slice number from query params, default to 0
    selected_slice = request.args.get('slice', '0')
    try:
        selected_slice = int(selected_slice)
        if selected_slice < 0 or selected_slice >= len(original_images):
            selected_slice = 0
    except ValueError:
        selected_slice = 0
    
    return render_template(
        "results.html",
        session_id=session_id,
        num_slices=stats["num_slices"],
        selected_slice=selected_slice,
        original_image=original_images[selected_slice],
        segmentation_image=segmentation_images[selected_slice],
        tumor_percentage=stats["tumor_percentage"],
        tumor_volume_pixels=stats["tumor_volume_pixels"]
    )

@app.route("/data/<session_id>/<filename>")
def serve_image(session_id, filename):
    """Serve images from memory"""
    if session_id in session_data and filename in session_data[session_id]["images"]:
        image_data = session_data[session_id]["images"][filename]
        image_data.seek(0)  # Reset pointer to start of file
        return send_file(image_data, mimetype='image/png')
    return "Image not found", 404

@app.route("/cleanup/<session_id>", methods=["POST"])
def cleanup_session(session_id):
    """Cleanup session data"""
    if session_id in session_data:
        del session_data[session_id]
    
    return redirect(url_for('home'))

# Optional: Periodically clean up old sessions (e.g., using a background thread)
# This would prevent memory leaks from abandoned sessions

if __name__ == "__main__":
    app.run(debug=True)


