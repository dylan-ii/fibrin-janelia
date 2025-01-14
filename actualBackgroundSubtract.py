import numpy as np
import os
from glob import glob
import tifffile as tiff
import cv2
import concurrent.futures

input_directory = "Experiment15Data/"  # Adjust this to be your raw tif directory
output_directory = "Experiment15Background/"  # Replace with your output directory

gausRadius = 30 # Adjust this to blur radius

max_workers = 4 # Number of cores on CPU - 2, probably leave at 4 unless you have a better processor

def process_image(input_path, output_path, sigma):
    volume = tiff.imread(input_path)

    if volume is None or len(volume.shape) != 3:
        print(f"Error reading 3D volume from {input_path}")
        return

    subtracted_volume = np.zeros_like(volume, dtype=np.uint8)

    for z in range(volume.shape[0]):
        slice_2d = volume[z]

        blurred_slice = cv2.GaussianBlur(slice_2d, (0, 0), sigma)
        subtracted_slice = cv2.subtract(slice_2d, blurred_slice)

        subtracted_volume[z] = np.clip(subtracted_slice, 0, 255).astype(np.uint8)

    print(f"Processed and saved: {output_path}")

    tiff.imwrite(output_path, subtracted_volume)

def process_directory(input_dir, output_dir, sigma=30, max_workers=4):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tiff_files = glob(os.path.join(input_dir, "*.tif"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, file_path, os.path.join(output_dir, os.path.basename(file_path)), sigma)
            for file_path in tiff_files
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(f"Processed {len(tiff_files)} volumes.")

process_directory(input_directory, output_directory, sigma=gausRadius, max_workers=6)