import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters, segmentation
from scipy import ndimage
import nibabel as nib  # For NIfTI format
try:
    import pydicom  # For DICOM format
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("pydicom not available. DICOM file support disabled.")

# Check if running in Google Colab
try:
    from google.colab import drive
    IN_COLAB = True
    print("Running in Google Colab environment")
    drive.mount('/content/drive')
    # Set your output path for Google Colab
    OUTPUT_DIR = '/content/drive/MyDrive/lung_analysis_results'
    # Set test image path
    TEST_IMAGE_PATH = '/content/drive/MyDrive/dataset/test/adenocarcinoma/000115.png'
except ImportError:
    IN_COLAB = False
    print("Running in local environment")
    # Set your local paths here
    OUTPUT_DIR = './lung_analysis_results'
    # Set test image path
    TEST_IMAGE_PATH = './dataset/test/adenocarcinomaa/000115.png'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(file_path):
    """Load image from various formats (PNG, JPG, DICOM, NIfTI)"""
    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        # Load regular image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        metadata = None

    elif ext == '.dcm' and DICOM_AVAILABLE:
        # Load DICOM
        dicom_data = pydicom.dcmread(file_path)
        image = dicom_data.pixel_array

        # Apply windowing if available
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            center = dicom_data.WindowCenter
            width = dicom_data.WindowWidth

            # Handle multiple window values
            if hasattr(center, '__iter__') and not isinstance(center, str):
                center = center[0]
            if hasattr(width, '__iter__') and not isinstance(width, str):
                width = width[0]

            image = window_image(image, center, width)

        metadata = dicom_data

    elif ext in ['.nii', '.nii.gz']:
        # Load NIfTI
        nifti_data = nib.load(file_path)
        image_data = nifti_data.get_fdata()

        # If 3D, take middle slice as default
        if len(image_data.shape) == 3:
            middle_slice = image_data.shape[2] // 2
            image = image_data[:, :, middle_slice]
        else:
            image = image_data

        metadata = nifti_data.header

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure image has correct data type
    image = image.astype(np.float32)

    # Normalize to [0, 1] if not already
    if image.max() > 1.0:
        image = image / image.max()

    return image, metadata

def window_image(image, center, width):
    """Apply windowing to DICOM image"""
    img_min = center - width // 2
    img_max = center + width // 2
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed

def segment_lungs(image):
    """Segment lung regions using intensity-based thresholding and morphological operations"""
    # Convert to binary using Otsu's thresholding
    threshold = filters.threshold_otsu(image)
    binary = image < threshold  # Lungs are typically darker than surrounding tissue

    # Apply morphological operations to clean up
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)

    # Find connected components
    labels = measure.label(cleaned)
    as = measure.regionprops(labels)

    # Sort regions by area and keep the largest ones (likely to be lungs)
    if len(props) < 2:
        # If less than 2 regions found, return original binary image
        return binary

    # Sort by area
    props = sorted(props, key=lambda x: x.area, reverse=True)

    # Create mask with the two largest regions (assumed to be lungs)
    lung_mask = np.zeros_like(image, dtype=bool)
    for i in range(min(2, len(props))):
        if props[i].area > 1000:  # Minimum size threshold
            lung_mask[labels == props[i].label] = True

    # Apply closing to fill holes in lung regions
    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(5))

    return lung_mask

def separate_lungs(lung_mask):
    """Separate left and right lungs"""
    # Label connected components
    labels = measure.label(lung_mask)
    regions = measure.regionprops(labels)

    if len(regions) < 2:
        # If only one region is found, try to separate using watershed
        distance = ndimage.distance_transform_edt(lung_mask)
        local_max = filters.peak_local_max(distance, min_distance=35, labels=lung_mask)
        markers = np.zeros_like(distance, dtype=np.int32)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        watershed_labels = segmentation.watershed(-distance, markers, mask=lung_mask)

        # Create separate masks for left and right lungs
        regions = measure.regionprops(watershed_labels)
        if len(regions) >= 2:
            # Sort by centroid x-coordinate to determine left and right
            regions = sorted(regions, key=lambda x: x.centroid[1])
            left_lung = watershed_labels == regions[0].label
            right_lung = watershed_labels == regions[1].label
            return left_lung, right_lung
        else:
            # If watershed fails, use simple left/right split
            h, w = lung_mask.shape
            left_half = np.zeros_like(lung_mask)
            right_half = np.zeros_like(lung_mask)
            left_half[:, :w//2] = lung_mask[:, :w//2]
            right_half[:, w//2:] = lung_mask[:, w//2:]
            return left_half, right_half
    else:
        # If multiple regions found, sort by centroid x-coordinate
        regions = sorted(regions, key=lambda x: x.centroid[1])
        left_lung = labels == regions[0].label
        right_lung = labels == regions[1].label
        return left_lung, right_lung

def draw_line(r0, c0, r1, c1):
    """Draw a line between two points"""
    # Simple implementation of Bresenham's line algorithm
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)

    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    err = dr - dc

    rr, cc = [], []
    r, c = r0, c0

    while True:
        rr.append(r)
        cc.append(c)

        if r == r1 and c == c1:
            break

        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return np.array(rr), np.array(cc)

def visualize_segmented_regions(original_image, left_lung, right_lung, background_mask=None, dark_areas=None):
    """
    Display each segmented region separately with clear backgrounds

    Args:
        original_image: Original CT scan image
        left_lung: Mask for left lung (blue)
        right_lung: Mask for right lung (purple)
        background_mask: Mask for mediastinum (orange)
        dark_areas: Mask for dark regions (green)

    Returns:
        matplotlib figure with 4 subplots, each showing one segmented region
    """
    # Create figure with subplots
    plt.figure(figsize=(20, 5))

    # If background_mask and dark_areas were not provided, compute them
    if background_mask is None or dark_areas is None:
        # Create background mask (everything that is not left or right lung)
        background_mask = ~(left_lung | right_lung)

        # Identify dark areas within background
        if len(original_image.shape) == 2:
            # For grayscale image
            dark_threshold = 0.3  # Adjust this threshold as needed
            dark_areas = (original_image < dark_threshold) & background_mask
        else:
            # For RGB image, use average intensity
            intensity = np.mean(original_image, axis=2) if len(original_image.shape) == 3 else original_image
            dark_threshold = 0.3  # Adjust this threshold as needed
            dark_areas = (intensity < dark_threshold) & background_mask

        # Remaining background (not dark areas)
        orange_areas = background_mask & ~dark_areas
    else:
        orange_areas = background_mask & ~dark_areas

    # Apply morphological operations to remove noise
    # Use opening operation to remove small objects (noise)
    kernel_size = 3
    kernel = morphology.disk(kernel_size)

    left_lung_clean = morphology.binary_opening(left_lung, kernel)
    right_lung_clean = morphology.binary_opening(right_lung, kernel)
    orange_areas_clean = morphology.binary_opening(orange_areas, kernel)
    dark_areas_clean = morphology.binary_opening(dark_areas, kernel)

    # Also fill small holes
    left_lung_clean = morphology.remove_small_holes(left_lung_clean, area_threshold=100)
    right_lung_clean = morphology.remove_small_holes(right_lung_clean, area_threshold=100)
    orange_areas_clean = morphology.remove_small_holes(orange_areas_clean, area_threshold=100)
    dark_areas_clean = morphology.remove_small_holes(dark_areas_clean, area_threshold=100)

    # Create a foggy green background (RGB: 144, 238, 144 - light green)
    # Scale to 0-1 for matplotlib
    bg_color = np.array([144/255, 238/255, 144/255])

    # Create visualizations with foggy green background
    if len(original_image.shape) == 2:
        # For grayscale images

        # Prepare RGB versions of the image
        image_rgb = np.stack([original_image, original_image, original_image], axis=-1)

        # Create base images with foggy green background
        left_lung_vis = np.ones((*original_image.shape, 3)) * bg_color[None, None, :]
        right_lung_vis = np.ones((*original_image.shape, 3)) * bg_color[None, None, :]
        background_vis = np.ones((*original_image.shape, 3)) * bg_color[None, None, :]
        dark_vis = np.ones((*original_image.shape, 3)) * bg_color[None, None, :]

        # Add image content only where the masks are True
        left_lung_vis[left_lung_clean] = image_rgb[left_lung_clean]
        right_lung_vis[right_lung_clean] = image_rgb[right_lung_clean]
        background_vis[orange_areas_clean] = image_rgb[orange_areas_clean]
        dark_vis[dark_areas_clean] = image_rgb[dark_areas_clean]

    else:
        # For RGB images

        # Create base images with foggy green background
        left_lung_vis = np.ones((*original_image.shape[:2], 3)) * bg_color[None, None, :]
        right_lung_vis = np.ones((*original_image.shape[:2], 3)) * bg_color[None, None, :]
        background_vis = np.ones((*original_image.shape[:2], 3)) * bg_color[None, None, :]
        dark_vis = np.ones((*original_image.shape[:2], 3)) * bg_color[None, None, :]

        # Add image content only where the masks are True
        for c in range(3):  # RGB channels
            left_lung_vis[..., c] = np.where(left_lung_clean, original_image[..., c], left_lung_vis[..., c])
            right_lung_vis[..., c] = np.where(right_lung_clean, original_image[..., c], right_lung_vis[..., c])
            background_vis[..., c] = np.where(orange_areas_clean, original_image[..., c], background_vis[..., c])
            dark_vis[..., c] = np.where(dark_areas_clean, original_image[..., c], dark_vis[..., c])

    # Add contours to better highlight the regions
    # Left lung
    left_contours = measure.find_contours(left_lung_clean.astype(np.uint8), 0.5)
    for contour in left_contours:
        for i in range(len(contour) - 1):
            rr, cc = draw_line(int(contour[i, 0]), int(contour[i, 1]),
                              int(contour[i+1, 0]), int(contour[i+1, 1]))
            try:
                left_lung_vis[rr, cc] = [0, 0, 1]  # Blue contour
            except IndexError:
                pass

    # Right lung
    right_contours = measure.find_contours(right_lung_clean.astype(np.uint8), 0.5)
    for contour in right_contours:
        for i in range(len(contour) - 1):
            rr, cc = draw_line(int(contour[i, 0]), int(contour[i, 1]),
                              int(contour[i+1, 0]), int(contour[i+1, 1]))
            try:
                right_lung_vis[rr, cc] = [0.5, 0, 0.8]  # Purple contour
            except IndexError:
                pass

    # Background
    bg_contours = measure.find_contours(orange_areas_clean.astype(np.uint8), 0.5)
    for contour in bg_contours:
        for i in range(len(contour) - 1):
            rr, cc = draw_line(int(contour[i, 0]), int(contour[i, 1]),
                              int(contour[i+1, 0]), int(contour[i+1, 1]))
            try:
                background_vis[rr, cc] = [1, 0.5, 0]  # Orange contour
            except IndexError:
                pass

    # Dark areas
    dark_contours = measure.find_contours(dark_areas_clean.astype(np.uint8), 0.5)
    for contour in dark_contours:
        for i in range(len(contour) - 1):
            rr, cc = draw_line(int(contour[i, 0]), int(contour[i, 1]),
                              int(contour[i+1, 0]), int(contour[i+1, 1]))
            try:
                dark_vis[rr, cc] = [0, 0.7, 0]  # Green contour
            except IndexError:
                pass

    # Left lung (blue)
    plt.subplot(1, 4, 1)
    plt.imshow(left_lung_vis)
    plt.title('Left Lung')
    plt.axis('off')

    # Right lung (purple)
    plt.subplot(1, 4, 2)
    plt.imshow(right_lung_vis)
    plt.title('Right Lung')
    plt.axis('off')

    # Mediastinum (orange)
    plt.subplot(1, 4, 3)
    plt.imshow(background_vis)
    plt.title('Mediastinum')
    plt.axis('off')

    # Dark regions (green)
    plt.subplot(1, 4, 4)
    plt.imshow(dark_vis)
    plt.title('Dark Regions')
    plt.axis('off')

    plt.tight_layout()
    return plt.gcf()

def save_individual_segments(image, left_lung, right_lung, filename):
    """Save each segmented region as a separate image with proper coloring"""
    # Get base filename without extension
    base_filename = os.path.splitext(filename)[0]

    # Create background mask (everything that is not left or right lung)
    background_mask = ~(left_lung | right_lung)

    # Identify dark areas within background
    if len(image.shape) == 2:
        # For grayscale image
        dark_threshold = 0.3  # Adjust this threshold as needed
        dark_areas = (image < dark_threshold) & background_mask
    else:
        # For RGB image, use average intensity
        intensity = np.mean(image, axis=2) if len(image.shape) == 3 else image
        dark_threshold = 0.3  # Adjust this threshold as needed
        dark_areas = (intensity < dark_threshold) & background_mask

    # Remaining background (not dark areas) - this is mediastinum
    mediastinum = background_mask & ~dark_areas

    # Apply morphological operations to remove noise
    kernel_size = 3
    kernel = morphology.disk(kernel_size)

    left_lung_clean = morphology.binary_opening(left_lung, kernel)
    right_lung_clean = morphology.binary_opening(right_lung, kernel)
    mediastinum_clean = morphology.binary_opening(mediastinum, kernel)
    dark_areas_clean = morphology.binary_opening(dark_areas, kernel)

    # Also fill small holes
    left_lung_clean = morphology.remove_small_holes(left_lung_clean, area_threshold=100)
    right_lung_clean = morphology.remove_small_holes(right_lung_clean, area_threshold=100)
    mediastinum_clean = morphology.remove_small_holes(mediastinum_clean, area_threshold=100)
    dark_areas_clean = morphology.remove_small_holes(dark_areas_clean, area_threshold=100)

    # Create RGB image if the original is grayscale
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    else:
        rgb_image = image.copy()

    # Create separate images for each segment
    # 1. Left Lung (blue)
    left_lung_img = rgb_image.copy()
    alpha = 0.3  # Transparency level
    for r in range(left_lung_img.shape[0]):
        for c in range(left_lung_img.shape[1]):
            if left_lung_clean[r, c]:
                left_lung_img[r, c] = (1-alpha) * left_lung_img[r, c] + alpha * np.array([0, 0, 1])  # Blue tint
            else:
                left_lung_img[r, c] = np.array([0, 0, 0])  # Black background

    # 2. Right Lung (purple)
    right_lung_img = rgb_image.copy()
    for r in range(right_lung_img.shape[0]):
        for c in range(right_lung_img.shape[1]):
            if right_lung_clean[r, c]:
                right_lung_img[r, c] = (1-alpha) * right_lung_img[r, c] + alpha * np.array([0.5, 0, 0.8])  # Purple tint
            else:
                right_lung_img[r, c] = np.array([0, 0, 0])  # Black background

    # 3. Mediastinum (orange)
    mediastinum_img = rgb_image.copy()
    for r in range(mediastinum_img.shape[0]):
        for c in range(mediastinum_img.shape[1]):
            if mediastinum_clean[r, c]:
                mediastinum_img[r, c] = (1-alpha) * mediastinum_img[r, c] + alpha * np.array([1, 0.5, 0])  # Orange tint
            else:
                mediastinum_img[r, c] = np.array([0, 0, 0])  # Black background

    # 4. Dark regions (green)
    dark_img = rgb_image.copy()
    for r in range(dark_img.shape[0]):
        for c in range(dark_img.shape[1]):
            if dark_areas_clean[r, c]:
                dark_img[r, c] = (1-alpha) * dark_img[r, c] + alpha * np.array([0, 0.7, 0])  # Green tint
            else:
                dark_img[r, c] = np.array([0, 0, 0])  # Black background

    # Save individual segment images
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_left_lung.png"), cv2.cvtColor((left_lung_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_right_lung.png"), cv2.cvtColor((right_lung_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_mediastinum.png"), cv2.cvtColor((mediastinum_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_dark_regions.png"), cv2.cvtColor((dark_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Create a combined segmentation image with all segments colored
    combined = rgb_image.copy()
    # Add each segment with its color
    for r in range(combined.shape[0]):
        for c in range(combined.shape[1]):
            if left_lung_clean[r, c]:
                combined[r, c] = (1-alpha) * combined[r, c] + alpha * np.array([0, 0, 1])  # Blue
            elif right_lung_clean[r, c]:
                combined[r, c] = (1-alpha) * combined[r, c] + alpha * np.array([0.5, 0, 0.8])  # Purple
            elif mediastinum_clean[r, c]:
                combined[r, c] = (1-alpha) * combined[r, c] + alpha * np.array([1, 0.5, 0])  # Orange
            elif dark_areas_clean[r, c]:
                combined[r, c] = (1-alpha) * combined[r, c] + alpha * np.array([0, 0.7, 0])  # Green

    # Save the combined image
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_all_segments.png"), cv2.cvtColor((combined*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    return {
        'left_lung_image': left_lung_img,
        'right_lung_image': right_lung_img,
        'mediastinum_image': mediastinum_img,
        'dark_regions_image': dark_img,
        'combined_image': combined
    }

def segment_and_visualize(image_path):
    """Segment lungs and visualize results"""
    # Load image
    image, _ = load_image(image_path)

    # Segment lungs
    lung_mask = segment_lungs(image)

    # Separate left and right lungs
    left_lung, right_lung = separate_lungs(lung_mask)

    # Create visualization
    fig = visualize_segmented_regions(image, left_lung, right_lung)

    # Save results
    filename = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_DIR, f"segmented_{filename}")
    fig.savefig(output_path)
    plt.close(fig)

    # Save individual segmented images
    segment_images = save_individual_segments(image, left_lung, right_lung, filename)

    print(f"Segmentation complete. Results saved to {OUTPUT_DIR}")

    return {
        'image': image,
        'lung_mask': lung_mask,
        'left_lung': left_lung,
        'right_lung': right_lung,
        'segment_images': segment_images
    }

if __name__ == "__main__":
    # Process a single test image
    results = segment_and_visualize(TEST_IMAGE_PATH)

    # Display results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(results['image'], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(results['segment_images']['combined_image'])
    plt.title('All Segments')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(results['segment_images']['left_lung_image'])
    plt.title('Left Lung (Blue)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(results['segment_images']['right_lung_image'])
    plt.title('Right Lung (Purple)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(results['segment_images']['mediastinum_image'])
    plt.title('Mediastinum (Orange)')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(results['segment_images']['dark_regions_image'])
    plt.title('Dark Regions (Green)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
