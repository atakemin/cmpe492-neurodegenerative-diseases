import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import zoom

def load_and_preprocess(path):
    img = nib.load(path)
    return img.get_fdata()

def normalize_image(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)

def resize_to_match(img_to_resize, target_shape):
    # Calculate zoom factors for each dimension
    zoom_factors = [t / s for t, s in zip(target_shape, img_to_resize.shape)]
    return zoom_factors, zoom(img_to_resize, zoom_factors, order=1) # Linear interpolation

def calculate_psnr_metric(img1, img2):
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim_metric(img1, img2):
    # For 3D, ensure we specify data_range and win_size
    return ssim(img1, img2, data_range=1.0)

def evaluate_mris(path1, path2):
    # 1. Load data
    data1 = load_and_preprocess(path1)
    data2 = load_and_preprocess(path2)
    
    # 2. Handle shape mismatch (Resize smaller to larger)
    if data1.shape != data2.shape:
        print(f"Shape mismatch detected: {data1.shape} vs {data2.shape}. Resizing...")
        if np.prod(data1.shape) < np.prod(data2.shape):
            _, data1 = resize_to_match(data1, data2.shape)
        else:
            _, data2 = resize_to_match(data2, data1.shape)
            
    # 3. Normalize
    data1 = normalize_image(data1)
    data2 = normalize_image(data2)
    
    # 4. Calculate Metrics
    ssim_val = calculate_ssim_metric(data1, data2)
    psnr_val = calculate_psnr_metric(data1, data2)
    
    return {
        "SSIM": ssim_val,
        "PSNR": psnr_val,
        "Final_Shape": data1.shape
    }