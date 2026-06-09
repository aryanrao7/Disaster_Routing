import rasterio
from rasterio.windows import Window

def crop_center_of_tiff(input_path, output_path, scale_factor=0.3):
    print(f"[System] Opening {input_path}...")
    
    with rasterio.open(input_path) as src:
        # Get original pixel dimensions
        width = src.width
        height = src.height
        
        # Calculate the new, smaller dimensions
        crop_width = int(width * scale_factor)
        crop_height = int(height * scale_factor)
        
        # Calculate offsets to grab the exact center
        col_off = (width - crop_width) // 2
        row_off = (height - crop_height) // 2
        
        # Define the cropping window
        window = Window(col_off, row_off, crop_width, crop_height)
        
        # Crucial: Update the GPS metadata so the new image knows exactly where it is!
        kwargs = src.meta.copy()
        kwargs.update({
            'height': crop_height,
            'width': crop_width,
            'transform': src.window_transform(window)
        })
        
        # Write the cropped data to a new file
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(src.read(window=window))
            
    print(f"[Success] Cropped image saved to {output_path}")
    print(f"[System] Size reduced to {scale_factor * 100}% of the original area.")

if __name__ == "__main__":
    input_tif = "test_data/la-fire.tif"
    output_tif = "test_data/la-fire-small.tif"
    
    # Change this from 0.3 to 0.05
    crop_center_of_tiff(input_tif, output_tif, scale_factor=0.05)