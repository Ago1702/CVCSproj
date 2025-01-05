import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Function to select a random image from a directory and check its dimensions
def check_image_dimensions(image_path):
    # Open the image using Pillow
    with Image.open(image_path) as img:
        width, height = img.size
        if width > 5000:
            print(image_path)
            return
        if height > 5000:
            print(image_path)
            return

# Function to process the image index
def process_image(i):
    image_path = '/work/cvcs2024/VisionWise/train/image-real-' + str(i).zfill(8) + '.png'
    check_image_dimensions(image_path=image_path)
    if (i + 1) % 1000 == 0:
        print(i + 1)

def main():
    # Use ProcessPoolExecutor to parallelize the work across 4 cores
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(process_image, range(700000, 1000000))

if __name__ == '__main__':
    main()
