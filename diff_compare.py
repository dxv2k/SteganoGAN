from PIL import Image
import numpy as np

# Load images
cover = np.array(Image.open("/home/dxv2k/stegnography/Week6/SteganoGAN/cola.jpg"))
stego = np.array(Image.open("./encoded_output.png"))

# Calculate the difference
diff = stego - cover

# Find modified pixel locations
modified_pixels = np.where(diff != 0)
for row, col in zip(modified_pixels[0], modified_pixels[1]):
    cover_pixel = cover[row, col]
    stego_pixel = stego[row, col]
    for channel in range(3):  # RGB channels
        cover_byte = cover_pixel[channel]
        stego_byte = stego_pixel[channel]
        # XOR to highlight changed bits
        diff_bits = bin(stego_byte ^ cover_byte)[2:].zfill(8)
        print(f"Pixel ({row}, {col}), Channel {channel}: Modified bits - {diff_bits}")