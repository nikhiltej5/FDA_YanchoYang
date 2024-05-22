import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np

# Load source and target images
im_src = Image.open("demo_images/source.png").convert('RGB')
im_trg = Image.open("demo_images/target.png").convert('RGB')

# Resize images
im_src = im_src.resize((1024, 512), Image.BICUBIC)
im_trg = im_trg.resize((1024, 512), Image.BICUBIC)

# Convert images to numpy arrays
im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

# Transpose arrays to have channel-first format
im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

# Perform Fourier Domain Adaptation (FDA)
src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.01)

# Transpose back to height-width-channel format
src_in_trg = src_in_trg.transpose((1, 2, 0))

# Clip values to the valid range [0, 255] and convert to uint8
src_in_trg = np.clip(src_in_trg, 0, 255).astype(np.uint8)

# Convert the numpy array to a PIL Image and save it
Image.fromarray(src_in_trg).save('demo_images/src_in_tar.png')
