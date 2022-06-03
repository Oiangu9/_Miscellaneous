# Xabier Oyanguren Asua 1456628

import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. CAPUTE THE FRAMES ###########################
video_good = 'good_light.mp4'
video_bad = 'bad_light.mp4'

vidcap = cv2.VideoCapture(video_good)
# 1.1 get the parameters for the captured images
succ, im = vidcap.read()
im_type = im.dtype
max_val = 2**8-1 if im_type==np.uint8 else 2**16-1 if im_type==np.uint16 else 2**32-1 if im_type==np.uint32 else 2**64-1 if im_type==np.uint64 else None
im_ds_good = np.zeros((200, im.shape[0], im.shape[1]), dtype=im_type)

# 1.2 Video with good lighting
# restart video capture
print(f"\n\nReading frames of {video_good}:#####")
vidcap = cv2.VideoCapture(video_good)
for i in range(200):
    succ, im = vidcap.read()
    if not succ:
        raise ValueError
    print(f"Succesfully read frame {i} of size {im.shape}, and type {im.dtype}")
    if len(im.shape)>2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_ds_good[i] = im

# 1.2 Video with bad lighting
# presumably both videos are of the exact same device and data characteristics
print(f"\n\nReading frames of {video_bad}:#####")
vidcap = cv2.VideoCapture(video_bad)
im_ds_bad = np.zeros((200, im.shape[0], im.shape[1]), dtype=im_type)

for i in range(200):
    succ, im = vidcap.read()
    if not succ:
        raise ValueError
    print(f"Succesfully read frame {i} of size {im.shape}, and type {im.dtype}")
    if len(im.shape)>2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_ds_bad[i] = im



# 2. COMPUTE THE DIM NOISE AND SMALL AND WIDE RANGE IMAGES ####################
im_ds_good_f = im_ds_good.astype(np.float64)
im_ds_bad_f = im_ds_bad.astype(np.float64)

# 2.1 the three images for the good illumination case
image_dim_noise_g_f = np.mean(im_ds_good_f[:100], axis=0)
small_range_g_f = im_ds_good_f[101]
wide_range_g_f = np.sum(im_ds_good_f[100:], axis=0)

# 2.2 the three for the bad lighting case
image_dim_noise_b_f = np.mean(im_ds_bad_f[:100], axis=0)
small_range_b_f = im_ds_bad_f[101]
wide_range_b_f = np.sum(im_ds_bad_f[100:], axis=0)



# 3. COMPUTE THE SNR AND PSNR FOR EACH CASE AND PLOT THE IMAGES AND COMPUTATIONS ###########
# 3.1 Compuations and plots for the good lighting case
fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(231)
ax.imshow( image_dim_noise_g_f.astype(im_type), cmap='gray')
ax.set_title("(c) Image considered as no-noise reference\nIt is the average of 100 good light")
ax = fig.add_subplot(234)
ax.imshow( (image_dim_noise_g_f-image_dim_noise_g_f).astype(im_type), cmap='gray')
ax.set_title("Noise of (c) relative to (c)\nSNR=infinite\nPSNR=infinite")
ax = fig.add_subplot(232)
ax.imshow( small_range_g_f.astype(im_type), cmap='gray')
ax.set_title("(a) Small range image\nIt is a single frame (the 101-th) of good light video")
ax = fig.add_subplot(235)
diff = np.abs(small_range_g_f-image_dim_noise_g_f)
snr=max_val/(2*np.std(diff))
psnr=10*np.log10(max_val**2/np.mean(diff**2))
ax.imshow( (diff).astype(im_type), cmap='gray')
ax.set_title(f"Noise in (a) relative to (c)\nSNR={snr:.3f}\nPSNR={psnr:.3f}")
ax = fig.add_subplot(233)
ax.imshow( (wide_range_g_f/100).astype(im_type), cmap='gray')
ax.set_title("(b) Wide range image normalized\nIt is the average of 100 frames (the 100:200) good light")
ax = fig.add_subplot(236)
diff = np.abs(wide_range_g_f/100-image_dim_noise_g_f)
snr=max_val/(2*np.std(diff))
psnr=10*np.log10(max_val**2/np.mean(diff**2))
ax.imshow( (diff).astype(im_type), cmap='gray')
ax.set_title(f"Noise in (b) relative to (c)\nSNR={snr:.3f}\nPSNR={psnr:.3f}")
fig.suptitle("GOOD LIGHT PROBLEM vs GOOD LIGHT REFERENCE")
fig.subplots_adjust(wspace=1.4, hspace=0.3)
plt.savefig("Good_vs_Good_lighting.png")
plt.show()

# 3.2 Compuations and plots for the bad lighting case
fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(231)
ax.imshow( image_dim_noise_b_f.astype(im_type), cmap='gray')
ax.set_title("(c) Image considered as no-noise reference\nIt is the average of 100 bad light")
ax = fig.add_subplot(234)
ax.imshow( (image_dim_noise_b_f-image_dim_noise_b_f).astype(im_type), cmap='gray')
ax.set_title("Noise of (c) relative to (c)\nSNR=infinite\nPSNR=infinite")
ax = fig.add_subplot(232)
ax.imshow( small_range_b_f.astype(im_type), cmap='gray')
ax.set_title("(a) Small range image\nIt is a single frame (the 101-th) of bad light video")
ax = fig.add_subplot(235)
diff = np.abs(small_range_b_f-image_dim_noise_b_f)
snr=max_val/(2*np.std(diff))
psnr=10*np.log10(max_val**2/np.mean(diff**2))
ax.imshow( (diff).astype(im_type), cmap='gray')
ax.set_title(f"Noise in (a) relative to (c)\nSNR={snr:.3f}\nPSNR={psnr:.3f}")
ax = fig.add_subplot(233)
ax.imshow( (wide_range_b_f/100).astype(im_type), cmap='gray')
ax.set_title("(b) Wide range image normalized\nIt is the average of 100 frames (the 100:200) bad light")
ax = fig.add_subplot(236)
diff = np.abs(wide_range_b_f/100-image_dim_noise_b_f)
snr=max_val/(2*np.std(diff))
psnr=10*np.log10(max_val**2/np.mean(diff**2))
ax.imshow( (diff).astype(im_type), cmap='gray')
ax.set_title(f"Noise in (b) relative to (c)\nSNR={snr:.3f}\nPSNR={psnr:.3f}")
fig.suptitle("BAD LIGHT PROBLEM vs BAD LIGHT REFERENCE")
fig.subplots_adjust(wspace=1.4, hspace=0.3)
plt.savefig("Bad_vs_Bad_lighting.png")
plt.show()

'''
# 3.3 Compuations and plots for the bad lighting case relative to the good lighting "no-noise" image
fig = plt.figure(figsize=(20,17))
ax = fig.add_subplot(231)
ax.imshow( image_dim_noise_g_f.astype(im_type), cmap='gray')
ax.set_title("(c) Image considered as no-noise reference\nIt is the average of 100 good lights")
ax = fig.add_subplot(234)
ax.imshow( (image_dim_noise_g_f-image_dim_noise_g_f).astype(im_type), cmap='gray')
ax.set_title("Noise of (c) relative to (c)\nSNR=infinite\nPSNR=infinite")
ax = fig.add_subplot(232)
ax.imshow( small_range_b_f.astype(im_type), cmap='gray')
ax.set_title("(a) Small range image\nIt is a single frame (the 101-th) of bad light video")
ax = fig.add_subplot(235)
diff = np.abs(small_range_b_f-image_dim_noise_g_f)
snr=max_val/(2*np.std(diff))
psnr=10*np.log10(max_val**2/np.mean(diff**2))
ax.imshow( (diff).astype(im_type), cmap='gray')
ax.set_title(f"Noise in (a) relative to (c)\nSNR={snr:.3f}\nPSNR={psnr:.3f}")
ax = fig.add_subplot(233)
ax.imshow( (wide_range_b_f/100).astype(im_type), cmap='gray')
ax.set_title("(b) Wide range image normalized\nIt is the average of 100 frames (the 100:200) good light")
ax = fig.add_subplot(236)
diff = np.abs(wide_range_b_f/100-image_dim_noise_g_f)
snr=max_val/(2*np.std(diff))
psnr=10*np.log10(max_val**2/np.mean(diff**2))
ax.imshow( (diff).astype(im_type), cmap='gray')
ax.set_title(f"Noise in (b) relative to (c)\nSNR={snr:.3f}\nPSNR={psnr:.3f}")
fig.suptitle("BAD LIGHT PROBLEM vs GOOD LIGHT REFERENCE")
fig.subplots_adjust(wspace=1.4, hspace=0.3)
plt.savefig("Bad_vs_Good_lighting.png")
plt.show()
'''



