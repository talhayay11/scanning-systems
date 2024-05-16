import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.io import imread
from skimage.color import rgb2gray

# Görüntüyü yükle
image_path = 'indir.jpeg'  # Buraya görüntü dosyanızın yolunu yazın
image = imread(image_path)

# Görüntüyü gri tonlamalıya dönüştür (Radon dönüşümü için gereklidir)
image_gray = rgb2gray(image)

# Görüntüye Radon dönüşümü uygula
theta = np.linspace(0., 180., max(image_gray.shape), endpoint=False)
sinogram = radon(image_gray, theta=theta)

# Radon dönüşümünden geri dönüşüm (inverse Radon dönüşümü) yaparak orijinal görüntüyü elde et
reconstructed_image = iradon(sinogram, theta=theta, filter_name='ramp')

# Orijinal görüntüyü, Radon dönüşümünü (sinogram) ve geri dönüştürülmüş görüntüyü görselleştir
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 8))

ax1.set_title("Orijinal Görüntü")
ax1.imshow(image , cmap=plt.cm.Greys_r)

ax2.set_title("Griye dönüştürülmüş Görüntü")
ax2.imshow(image_gray, cmap=plt.cm.Greys_r)

ax3.set_title("Radon Dönüşümü\n(Sinogram)")
ax3.set_xlabel("Projeksiyon açısı (derece)")
ax3.set_ylabel("Radyal koordinat (piksel)")
ax3.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

ax4.set_title("Geri Dönüştürülmüş Görüntü")
ax4.imshow(reconstructed_image, cmap=plt.cm.Greys_r)

plt.tight_layout()
plt.show()