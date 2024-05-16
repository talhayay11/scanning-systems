import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, iradon_sart
from skimage.io import imread
from skimage.color import rgb2gray
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def apply_radon_transform(image):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    reconstructed_image = iradon(sinogram, theta=theta, filter_name='ramp', circle=False)
    return reconstructed_image, sinogram

def apply_fourier_transform(image):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_transform_shifted))
    f_ishift = ifftshift(f_transform_shifted)
    reconstructed_image = np.abs(ifft2(f_ishift))
    return reconstructed_image, magnitude_spectrum

def visualize_results(image, original, radon_reconstructed, radon_sinogram, fourier_reconstructed, fourier_spectrum):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].set_title("Orijinal Görüntü")
    axes[0, 0].imshow(image, cmap=plt.cm.Greys_r)
    axes[0, 0].axis('off')

    # Orijinal görüntü
    axes[0, 1].set_title("Gray Görüntü")
    axes[0, 1].imshow(original, cmap=plt.cm.Greys_r)
    axes[0, 1].axis('off')

    # Radon dönüşümü (sinogram)
    axes[0, 2].set_title("Radon Dönüşümü\n(Sinogram)")
    axes[0, 2].imshow(radon_sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, radon_sinogram.shape[0]), aspect='auto')
    axes[0, 2].axis('off')

    # Radon ile geri dönüştürülmüş görüntü
    axes[1, 0].set_title("Radon ile Geri Dönüşüm")
    axes[1, 0].imshow(radon_reconstructed, cmap=plt.cm.Greys_r)
    axes[1, 0].axis('off')

    # Fourier dönüşümünün büyüklük spektrumu
    axes[1, 1].set_title("Fourier Dönüşümü\n(Büyüklük Spektrumu)")
    axes[1, 1].imshow(fourier_spectrum, cmap=plt.cm.Greys_r)
    axes[1, 1].axis('off')

    # Fourier ile geri dönüştürülmüş görüntü
    axes[1, 2].set_title("Fourier ile Geri Dönüşüm")
    axes[1, 2].imshow(fourier_reconstructed, cmap=plt.cm.Greys_r)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Görüntüyü yükle
    image_path = 'indir.jpeg'  # Gerçek dosya yolunu belirtin
    image = imread(image_path)
    image_gray = rgb2gray(image)

    # Radon dönüşümü ve geri dönüşüm
    radon_reconstructed, radon_sinogram = apply_radon_transform(image_gray)

    # Fourier dönüşümü ve geri dönüşüm
    fourier_reconstructed, fourier_spectrum = apply_fourier_transform(image_gray)

    # Sonuçları görselleştir
    visualize_results(image, image_gray, radon_reconstructed, radon_sinogram, fourier_reconstructed, fourier_spectrum)

if __name__ == "__main__":
    main()
