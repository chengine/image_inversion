import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy

class ImageProcessor:
    def __init__(self, image_path):
        self.image = self.rgb2gray(plt.imread(image_path))
        H, W = self.image.shape
        self.model = Model(H, W)

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def blur(self, mode='motion', kernel_size=20):
        if mode == 'avg':
            return self.model.avg_blur(self.image)
        elif mode == 'gaussian':
            return self.model.gaussian_blur(self.image)
        elif mode == 'motion':
            h = np.eye(kernel_size) / kernel_size
            return convolve2d(self.image, h, mode='valid')

    @staticmethod
    def add_gaussian_noise(img, sigma):
        gauss = np.random.normal(0, sigma, np.shape(img))
        noisy_img = img + gauss
        noisy_img = np.clip(noisy_img, 0, 255)
        return noisy_img

    def wiener_filter(self, img, kernel, K):
        kernel /= np.sum(kernel)
        dummy = fft2(img)
        kernel = fft2(kernel, s=img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)
        dummy = dummy * kernel
        return np.abs(ifft2(dummy))

    def process(self, kernel_size=20, sigma=120, K=30):
        motion_blurred_img = self.blur(mode='motion', kernel_size=kernel_size)
        noisy_img = self.add_gaussian_noise(motion_blurred_img, sigma=sigma)
        kernel = self.model.gaussian_kernel(kernel_size)
        filtered_img = self.wiener_filter(noisy_img, kernel, K=K)
        return [self.image, motion_blurred_img, noisy_img, filtered_img]

if __name__ == "__main__":
    processor = ImageProcessor(os.path.join('tiger.jpg'))
    display = processor.process()
    label = ['Original Image', 'Motion Blurred Image', 'Motion Blurring + Gaussian Noise', 'Wiener Filter applied']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, img, title in zip(axes.flatten(), display, label):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
    plt.show()