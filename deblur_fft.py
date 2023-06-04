import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageProcessor:
    def __init__(self, img_path, noise_sigma, filter_sigma):
        self.noise_sigma = noise_sigma
        self.filter_sigma = filter_sigma
        self.img = self.load_image(img_path)
        self.n, self.m = self.img.shape

    def load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))
        return np.array(img, dtype=np.float64) / 255.0

    def gaussian_filter(self):
        return cv2.getGaussianKernel(self.n, self.filter_sigma) @ cv2.getGaussianKernel(self.m, self.filter_sigma).T

    def fourier_transform(self, arr):
        return np.fft.fft2(arr)

    def blur_image(self):
        F = self.fourier_transform(self.img)
        H = self.fourier_transform(self.gaussian_filter())
        return np.abs(np.fft.ifft2(F * H))

    def add_noise(self, arr):
        noisy_img = arr + self.noise_sigma * np.random.randn(*arr.shape)
        noisy_img[noisy_img < 0] = 0
        noisy_img[noisy_img > 255] = 255
        return noisy_img

    def compute_reconstruction(self, g, H):
        G = self.fourier_transform(g)
        Fest = G / (H + np.finfo(float).eps)
        return np.abs(np.fft.ifft2(Fest))

    def run_EM_algorithm(self, g, H):
        nrip = 200
        err = np.zeros(nrip)
        fk = g.copy()
        normf = np.linalg.norm(self.img)
        bestRec = np.zeros_like(self.img)
        bestK = 1
        errMin = 1

        normDif = np.linalg.norm(g - self.img)
        startErr = normDif / normf

        for i in range(nrip):
            FK = self.fourier_transform(fk)
            D = FK * H
            d = np.fft.ifft2(D)

            n = g / (np.abs(d) + np.finfo(float).eps)
            N = self.fourier_transform(n) * np.conj(H)

            a = np.fft.ifft2(N)
            fk1 = fk * np.abs(a)

            fk = fk1

            normDif = np.linalg.norm(fk1 - self.img)
            err[i] = normDif / normf

            if err[i] < errMin:
                bestRec = fk1
                errMin = err[i]
                bestK = i

        perc = (startErr - errMin) * 100
        return err, bestRec, perc

    def plot_results(self, img, title):
        plt.figure()
        plt.imshow(np.abs(img), cmap='gray')
        plt.title(title)
        plt.show()


def main():
    img_path = 'tiger.jpg'
    noise_sigma = 0.01
    filter_sigma = 1

    image_processor = ImageProcessor(img_path, noise_sigma, filter_sigma)

    blurred_image = image_processor.blur_image()
    noisy_image = image_processor.add_noise(blurred_image)

    image_processor.plot_results(image_processor.img, 'Original Image')
    image_processor.plot_results(blurred_image, 'Blurred Image')
    image_processor.plot_results(noisy_image, 'Noisy Image')

    H = image_processor.fourier_transform(image_processor.gaussian_filter())
    reconstructed_image = image_processor.compute_reconstruction(
        noisy_image, H)
    err, bestRec, perc = image_processor.run_EM_algorithm(noisy_image, H)

    image_processor.plot_results(
        bestRec, f'Best reconstruction, image recovered={perc}%')

    plt.figure()
    plt.plot(np.linspace(1, 200, 200), err)
    plt.grid(True)
    plt.title('Error of the reconstruction')
    plt.legend(['Reconstruction error'])
    plt.show()


if __name__ == "__main__":
    main()
