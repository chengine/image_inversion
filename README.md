# Image Inversion

3 blurring models (linear: motion blur, average blur, gaussian blur), grayscale conversion, and nonlinear transformation (square root) comprise the forward model. The inversion problem is performed first on a grayscale image using the pseudo-inverse. Then PCGA is used to estimate the nonlinear transformation. Finally, the full color image is estimated.
