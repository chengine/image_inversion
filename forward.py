#%% 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.sparse.linalg as linalg
import scipy

def find_POI(img_rgb, render=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    if render:
        feat_img = cv2.drawKeypoints(img_gray, keypoints, img)
    else:
        feat_img = None

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)

    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)

    extras = {
    'features': feat_img
    }

    return xy, extras # pixel coordinates

class Model():
    def __init__(self, H, W) -> None:
        self.H = H
        self.W = W

        # Initializations
        
    def motion_blur_H(self):
        N = int(round(0.05 * self.H, 0))

        mask = np.zeros((self.H, self.H))
        for r, row in enumerate(mask[0:-N]):
            row[r:r+N] = [round(1/N, 2)]*N

        self.mb_H = mask

        return mask

    def avg_blur_H(self):
        vec_dim = self.H*self.W

        test_vec = np.zeros(vec_dim)

        K = []

        for i in range(self.H*self.W):
            test_vec[i] = 1
            test_img = test_vec.reshape(self.H, self.W)
            output_img = self.avg_blur(test_img)
            sparse_output = scipy.sparse.csc_matrix(output_img.reshape(-1, 1))
            K.append(sparse_output)
            test_vec[i] = 0

        K = scipy.sparse.hstack(K)
  
        self.ab_H = K

        return K

    def gaussian_blur_H(self):
        vec_dim = self.H*self.W

        test_vec = np.zeros(vec_dim)

        K = []

        for i in range(self.H*self.W):
            test_vec[i] = 1
            test_img = test_vec.reshape(self.H, self.W)
            output_img = self.gaussian_blur(test_img)
            K.append(scipy.sparse.csc_matrix(output_img.reshape(-1, 1)))
            test_vec[i] = 0

        K = scipy.sparse.hstack(K)
  
        self.gb_H = K

        return K

    def cvt_grayscale(self, img):
        # Perform gray scaling
        img = 0.2989*img[..., 0] + 0.5870*img[..., 1] + 0.1140*img[..., 2]

        return img

    def motion_blur(self, img):
        flattened_img = img.flatten()
        
        blurred_img = self.mb_H @ flattened_img
        blurred_img = blurred_img.reshape(img.shape)

        # normalize img to [0,1]
        img = (blurred_img - blurred_img.min()) / (blurred_img.max()-blurred_img.min())

        return img 
    
    def avg_blur(self, img):
        # img = img.permute(2, 0, 1)
        # img = img.unsqueeze(0)

        # img = torch.nn.functional.avg_pool2d(img, 5, stride=1, padding=2, count_include_pad=False)
        # img = img.squeeze()
        # img = img.permute(1, 2, 0)

        img = cv2.blur(img, (5, 5), cv2.BORDER_DEFAULT)
    
        return img
    
    def gaussian_blur(self, img):

        # Perform Gaussian blur on img
        img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

        return img

    def nonlinear_blur(self, img):

        return img**(1/2)

    def prior(self, gt_img, Nsamples=100000):
        # Returns a prior on the image (e.g. return a subset of ground-truth pixels just randoming sampling)

        W_obs = gt_img.shape[0]
        H_obs = gt_img.shape[1]

        if Nsamples is None:
            Nsamples = int(W_obs * H_obs / 10)

        elif Nsamples > W_obs * H_obs:
            Nsamples = int(W_obs * H_obs)

        # find points of interest of the observed image
        POI, extras = find_POI(gt_img)  # xy pixel coordinates of points of interest (N x 2)

        print(f'Found {POI.shape[0]} features')
     
        gt_img = (np.array(gt_img) / 255.).astype(np.float32)

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, H_obs - 1, H_obs), np.linspace(0, W_obs - 1, W_obs)), -1), dtype=int)

        # create sampling mask for interest region sampling strategy
        interest_regions = np.zeros((H_obs, W_obs, ), dtype=np.uint8)
        interest_regions[POI[:,0], POI[:,1]] = 1

        I = 20
        interest_regions = cv2.dilate(interest_regions, np.ones((3, 3), np.uint8), iterations=I)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]

        rand_inds = np.random.choice(interest_regions.shape[0], size=Nsamples, replace=False)
        batch = interest_regions[rand_inds]

        prior = np.ones_like(gt_img)
        prior[batch[:, 0], batch[:, 1]] = gt_img[batch[:, 0], batch[:, 1]]

        return prior
    
    def PSNR(self, img1, img2):
        mse = np.mean((img1 - img2)**2)

        PSNR = -np.log10(mse)

        return PSNR
    
#%%
filename = 'tiger.jpg'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img/255.

# Resize image to be 100 by 100 pixels 
img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)

# Instantiate model class
model = Model(img.shape[0], img.shape[1])

# Convert image to grayscale so there's only 1 channel
img = model.cvt_grayscale(img)

# Perform average blurring
obs_img = model.avg_blur(img)
H_avg_blur = model.avg_blur_H()     # and get the H matrix

#%% Solving inverse problem using pseudoinverse

pred_img = linalg.spsolve(H_avg_blur, obs_img.flatten())
pred_img = pred_img.reshape(img.shape)  # The predicted input

#%% Plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(15, 15))
ax1.imshow(img, cmap='gray')    # Original img
ax2.imshow(obs_img, cmap='gray')    # Blurred image
ax3.imshow(pred_img, cmap='gray')   # Inverse problem from blurred image
ax4.imshow(model.avg_blur(pred_img), cmap='gray')
plt.show()

print('PSNR', model.PSNR(img, pred_img))
# %%
