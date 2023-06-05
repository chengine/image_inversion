#%% 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def logit(y):

    return torch.log(y / (1. - y))

class Model():
    def __init__(self, H, W) -> None:
        self.H = H
        self.W = W

    def cvt_grayscale(self, img):
        # Perform gray scaling
        img = 0.2989*img[..., 0] + 0.5870*img[..., 1] + 0.1140*img[..., 2]

        return img

    def blur(self, img):

        # Perform Gaussian blur on img
        img = torch.nn.functional.avg_pool2d(img,5,stride=1, padding=2, count_include_pad=False)

        return img

    def prior(self, gt_img, Nsamples=10000):
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

        prior = np.random.uniform(size=gt_img.shape)
        mask = np.ones(gt_img.shape[:-1], dtype=bool)
        prior[batch[:, 0], batch[:, 1]] = gt_img[batch[:, 0], batch[:, 1]]
        mask[batch[:, 0], batch[:, 1]] = False

        return prior, mask
    
    def PSNR(self, img1, img2):
        mse = torch.mean((img1 - img2)**2)

        PSNR = -torch.log10(mse)

        return PSNR
    
#%%
filename = 'tiger.jpg'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize image to be 100 by 100 pixels 
img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)

# Instantiate model class
model = Model(img.shape[0], img.shape[1])
prior, mask = model.prior(img, Nsamples=1)

img = img/255.

img = torch.tensor(img, device=device)
prior, mask = torch.tensor(prior, device=device), torch.tensor(mask, device=device)

# Convert image to grayscale so there's only 1 channel
gray_img = model.cvt_grayscale(img)

# Perform average blurring
obs_img = model.blur(gray_img[None, None, ...])
obs_img = (obs_img**2).squeeze()

# obs_img_offset = obs_img - model.blur(model.cvt_grayscale(prior)[None, None,...]).squeeze()
#%% Generate Q matrix
theta1 = 1
r = 1e-3

# Y, X = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
# xv = X.flatten()
# yv = Y.flatten()
# d = theta1*np.exp(-(0.5)*np.sqrt((xv[0]-xv)**2 + (yv[0]-yv)**2))
# d[0] = 0

# Q_element = scipy.linalg.toeplitz(d)
# Q_element = Q_element.T
# Q_element = Q_element.T @ Q_element
# Q_inv = torch.tensor(Q_element, device=device, dtype=torch.float32)

r_inv = 1/r

#%% Optimize
s = torch.randn(img.shape, device=device, requires_grad=True)
optimizer = optim.Adam([s], lr=0.01)

N = 5000

for i in range(N):
    optimizer.zero_grad()

    s_ = s*mask[..., None] + prior
    gray_img_pred = model.cvt_grayscale(s_)
    pred_obs = model.blur(gray_img_pred[None, None, ...])

    pred_loss = r_inv * torch.mean((obs_img - pred_obs**2)**2)
    reg_loss = torch.mean((model.blur(s_) - s_)**2)

    loss = pred_loss + reg_loss

    loss.backward()
    optimizer.step()
    print(i, loss)
#%% Plotting
pred_img = torch.clip(s_, 0., 1.)
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))
ax1.imshow(gray_img.cpu(), cmap='gray')    # Original img
ax2.imshow(obs_img.cpu().reshape(img.shape[:-1]), cmap='gray')    # Blurred image
ax3.imshow(pred_img.detach().cpu())   # Inverse problem from blurred image
plt.show()

print('PSNR', model.PSNR(img, pred_img))
print('PSNR', model.PSNR(img, prior))
# %% Save images
cv2.imwrite('og_grayscale.jpg', (255*gray_img.cpu().numpy()).astype(np.uint8))
cv2.imwrite('obs_img.jpg', (255*obs_img.cpu().numpy()).astype(np.uint8))
cv2.imwrite('pred_img.jpg', cv2.cvtColor((255*pred_img.detach().cpu().numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite('prior.jpg', cv2.cvtColor((255*prior.cpu().numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR))
# %%
