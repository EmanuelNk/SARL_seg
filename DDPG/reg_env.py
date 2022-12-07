import numpy as np
import matplotlib.pyplot as plt
# from IPython.core.display_functions import display
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Resize, ToTensor
# import cv2 as cv
from collections import deque
import json
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import os
# from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce
# from torchsummary import summary
# from torch.distributions import Categorical

def generate_cropped_im(im, cropped_dims):
    h_shift = 10
    s_shift = 10
    v_shift = 51
    # generate random crop shifts
    crop_attributes = {}
    cropped_shift = [np.random.randint(0,im.size[0]-cropped_dims[0]),np.random.randint(0,im.size[1]-cropped_dims[1])]
    # cropped_shift = [37,37]
    crop_coords = (cropped_shift[0], cropped_shift[1], cropped_shift[0]+cropped_dims[0], cropped_shift[1]+cropped_dims[1])
    cropped_im = im.crop(crop_coords)
    cropped_im = cropped_im.convert('HSV')
    h, s, v = cropped_im.split()
    h = h.point(lambda p: p + h_shift)
    s = s.point(lambda p: p + s_shift)
    v = v.point(lambda p: p + v_shift)
    cropped_im = Image.merge('HSV', (h, s, v))
    cropped_im = cropped_im.convert('RGB')
    cropped_im_shape = cropped_im.size
    crop_attributes['crop_coords'] = crop_coords
    crop_attributes['cropped_shift'] = cropped_shift
    crop_attributes['cropped_im'] = cropped_im
    crop_attributes['cropped_im_shape'] = cropped_im_shape
    
    return crop_attributes

class Env:
    def __init__(self, fixed_image, moving_image, crop_coords=(37, 37, 74, 74)):
        self.crop_coords = crop_coords
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.fixed_x = fixed_image.size[0]
        self.fixed_y = fixed_image.size[1]
        self.crop_x = crop_coords[0]
        self.crop_y = crop_coords[1]
        self.shift_x = 0
        self.shift_y = 0
        self.x_limit = fixed_image.size[0] - moving_image.size[0]
        self.y_limit = fixed_image.size[1] - moving_image.size[1]
        self.env_image = fixed_image.copy()
        self.env_image.paste(moving_image, (0,0))
        self.action_space = 4

#     def reset(self):
#         # move image to a random position
#         print(self.x_limit)
#         self.shift_x = np.random.randint(0, self.x_limit)
#         self.shift_y = np.random.randint(0, self.y_limit)
#         self.env_image = self.fixed_image.copy()
#         self.env_image.paste(self.moving_image, (self.shift_x, self.shift_y))
#         return self.env_image
    
    def reset(self):
        crop_attr = generate_cropped_im(self.fixed_image, [37,37])
        self.crop_coords = crop_attr['crop_coords']
        print(f"Crop coords: {self.crop_coords}")
        self.moving_image = crop_attr['cropped_im']
        self.crop_x = self.crop_coords[0]
        self.crop_y = self.crop_coords[1]
        self.x_limit = self.fixed_image.size[0] - self.moving_image.size[0]
        self.y_limit = self.fixed_image.size[1] - self.moving_image.size[1]
        # move image to a random position
        print(self.x_limit)
        self.shift_x = np.random.randint(0, self.x_limit)
        self.shift_y = np.random.randint(0, self.y_limit)
        self.env_image = self.fixed_image.copy()
        self.env_image.paste(self.moving_image, (self.shift_x, self.shift_y))
        
        return self.env_image

    def check_frame(self, _im):
        _im_shape = _im.size
        _im_shape = [_im_shape[0] - 1, _im_shape[1] - 1]
        # check bottom border
        for i in range(_im_shape[0]):
            px = _im.getpixel((i, 0))
            if not (px[0] == 0 and px[1] == 0 and px[2] == 0):
                return False
        # check top border
        for i in range(_im_shape[0]):
            px = _im.getpixel((i, _im_shape[1]))
            if not (px[0] == 0 and px[1] == 0 and px[2] == 0):
                return False
        # check left border
        for i in range(_im_shape[1]):
            px = _im.getpixel((0, i))
            if not (px[0] == 0 and px[1] == 0 and px[2] == 0):
                return False
        # check right border
        for i in range(_im_shape[1]):
            px = _im.getpixel((_im_shape[0], i))
            if not (px[0] == 0 and px[1] == 0 and px[2] == 0):
                return False
        return True

    def move_image(self, _x, _y):
        EDGE_PENALTY = 0
        shift_x = int(self.shift_x + _x)
        shift_y = int(self.shift_y + _y)
        # shift_x = _x
        # shift_y = _y
        # check if image can be moved
        reward_bonus = 0
        if shift_x > self.x_limit:
            shift_x = self.x_limit
            reward_bonus -= EDGE_PENALTY
            # print('x limit reached')
        elif shift_x < 0:
            shift_x = 0
            reward_bonus -= EDGE_PENALTY
            # print('x limit reached')
        if shift_y > self.y_limit:
            shift_y = self.y_limit
            reward_bonus -= EDGE_PENALTY
            # print('y limit reached')
        elif shift_y < 0:
            shift_y = 0
            reward_bonus -= EDGE_PENALTY
            # print('y limit reached')
        if shift_x < 0 or shift_y < 0:
            print(f"Image can't be moved to {shift_x}, {shift_y}")
            return self.env_image, self.get_reward()
        self.shift_x = shift_x
        self.shift_y = shift_y
        env_copy = self.fixed_image.copy()
        env_copy.paste(self.moving_image, (shift_x, shift_y))
        self.env_image = env_copy
        reward = self.get_reward()
        # print(f"shift x {shift_x}, shift y {shift_y}, moved x {_x}, moved y {_y} moved x {self.crop_x}, moved y {self.crop_y}, reward {reward}")
        return env_copy, reward, reward_bonus

    def get_reward(self):
        distance = np.sqrt((self.shift_x - self.crop_x)**2 + (self.shift_y - self.crop_y)**2)
        return -distance

    def get_target(self, shift_x, shift_y):
        _x = self.crop_x - shift_x
        _y = self.crop_y - shift_y
        target_x = 0.5+(_x/(2*self.x_limit))
        target_y = 0.5+(_y/(2*self.y_limit))
        return target_x, target_y

    def get_pred_target(self, pred):
        shift_x = (pred[0]*2-1) * self.x_limit
        shift_y = (pred[1]*2-1) * self.y_limit
        _x = self.crop_x - shift_x
        _y = self.crop_y - shift_y
        pred_target_x = 0.5+(_x/(2*self.x_limit))
        pred_target_y = 0.5+(_y/(2*self.y_limit))
        return pred_target_x, pred_target_y
        # return _x, _y

    def move_image_old(self, x, y):
        # move image using pil
        new_im = self.moving_image.transform(self.moving_image.size, Image.AFFINE, (1, 0, x, 0, 1, y))
        can_move = self.check_frame(new_im)
        if can_move:
            self.moving_image = new_im
            return new_im
        else:
            return self.moving_image

    def step(self, action, amount=1):
        """
        action: 0 = left, 1 = right, 2 = up, 3 = down
        """
        
        x_amount = action[0][0]*self.fixed_image.size[0]
        y_amount = action[0][1]*self.fixed_image.size[1]
        new_im, reward, bonus = self.move_image(x_amount, y_amount)

        if reward >= -1:
            reward = (self.fixed_image.size[0] + 100)
            return new_im, reward+bonus, True
        else:
            return new_im, (reward+bonus), False

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', autosize = False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        autosize    - Optional  : automatically resize the length of the progress bar to the terminal window (Bool)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % (prefix, fill, percent, suffix)
    if autosize:
        cols, _ = shutil.get_terminal_size(fallback = (length, 1))
        length = cols - len(styling)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s' % styling.replace(fill, bar), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def preprocess_data(path, display=False):
    h_shift = 0
    s_shift = 0
    v_shift = 0
    envs = []
    envs_info = {}
    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith(".jpg") or filename.endswith(".JPEG"):
            im = Image.open('/'.join([path, filename]))
            im_shape = im.size
            # check if image has 3 channels
            if len(im.getbands()) < 3:
                # print(f"Image {filename} has less than 3 channels")
                continue
            # add progress bar
            printProgressBar(i, len(os.listdir(path)), prefix='Prepearing envs:', suffix='Complete', length=100)
            if im_shape[0] >= 112 and im_shape[1] >= 112:
                
                im_scale_x = im_shape[0]/112
                im_scale_y = im_shape[1]/112

                scale = 1/2
                im = im.resize((int(im_shape[0]/im_scale_x),int(im_shape[1]/im_scale_y)),Image.ANTIALIAS)
                im_shape = im.size
                im = im.crop((0, 0, 112, 112))
                im_shape = im.size
                cropped_dims = [37, 37]
                # add noise to cropped dims
                crop_noise = np.random.randint(0, 15)
                cropped_dims[0] += crop_noise
                cropped_dims[1] += crop_noise
                crop_attr = generate_cropped_im(im, cropped_dims)
                cropped_im = crop_attr['cropped_im']
                # change cropped image hue
                cropped_im = cropped_im.convert('HSV')
                cropped_im_shape = crop_attr['cropped_im_shape']
                # print(f'im_shape: {im_shape}, cropped_im_shape: {cropped_im_shape}')
                crop_coords = crop_attr['crop_coords']

                # red_dot = Image.new('RGB', (10, 10), color = 'red')
                # green_dot = Image.new('RGB', (10, 10), color = 'green')
                env_im = im
                # cropped_im.paste(red_dot, (0,0))
                fixed_x = crop_coords[0]
                fixed_y = crop_coords[1]
                shift_x = im_shape[0] - cropped_im_shape[0]
                shift_y = 0

                fixed = (fixed_x, fixed_y)
                moving = (shift_x, shift_y)

                distance = np.sqrt((fixed[0] - moving[0])**2 + (fixed[1] - moving[1])**2)

                # env_im.paste(cropped_im, (shift_x, shift_y))
                # env_im.paste(green_dot, (fixed_x, fixed_y))
                env = Env(im, cropped_im, crop_coords)
                envs.append(env)
                # print(f'image: {path},  distance: {distance}')
                envs_info[filename] = {'path': path, 'distance': distance, 'crop_dims': cropped_dims, 'crop_coords': crop_coords}
                im2 = im.copy()
                if display:
                    plt.imshow(im2)
                    plt.show()
                    plt.imshow(cropped_im)
                    plt.show()
                # resize to imagenet size
                transform = Compose([Resize((112, 112)), ToTensor()])
                x = transform(im)
                x = x.unsqueeze(0) # add batch dim
                x.shape
                # save envs info to file
                with open('tmp/envs_info.json', 'w') as f:
                    json.dump(envs_info, f)


    return envs