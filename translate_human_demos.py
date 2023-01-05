"""Translates human demo image observations in a replay buffer.

This expects a checkpoint file in `./checkpoints/<checkpoint_name>/` with the name `latest_net_G.pth`.
You should copy the G_A checkpoint that you want to load (e.g. `23_net_G_A.pth`) and name that copy `latest_net_G.pth`.
Then in the current directory place a replay buffer .pkl file with the name `human_demos.pkl`. The script will translate this and output `translated_demos.pkl`.

Usage example (the --dataroot arg doesn't matter here, but I'm putting it there anyway because it's a required "option"):
    python translate_human_demos.py --dataroot hchd/images --name pen_grasp_6-5-22-on_iris-hi-batchsize16 --model test --no_dropout
"""
import numpy as np
import os
import pickle as pkl
import torch
from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import util
from tqdm import tqdm

def translate_single_image(model, image):
    # We need to only translate the top half of the image and leave the bottom half as is.
    # Here, the image shape starts off as (3, H, W).
    image = np.asarray([image]) # e.g. (3, 50, 100) -> (1, 3, 50, 100)
    data = {'A': None, 'A_paths': None} # if we don't have the 'A_paths' key, a runtime error is thrown later
    data['A'] = torch.FloatTensor(image)
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    result_image = model.get_current_visuals()['fake']
    result_image = util.tensor2im(result_image)
    result_image = np.transpose(result_image, (2, 0, 1)) # model outputs (50, 100, 3), so shape into (3, 50, 100)
    return result_image


def get_output_pkl_filename(demo_filename):
    """Returns a filename for the output pickle file for the updated replay buffer.
    Example: red-cube-grasping-robot-demos.pkl -> red-cube-grasping-robot-demos_cropped-out-fingers.pkl"""
    split_list = demo_filename.split('.pkl') # Example: 'red-cube-grasping-robot-demos.pkl' -> ['red-cube-grasping-robot-demos', '']
    assert len(split_list) == 2
    return split_list[0] + '_cyclegan-translated.pkl'


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    demo_file_name = opt.data_path
    if not os.path.isfile(demo_file_name):
        print("Bad file given as arg. Exiting.")
        exit(0)
    replay_buffer = pkl.load(open(demo_file_name, "rb"))
    num_steps = len(replay_buffer)
    print('Number of steps in replay buffer:', num_steps)

    for i in tqdm(range(num_steps)):
        image = replay_buffer.hand_img_obses[i]
        result_image = translate_single_image(model, image)
        replay_buffer.hand_img_obses[i] = result_image
        image = replay_buffer.next_hand_img_obses[i]
        result_image = translate_single_image(model, image)
        replay_buffer.next_hand_img_obses[i] = result_image

    output_file_name = get_output_pkl_filename(demo_file_name)
    pkl.dump(replay_buffer, open(output_file_name, "wb"), protocol=pkl.HIGHEST_PROTOCOL)
