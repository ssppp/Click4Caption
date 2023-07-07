import argparse
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr
from PIL import Image
# import base64
# from io import BytesIO

from click4caption.common.config import Config
from click4caption.common.dist_utils import get_rank
from click4caption.common.registry import registry
from click4caption.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from click4caption.datasets.builders import *
from click4caption.models import *
from click4caption.processors import *
from click4caption.runners import *
from click4caption.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.)

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--tl_x", type=int, default=-1, help="top-left coord x")  # -1 stands for using the whole figure
    parser.add_argument("--tl_y", type=int, default=-1, help="top-left coord y")
    parser.add_argument("--br_x", type=int, default=-1, help="bottom-right coord x")
    parser.add_argument("--br_y", type=int, default=-1, help="bottom-right coord y")

    parser.add_argument("--input_text", type=str, default="image[IMG] Tell me what it is and write a description for it.",
                        help="question about the image, use [IMG] as image embs placeholder")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def upload_img(chat, gr_img, chat_state, tl_x, tl_y, br_x, br_y, img_list, model_image_size):
    if chat_state is None:
        chat_state = CONV_VISION.copy()
    if img_list is None:
        img_list = []
    
    # process the coords
    tl_x = 0 if tl_x < 0 else tl_x
    tl_y = 0 if tl_y < 0 else tl_y
    br_x = gr_img.size[0] if br_x < 0 else br_x
    br_y = gr_img.size[1] if br_y < 0 else br_y
    x_scale = model_image_size / gr_img.size[0]
    y_scale = model_image_size / gr_img.size[1]
    
    # upload
    llm_message = chat.upload_img(gr_img, chat_state, img_list, tl_x*x_scale, tl_y*y_scale, br_x*x_scale, br_y*y_scale)

    # draw bbox on img
    _img = np.array(gr_img.copy())
    line_width = int((gr_img.size[0]+gr_img.size[1])/224)
    _img[tl_y:tl_y+line_width, tl_x:br_x] = np.array([255, 0, 0])
    _img[br_y-line_width+1:br_y+1, tl_x:br_x] = np.array([255, 0, 0])
    _img[tl_y:br_y, tl_x:tl_x+line_width] = np.array([255, 0, 0])
    _img[tl_y:br_y, br_x-line_width+1:br_x+1] = np.array([255, 0, 0])
    _img = Image.fromarray(_img)

    return chat_state, img_list, _img


if __name__ == '__main__':
    # init
    print('Initializing')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model = model.eval()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    model_image_size = vis_processor_cfg.image_size
    print('Initialization Finished')

    # upload image
    image = Image.open(args.image_path).convert("RGB")
    chat_state, img_list, img_with_bbox = upload_img(chat, image, None, args.tl_x, args.tl_y, args.br_x, args.br_y, None, model_image_size)
    image_save_path = osp.join(osp.dirname(osp.realpath(__file__)), "img_with_bbox.jpg")
    print(f"saving the img with drawn bbox in {image_save_path}")
    img_with_bbox.save(image_save_path)

    # ask and answer
    user_message = args.input_text
    if "image[IMG]" not in user_message:
        print(f"Warning: we recommend to use format 'image[IMG] question' as input text")
    chat.ask(user_message, chat_state)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=args.num_beams,
                              temperature=args.temperature,
                              max_new_tokens=300,  # 800,
                              max_length=2000)[0]
    print(f"=====LLM reply=====\n{llm_message}")
