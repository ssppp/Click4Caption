import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
import base64
from io import BytesIO

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
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
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
print('Initialization Finished')
model_image_size = vis_processor_cfg.image_size

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list, tl_x, tl_y, br_x, br_y, is_top_left):
    if chat_state is not None:
        # chat_state.messages = []
        chat_state = None
    if img_list is not None:
        # img_list = []
        img_list = None
    tl_x, tl_y, br_x, br_y = -1, -1, -1, -1
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False), gr.update(value="Upload for QA", interactive=True), chat_state, img_list, tl_x, tl_y, br_x, br_y, True

def upload_img(gr_img, text_input, chat_state, tl_x, tl_y, br_x, br_y, img_list, chatbot):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
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
    
    # draw bbox on img and show in chatbot
    _img = np.array(gr_img.copy())
    line_width = int((gr_img.size[0]+gr_img.size[1])/224)
    _img[tl_y:tl_y+line_width, tl_x:br_x] = np.array([255, 0, 0])
    _img[br_y-line_width+1:br_y+1, tl_x:br_x] = np.array([255, 0, 0])
    _img[tl_y:br_y, tl_x:tl_x+line_width] = np.array([255, 0, 0])
    _img[tl_y:br_y, br_x-line_width+1:br_x+1] = np.array([255, 0, 0])
    _img = Image.fromarray(_img)

    buffered = BytesIO()
    _img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
    # chatbot = chatbot + [[img_str, llm_message]]
    chatbot = chatbot + [[img_str, None]]
    return gr_img, gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Upload More"), chat_state, img_list, chatbot

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,  # 800,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


def gradio_describe(interaction_mode, is_top_left, gr_img, chatbot, tl_x, tl_y, br_x, br_y, num_beams, temperature):
    if interaction_mode == "QA":
        return chatbot
    if not is_top_left:  # clear chat history
        return None
    _, _, _, chat_state, img_list, chatbot = upload_img(gr_img, None, None, tl_x, tl_y, br_x, br_y, None, chatbot)
    user_message = "image[IMG] Tell me what it is and write a description for it."
    _, chatbot, chat_state = gradio_ask(user_message, chatbot, chat_state)
    chatbot, chat_state, img_list = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    return chatbot


title = """<h1 align="center">Click4Caption</h1>"""
description = """Gradio demo for [Click4Caption](https://github.com/ssppp/Click4Caption). Please refer to README for detailed instructions.
"""

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload for QA", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="Beam search numbers",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            interaction_mode = gr.Radio(
                choices=["Describe", "QA"],
                value="Describe",
                label="Interaction mode",
                interactive=True,
            )

            is_top_left = gr.Radio(
                choices=[True, False],
                value=True,
                label="Is selecting top-left coord (True) or bottom-right (False)",
                interactive=True,
            )

            with gr.Row():
                tl_x = gr.Number(
                    value=-1,  # -1 stands for using the whole figure
                    interactive=True,
                    label="top-left x",
                    min_width=60,
                    precision=0,
                )
                tl_y = gr.Number(
                    value=-1,
                    interactive=True,
                    label="top-left y",
                    min_width=60,
                    precision=0,
                )
                br_x = gr.Number(
                    value=-1,
                    interactive=True,
                    label="bottom-right x",
                    min_width=60,
                    precision=0,
                )
                br_y = gr.Number(
                    value=-1,
                    interactive=True,
                    label="bottom-right y",
                    min_width=60,
                    precision=0,
                )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Chatbot')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

    upload_button.click(upload_img, [image, text_input, chat_state, tl_x, tl_y, br_x, br_y, img_list, chatbot], [image, text_input, upload_button, chat_state, img_list, chatbot])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list, tl_x, tl_y, br_x, br_y, is_top_left], 
                [chatbot, image, text_input, upload_button, chat_state, img_list, tl_x, tl_y, br_x, br_y, is_top_left], queue=False)

    def get_point(img, tl_x, tl_y, br_x, br_y, is_top_left, evt: gr.SelectData):
        # print(f"image shape: {img.size}")
        x, y = evt.index
        if is_top_left:
            tl_x, tl_y = x, y
            is_top_left = False
        else:
            br_x, br_y = x, y
            is_top_left = True
        return tl_x, tl_y, br_x, br_y, is_top_left
    
    image.select(get_point, [image, tl_x, tl_y, br_x, br_y, is_top_left], [tl_x, tl_y, br_x, br_y, is_top_left]).then(
        gradio_describe, [interaction_mode, is_top_left, image, chatbot, tl_x, tl_y, br_x, br_y, num_beams, temperature], [chatbot]
    )

    interaction_mode.change(gradio_reset, [chat_state, img_list, tl_x, tl_y, br_x, br_y, is_top_left], 
                            [chatbot, image, text_input, upload_button, chat_state, img_list, tl_x, tl_y, br_x, br_y, is_top_left], queue=False)

demo.launch(share=True, enable_queue=True)
# demo.launch(server_name='0.0.0.0', server_port=args.port, share=False, enable_queue=True)
