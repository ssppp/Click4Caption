import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from click4caption.common.registry import registry
from click4caption.models.blip2 import Blip2Base, disabled_train
# from click4caption.models.modeling_llama import LlamaForCausalLM
# from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

from click4caption.models.sam_prompt_encoder import PositionEmbeddingRandom


@registry.register_model("click4caption")
class Click4Caption(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/click4caption.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        llama_proj_ckpt=None,
        delete_vit_cls_token=False,
        add_img_pe=False,
        use_vit_multiblock_feat=None,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print(f"==> model image_size={img_size}")
        self.img_size = img_size
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        # use the pad_token following original Vicuna
        DEFAULT_PAD_TOKEN = "[PAD]"
        self.llama_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        # .from_pretrained doc: The model is set in evaluation mode by default
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_ckpt:
            print(f"Loading llama_proj from {llama_proj_ckpt}...")
            msg = self.load_state_dict(torch.load(llama_proj_ckpt, map_location="cpu")['model'], strict=False)
            print(f"unexpected_keys: {msg.unexpected_keys}")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        
        ### for bbox input
        _embed_dim = self.query_tokens.shape[-1]
        self.coordinate_pe_layer = PositionEmbeddingRandom(_embed_dim // 2)
        self.coordinate_type_emb = nn.ModuleList([nn.Embedding(1, _embed_dim) for i in range(2)])  # top left & bottom right two types

        self.add_img_pe = add_img_pe
        if self.add_img_pe:
            print("==> Add pe in the image embs")
            self.img_embs_pe_proj = nn.Linear(_embed_dim, self.visual_encoder.num_features)
        self.delete_vit_cls_token = delete_vit_cls_token
        if delete_vit_cls_token:
            print("==> Delete the vit CLS token when feeding in Qformer")
        self.use_vit_multiblock_feat=use_vit_multiblock_feat
        if use_vit_multiblock_feat:
            self.use_vit_multiblock_feat = list(self.use_vit_multiblock_feat)
            print(f"==> Use vit multi-block feats: block {self.use_vit_multiblock_feat}")
        ###

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
    
    def encode_img_with_bbox(self, image, boxes, image_embeds=None):  # image_embeds is used when selecting diff boxes in the same image in eval
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            bsz, num_regions = boxes.shape[:2]
            if image_embeds is None:
                # get multi-block feats
                if self.use_vit_multiblock_feat:
                    image_embeds = torch.stack(self.visual_encoder.get_intermediate_layers(image, block_id=self.use_vit_multiblock_feat)).mean(dim=0)
                    image_embeds = self.ln_vision(image_embeds).to(device)
                else:
                    image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)

                # delete the first token [CLS]
                if self.delete_vit_cls_token:
                    image_embeds = image_embeds[:, 1:]

                # add pe
                if self.add_img_pe:
                    img_embs_pe = self.coordinate_pe_layer((self.img_size//14, self.img_size//14)).flatten(1).transpose(0, 1)  # (16*16) * C
                    if not self.delete_vit_cls_token:
                        img_embs_pe = torch.cat([img_embs_pe.mean(0, keepdim=True), img_embs_pe], dim=0)  # add one for CLS emb
                    img_embs_pe = img_embs_pe.unsqueeze(0)
                    image_embeds = image_embeds + self.img_embs_pe_proj(img_embs_pe).expand(image_embeds.shape[0], -1, -1)
            else:
                print("=> use the cached vit-embs of last image!")
            # extend
            cross_image_embeds = image_embeds.repeat_interleave(num_regions, dim=0)
            cross_image_atts = torch.ones(cross_image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(cross_image_embeds.shape[0], -1, -1)
            ### modified from sam_prompt_encoder
            # boxes = boxes + 0.5  # Shift to center of pixel
            boxes = boxes.clip(min=0, max=self.img_size)
            coords = boxes.reshape(-1, 2, 2)
            corner_embedding = self.coordinate_pe_layer.forward_with_coords(coords, (self.img_size, self.img_size))
            corner_embedding[:, 0, :] += self.coordinate_type_emb[0].weight
            corner_embedding[:, 1, :] += self.coordinate_type_emb[1].weight
            query_tokens = torch.cat([query_tokens, corner_embedding], dim=1)
            ###
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=cross_image_embeds,
                encoder_attention_mask=cross_image_atts,
                return_dict=True,
            )

            # inputs_llama = self.llama_proj(query_output.last_hidden_state)
            inputs_llama = self.llama_proj(query_output.last_hidden_state[:, :-corner_embedding.shape[1]])  # remove the coordinate embs
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama, image_embeds

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        llama_proj_ckpt = cfg.get("llama_proj_ckpt", None)
        delete_vit_cls_token = cfg.get("delete_vit_cls_token", False)
        add_img_pe = cfg.get("add_img_pe", False)
        use_vit_multiblock_feat = cfg.get("use_vit_multiblock_feat", None)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            llama_proj_ckpt=llama_proj_ckpt,
            delete_vit_cls_token=delete_vit_cls_token,
            add_img_pe=add_img_pe,
            use_vit_multiblock_feat=use_vit_multiblock_feat,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load Model Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print(f"Unexpected_keys: {msg.unexpected_keys}")

        return model
