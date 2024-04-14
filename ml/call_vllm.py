import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from urllib.request import urlopen

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_hub_download(
    repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir="./"
)
from models import CLIPVisionTower

DEVICE = "cuda"
PROMPT = "This is a dialog with AI assistant.\n"

tokenizer = AutoTokenizer.from_pretrained(
    "AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tokenizer", use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(
    "AIRI-Institute/OmniFusion",
    subfolder="OmniMistral-v1_1/tuned-model",
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)

hf_hub_download(
    repo_id="AIRI-Institute/OmniFusion",
    filename="OmniMistral-v1_1/projection.pt",
    local_dir="./",
)
hf_hub_download(
    repo_id="AIRI-Institute/OmniFusion",
    filename="OmniMistral-v1_1/special_embeddings.pt",
    local_dir="./",
)
projection = torch.load("OmniMistral-v1_1/projection.pt", map_location=DEVICE)
special_embs = torch.load("OmniMistral-v1_1/special_embeddings.pt", map_location=DEVICE)

clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
clip.load_model()
clip = clip.to(device=DEVICE, dtype=torch.bfloat16)


def gen_answer(model, tokenizer, clip, projection, query, special_embs, image=None):
    bad_words_ids = tokenizer(
        ["\n", "</s>", ":"], add_special_tokens=False
    ).input_ids + [[13]]
    gen_params = {
        "do_sample": False,
        "max_new_tokens": 60,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "forced_eos_token_id": 2,
        "use_cache": False,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
    }
    with torch.no_grad():
        image_features = clip.image_processor(image, return_tensors="pt")
        image_embedding = clip(image_features["pixel_values"]).to(
            device=DEVICE, dtype=torch.bfloat16
        )

        projected_vision_embeddings = projection(image_embedding).to(
            device=DEVICE, dtype=torch.bfloat16
        )
        prompt_ids = tokenizer.encode(
            f"{PROMPT}", add_special_tokens=False, return_tensors="pt"
        ).to(device=DEVICE)
        question_ids = tokenizer.encode(
            query, add_special_tokens=False, return_tensors="pt"
        ).to(device=DEVICE)

        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)

        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs["SOI"][None, None, ...],
                projected_vision_embeddings,
                special_embs["EOI"][None, None, ...],
                special_embs["USER"][None, None, ...],
                question_embeddings,
                special_embs["BOT"][None, None, ...],
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
        out = model.generate(inputs_embeds=embeddings, **gen_params)
    # out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts


def make_few_shot(descriptions, max_length: int = 300):
    s = ""
    for desc in descriptions:
        s += "- " + desc[:max_length] + "..." * bool(len(desc) > max_length) + "\n"
    return s[:-1]


PROMPT_TEMPLATE = """
Используйте прилагаемое изображение музейного экспоната и следующие несколько описаний схожих предметов, чтобы создать новое подробное описание для этого экспоната, сосредоточив внимание на конкретных деталях, внешнем виде и атрибутах. Описание должно быть фактическим, ориентированным на наблюдаемые элементы изображения и ключевые аспекты из предоставленных описаний. Укажите материалы, форму, цвет и украшения, избегая абстрактных интерпретаций и философских размышлений.
Примеры описаний:
{}
Описание должно предмета на изображении быть фактическим, ориентированным на наблюдаемые элементы. не нужно описывать для чего используется предмет - необходимо описать его внешние характеристики, такие как материалы и форма:
"""


def call_vision_api(descriptions, image):
    few_shot = make_few_shot(descriptions)
    question = PROMPT_TEMPLATE.format(few_shot)
    return gen_answer(
        model,
        tokenizer,
        clip,
        projection,
        query=question,
        special_embs=special_embs,
        image=image,
    )
