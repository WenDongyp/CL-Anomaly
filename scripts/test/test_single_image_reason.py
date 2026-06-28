import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from PIL import Image
import requests
import copy
import torch

import json
import time
import os

import sys
import warnings

warnings.filterwarnings("ignore")

import argparse
from openpyxl import load_workbook, Workbook

def eval_model(args):

    model_path = args.model_checkpoint
    model_base = args.model_base
    #model_name = get_model_name_from_path(model_path)
    model_name = "llava_qwen_lora"
    device = "cuda"
    #device_map = "auto"
    device_map = "cuda"
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 152064}
    #overwrite_config = {'vocab_size': 152064}  #7b 151936
    print("model_path", model_path)
    print("model_base", model_base)
    print("model_name", model_name)
    print("data bench", args.bench_json)

    result_dir = '/data1/dongwen_data/CL-Anomaly/results'
    #os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"cl-anomaly-reason_metrics_{args.bench_json}")
    print("result_path", result_path)

    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, device_map=device_map, cache_dir='./cache', torch_dtype="bfloat16", overwrite_config=overwrite_config)

    if args.size != '7b':
        model.lm_head.weight = model.model.embed_tokens.weight
        print("Testing 0.5B model, set lm_head weight to embed_tokens weight")
    else:
        print("Testing 7B model, no need to set lm_head weight")

    model.eval()

    responses = []

    def create_conv(history_questions, history_responses, question):
        conv = copy.deepcopy(conv_templates[conv_template])
        for q, r in zip(history_questions, history_responses):
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], r)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv

    with open(os.path.join(args.data_dir, args.bench_json), 'r') as f:
        data = json.load(f)

    results = []
    for index, d in enumerate(data):
        question_set = []
        answer_set = []
        for turn in d['conversations']:
            text = turn['value'].replace('<image>\n', '')
            if turn['from'] == 'human':
                question_set.append(text)
            elif turn['from'] == 'gpt':
                answer_set.append(text)

        history_questions = []
        history_responses = []
        temp_answers = []

        anomaly = 1 if d['metadata']['anomaly'] else 0
        image_path = os.path.join(args.data_dir, d['image'])
        image = Image.open(image_path).convert("RGB")
        # if the longest side of the image is greater than 1024, resize it to 1024 while keeping the aspect ratio
        if max(image.size) > 1024:
            if image.width > image.height:
                new_width = 1024
                new_height = int(1024 * image.height / image.width)
            else:
                new_height = 1024
                new_width = int(1024 * image.width / image.height)
            image = image.resize((new_width, new_height))

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

        question = DEFAULT_IMAGE_TOKEN + "\n" + question_set[0]
        history_questions.append(question)

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        temp_answers.append(text_outputs[0])
        history_responses.append(text_outputs[0])


        for i in range(1, len(question_set)):
            question = question_set[i]
            conv = create_conv(history_questions, history_responses, question)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size]

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            #print(question)
            #print(text_outputs[0])
            temp_answers.append(text_outputs[0])
            history_responses.append(text_outputs[0])
            history_questions.append(question)

        record_out = {
            "id": d["id"],
            "image": d["image"],
            "questions": question_set,
            "answers": answer_set,
            "response": temp_answers
        }
        #print(record_out)

        out_file = result_path
        with open(out_file, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record_out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test detection performance of the model")
    parser.add_argument("--data_dir", type=str, default='your_data_dir', help="Path to your data directory")
    parser.add_argument("--bench_json", type=str, default='VisA/test_data.json', help="Path to your benchmark json file")
    parser.add_argument("--model_checkpoint", type=str, default='your_model_path', help="Path to your pretrained model")
    parser.add_argument("--model_base", type=str, default='your_base_model_path', help="Path to your pretrained model")
    parser.add_argument("--size", type=str, default='7b', help="Model size")
    parser.add_argument("--save_path", type=str, default='your_save_path', help="Path to save the results")
    args = parser.parse_args()

    eval_model(args)


