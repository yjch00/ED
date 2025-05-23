import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import torch.nn.functional as F


from transformers import set_seed
from ed_utils.sampling_fasted import evolve_ed_sampling
evolve_ed_sampling()
import random
import numpy as np



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def eval_model(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    batch_size = args.batch_size
    num_batches = len(questions) // batch_size + (1 if len(questions) % batch_size != 0 else 0)
    model.eval()
    
    

    for batch_idx in tqdm(range(num_batches)):
        batch_questions = questions[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        input_ids_batch = []
        image_tensors_batch = []
        image_tensors_ed_batch = [[] for _ in range(4)]  


        for line in batch_questions:
            idx = line["question_id"]
            image_file = line["image"]
            label = line["label"]
            qs = line["text"]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            if args.use_description:
                conv.append_message(conv.roles[0], qs)  # description, MME
            else:
                conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")  # POPE

            
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').squeeze(0).cuda()
            input_ids_batch.append(input_ids)
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensors_batch.append(image_tensor)
        

            if args.use_ed:
                crop_size = 336
                width, height = image.size
            
                if width > 672 or height > 672:
                    image = image.resize((448, 448))
                width, height = image.size
                
                left_top = (0, 0, crop_size, crop_size)
                right_top = (width - crop_size, 0, width, crop_size)
                left_bottom = (0, height - crop_size, crop_size, height)
                right_bottom = (width - crop_size, height - crop_size, width, height)
                
                image1 = image.crop(left_top)   
                image2 = image.crop(right_top)   
                image3 = image.crop(left_bottom)   
                image4 = image.crop(right_bottom)   
                image_tensors_ed_batch[0].append(image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0])
                image_tensors_ed_batch[1].append(image_processor.preprocess(image2, return_tensors='pt')['pixel_values'][0])
                image_tensors_ed_batch[2].append(image_processor.preprocess(image3, return_tensors='pt')['pixel_values'][0])
                image_tensors_ed_batch[3].append(image_processor.preprocess(image4, return_tensors='pt')['pixel_values'][0])
            else:
                for i in range(4):
                    image_tensors_ed_batch[i].append(None)


        max_len = max(len(ids) for ids in input_ids_batch)
        input_ids_batch = [F.pad(ids, (max_len - len(ids), 0), value=tokenizer.pad_token_id) for ids in input_ids_batch]
        input_ids_batch = torch.stack(input_ids_batch)

        image_tensors_batch = torch.stack(image_tensors_batch).half().cuda()
        image_tensors_ed_batch = [torch.stack(tensor_list).half().cuda() if tensor_list[0] is not None else None for tensor_list in image_tensors_ed_batch]


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        if not args.use_ed:
            image_tensors_ed_batch[0] = None
            image_tensors_ed_batch[1] = None
            image_tensors_ed_batch[2] = None
            image_tensors_ed_batch[3] = None
        attention_mask = (input_ids_batch != tokenizer.pad_token_id).long().cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_batch,
                attention_mask = attention_mask,
                images=image_tensors_batch,
                images_ed1=image_tensors_ed_batch[0],
                images_ed2=image_tensors_ed_batch[1],
                images_ed3=image_tensors_ed_batch[2],
                images_ed4=image_tensors_ed_batch[3],
                
                ed_alpha = args.ed_alpha,
                ed_beta = args.ed_beta,
                
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=512,
                use_cache=True)

        input_token_len = input_ids_batch.shape[1]
        for i, output in enumerate(output_ids):
            n_diff_input_output = (input_ids_batch[i] != output[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            outputs = tokenizer.decode(output[input_token_len:], skip_special_tokens=True)
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            
            line = batch_questions[i]
            idx = line["question_id"]
            image_file = line["image"]
            label = line["label"]
            qs = line["text"]

            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": qs,
                                       "text": outputs,
                                       "model_id": model_name,
                                       "image": image_file,
                                       "gts": label}) + "\n")
            ans_file.flush()

    ans_file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--use_description", action='store_true', default=False)
    parser.add_argument("--use_ed", action='store_true', default=False)
    
    parser.add_argument("--ed_alpha", type=float, default=0.5)
    parser.add_argument("--ed_beta", type=float, default=0.5)
    
    parser.add_argument("--seed", type=int, default=67) 
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)

    