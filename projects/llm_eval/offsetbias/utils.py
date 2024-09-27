import yaml
import json
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import datasets

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대해 시드 고정
    # Deterministic 모드 활성화 (GPU 성능이 약간 저하될 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_yaml(data, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

def load_config(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class CustomDatasetForDev(Dataset):
    def __init__(self, args, tokenizer, prompt_template):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []
        self.fname = args.input_file
        # PROMPT = 'You are EXAONE model from LG AI Research, a helpful assistant.'
        if self.fname == 'NCSOFT/offsetbias':
            data = datasets.load_dataset(self.fname, split='train')
        else:
            data = datasets.load_dataset(self.fname)

        def make_chat(data, ):
            if args.reverse:
                user_message = prompt_template.format(input=data['instruction'], 
                                                output_1=data['output_2'], 
                                                output_2=data['output_1'])
            else:
                user_message = prompt_template.format(input=data['instruction'], 
                                                output_1=data['output_1'], 
                                                output_2=data['output_2'])
            return user_message
        
        for example in data:
            chat = make_chat(example)
            message = [{"role": "user", "content": chat}]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            target = example["label"]

            self.inp.append(source)
            self.label.append(target)
        

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]
