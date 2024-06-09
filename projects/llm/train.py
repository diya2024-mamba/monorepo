"""Modified from https://github.com/redotvideo/mamba-chat/blob/main/train_mamba.py"""

import argparse
import os
from datetime import datetime

import mlflow
import torch
from dotenv import load_dotenv
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from trainer.data import ChatDataModule
from trainer.mamba_trainer import MambaTrainer
from transformers import AutoTokenizer, TrainingArguments

load_dotenv()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Training Example")
mlflow.set_tag("mlflow.runName", datetime.now().strftime("%Y%m%d%H%M%S"))
mlflow.autolog()


def run(args):
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.float32, device=device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

    data_module = ChatDataModule(
        tokenizer=tokenizer, data_path=args.data_path, conversation_template=tokenizer.chat_template, max_tokens=2048
    )

    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir="output",
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_module.data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="./data/ultrachat_small.jsonl")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)