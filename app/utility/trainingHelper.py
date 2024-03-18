from datasets import Dataset
import pandas as pd
import os
import torch
import random
import numpy as np
from transformers import LlamaTokenizerFast, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig
from trl import SFTTrainer


global_var = None
tokenizer = None
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQUENCE_LENGTH = 2048
FP16 = False
GRADIENT_CLIPPING = 1
script_dir = os.path.dirname(__file__)
rel_path = "\\files\\"
FILE_PATH_LOCATION = script_dir + "\\..\\"+ rel_path


def start_training_model():

    global tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"

    dataset = get_dataset()

    model = LlamaForCausalLM.from_pretrained(BASE_MODEL)

    training_params = TrainingArguments(
        output_dir="./deleteme",
        fp16=FP16,
        use_cpu=True,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        seed=3407
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        max_seq_length=MAX_SEQUENCE_LENGTH

    )

    trainer.train()
    global global_var
    global_var = model

    return global_var


def get_response(message):
    global global_var
    global tokenizer

    pipe = pipeline(task="text-generation",
                model=global_var,
                tokenizer=tokenizer,
                max_new_tokens=50,
                num_beams=10,
                early_stopping=True,
                no_repeat_ngram_size=2)

    prompt = f"<|user|>"+message+"<|assistant|>"
    result = pipe(prompt)

    return result

def get_dataset():
    data = Dataset.from_pandas(pd.read_csv(FILE_PATH_LOCATION + "dataset.csv"))
    data_df = pd.DataFrame(data)

    data_df["text"] = data_df.apply(lambda x: f"<|user|>\n{x['user']}</s>\n<|assistant|>\n{x['assistant']}</s>", axis=1)
   
    dataset = Dataset.from_pandas(data_df)
    return dataset
