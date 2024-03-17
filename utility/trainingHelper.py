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
DATASET_NAME="tatsu-lab/alpaca"
RANDOM_SEED = 42
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 2e-5
LR_SCHEDULER = "cosine"
OPTIMIZER = "paged_adamw_32bit"
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-5
WARMUP_STEPS = 2000
LOGGING_STEPS = 1000
WEIGHT_DECAY = 0.1
MAX_SEQUENCE_LENGTH = 2048
FP16 = False
GRADIENT_CLIPPING = 1
script_dir = os.path.dirname(__file__)
rel_path = "\\files\\"
FILE_PATH_LOCATION = script_dir + "\\..\\"+ rel_path


def start_training_model():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    tokenizer = LlamaTokenizerFast.from_pretrained(BASE_MODEL)    
    tokenizer.padding_side = "right"    

    dataset = get_dataset()

    model = LlamaForCausalLM.from_pretrained(BASE_MODEL)

    training_params = TrainingArguments(
        output_dir="./deleteme",
        fp16=FP16,
        use_cpu=True
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
    global_var = model
    return global_var

    
def get_response(message):
    pipe = pipeline(task="text-generation",
                model=global_var,
                tokenizer=tokenizer,
                max_new_tokens=512,
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
    data_df.to_csv("output.csv", index=False)
    dataset = Dataset.from_pandas(data_df)
    return dataset
    