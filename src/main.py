import os
import json
import random
import subprocess
from tqdm import tqdm
from train_utils.trainer import train, validate
from train_utils.dataset import PromptResponseDataset, generate_prompt_fst5
import torch
import numpy as np
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import logging
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Argument Parser')

# Add string arguments
parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
parser.add_argument('--model_name', type=str, help='Name of the fine-tuned model')
parser.add_argument('--pretrained_model', type=str, help='Name of the pretrained model', default="lmsys/fastchat-t5-3b-v1.0")

# Add boolean arguments
parser.add_argument('--all_gpt', action='store_true', help='Whether to use all GPT models')
parser.add_argument('--use_dtot', action='store_true', help='Whether to use dtot method')
parser.add_argument('--binary_only', action='store_true', help='Whether to use binary models only')

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
pretrained_model = args.pretrained_model
model_name = args.model_name
all_gpt = args.all_gpt
binary_only = args.binary_only
use_dtot = args.use_dtot


if dataset_name == "toxigen":
    dataset_root = "./parsed_response/toxigen"
    if use_dtot:
        with open(f"{dataset_root}/rationale_data_train_90_1st_demo.json", "r") as json_file:
            raw_data = json.load(json_file)
    else:
        with open(f"{dataset_root}/rationale_data_train_0_1st.json", "r") as json_file:
            raw_data = json.load(json_file) 

else:
    dataset_type = "train"
    dataset_root = "./parsed_response"
    if use_dtot:
        version = "second_pass_with_demo_and_r"
        rationale_path = f"{dataset_root}/{version}/{dataset_name}/{dataset_type}/rationale_data_with_demo_and_r.json"
    else:
        version = "first_pass"
        rationale_path = f"{dataset_root}/{version}/{dataset_name}/{dataset_type}/rationale_data.json"
    
    with open(rationale_path, "r") as json_file:
        raw_data = json.load(json_file)
        
    print(dataset_name, model_name, use_dtot, binary_only, version)
    

context = f"Toxic content includes hate speech, biased or discriminatory content,"\
          f"sexual content, violent content, bullying content, and other implicit toxic content."\
          f"Toxic content can be hateful, aggressive, disrespectful, rude, and unreasonable."
question = "Does this sentence contain toxic content? Answer in 'Yes' or 'No', and then output your rationale."


source_text = []
target_text = []
for data_id in raw_data:
    if all_gpt:
        raw_data[data_id]["correct"] = 1

    # Generate prompt as source
    if dataset_name == "toxigen":
        text = raw_data[data_id]["prompt"][2:-1].strip() + "."
    else:
        text = raw_data[data_id]["prompt"].strip() + "."
        
    prompt = generate_prompt_fst5(text, context, question)
    source_text.append(prompt)
    
    # Generate answer and rationale as target
    rationale = raw_data[data_id]["rationale"]
    if raw_data[data_id]["pred"].lower().startswith("yes"):
        raw_data[data_id]["pred"] = "yes"
            
    else:
        raw_data[data_id]["pred"] = "no"
            
    if raw_data[data_id]["pred"] == "yes":
        if binary_only:
            if raw_data[data_id]["correct"] == 1:
                response = f"""Yes."""
            else:
                
                response = f"""No."""
        else:
            if raw_data[data_id]["correct"] == 1:
                response = f"""Yes, this sentence contains toxic content. Rationale: {rationale}"""
            else:
                response = f"""No."""
                
    else:
        if binary_only:
            if raw_data[data_id]["correct"] == 1:
                response = f"""No."""
            else:
                response = f"""Yes."""
        else:
            if raw_data[data_id]["correct"] == 1:
                response = f"""No, this sentence does not contain toxic content. Rationale: {rationale}"""
            else:
                response = f"""Yes."""
                   
    target_text.append(response)
    
    
print(source_text[10])
print(target_text[10])
print(len(source_text), len(raw_data))


# Combine the input and label lists
raw_data = list(zip(source_text, target_text))

# Shuffle the combined list
random.seed(42)  # Set a random seed for reproducibility
random.shuffle(raw_data)

# Unzip the shuffled list back into separate input and label lists
shuffled_inputs, shuffled_targets = zip(*raw_data)

train_size = int(len(shuffled_inputs) * 0.95)
data = {
    "train": {
        "source_text": shuffled_inputs[:train_size], 
        "target_text": shuffled_targets[:train_size]
    },
    "val": {
        "source_text": shuffled_inputs[train_size:], 
        "target_text": shuffled_targets[train_size:]
    }
}

model_params={
    "MODEL":  pretrained_model, #"google/flan-t5-base"
    "TRAIN_BATCH_SIZE":8,          # training batch size
    "VALID_BATCH_SIZE":8,          # validation batch size
    "TRAIN_EPOCHS":5,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":256,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":256,   # max length of target text
    "SEED": 42,                     # set seed for reproducibility 
    "SAVED_MODEL_FILES": model_name,
    "RATIONALE_LOSS_W": 1
}



def create_logger(log_name="test"):
    # Create a logger
    subprocess.run(['mkdir', "-p", "./logs"])
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a file handler and set its log level
    file_handler = logging.FileHandler(f'logs/{log_name}.log', mode='w')
    file_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Example usage
    logger.info('Info message')
    
    return logger



"""
T5 trainer

"""
def main():
    accelerator = Accelerator()
    output_dir = "outputs"
    logger = create_logger(model_params["SAVED_MODEL_FILES"])

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"]) # pytorch random seed
    np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    logger.info(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    logger.info(model_params)
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"], device_map="auto")

    # logging
    logger.info(f"[Data]: Reading data...\n")

    train_source_text = data["train"]["source_text"]
    train_target_text = data["train"]["target_text"]
    val_source_text = data["val"]["source_text"]
    val_target_text = data["val"]["target_text"]


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = PromptResponseDataset(tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], train_source_text, train_target_text)
    val_set = PromptResponseDataset(tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], val_source_text, val_target_text)


    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }


    val_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    train_dataloader = DataLoader(training_set, **train_params)
    val_dataloader = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    num_epochs = model_params["TRAIN_EPOCHS"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Training loop
    logger.info(f'[Initiating Fine Tuning]...\n')

    for epoch in range(num_epochs):
        train(
            epoch, 
            tokenizer, 
            model, 
            train_dataloader, 
            optimizer, 
            lr_scheduler,
            accelerator,
            log_interval=10,
            logger=logger,
            rationale_loss_w=model_params["RATIONALE_LOSS_W"]
        )
        report = validate(
            epoch, 
            tokenizer, 
            model, 
            val_dataloader,
            log_interval=10,
            logger=logger
        )
        acc, precision, recall = report["accuracy"], report["1"]["precision"], report["1"]["recall"]
        logger.info(f"Accuracy: {acc}, precision: {precision}, recall: {recall}")

        if epoch % 1 == 0:
            logger.info(f"[Saving Model]...\n")
            #Saving the model after training
            path = os.path.join(output_dir, model_params["SAVED_MODEL_FILES"])
            model = accelerator.unwrap_model(model)
            model.save_pretrained(path + f"_ep_{epoch}")
            tokenizer.save_pretrained(path + f"_ep_{epoch}")
            model = accelerator.prepare(model)

    logger.info(f"""[Model] Model saved @ {os.path.join(output_dir, model_params["SAVED_MODEL_FILES"])}\n""")

from accelerate import notebook_launcher
notebook_launcher(main, num_processes=1)