import json
import pandas as pd
import re
import random


def sample_text_from_SBIC(dataset_root_path = "toxigen"):
    data_benign_sample = {}
    data_toxic_sample = {}
    
    for mode in ["train", "dev", "test"]:
        data = pd.read_csv(f"./{dataset_root_path}/SBIC.v2.agg.{mode}.csv")
        clean_post = lambda x: re.sub(r'@\w+', '', x)
        filter_special_chars = lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x)
    
        data = data.loc[:, ["post", "hasBiasedImplication"]]
        data['post'] = data['post'].map(clean_post).map(filter_special_chars)

        benign_data = data[data["hasBiasedImplication"] == 1]
        toxic_data = data[data["hasBiasedImplication"] == 0]
        
        random.seed(42)
        if mode == "train":
            data_benign_sample[mode] = random.sample(benign_data["post"].tolist(), 2000)
            data_toxic_sample[mode] = random.sample(toxic_data["post"].tolist(), 2000)
        else:
            data_benign_sample[mode] = random.sample(benign_data["post"].tolist(), 500)
            data_toxic_sample[mode] = random.sample(toxic_data["post"].tolist(), 500)

    return data_benign_sample, data_toxic_sample


def sample_text_from_dynhate(dataset_root_path = "toxigen"):
    # Dataset: Dynamic hate
    data = pd.read_csv(f"./{dataset_root_path}/dynhate.csv")
    data = data[["text", "label", "split"]].dropna()
    text = data["text"].tolist()
    label = data["label"].tolist()
    split = data["split"].tolist()

    data_benign = {
        "train": [],
        "test": [],
        "dev": []
    }

    data_toxic = {
        "train": [],
        "test": [],
        "dev": []
    }

    for i in range(len(text)):
        if label[i] == "hate":
            data_toxic[split[i]].append(text[i])
        else:
            data_benign[split[i]].append(text[i])
    
    random.seed(42)
    data_benign_sample = {
        "train": random.sample(data_benign["train"], 2000),
        "test": random.sample(data_benign["test"], 500),
        "dev": random.sample(data_benign["dev"], 500)
    }

    data_toxic_sample = {
        "train": random.sample(data_toxic["train"], 2000),
        "test": random.sample(data_toxic["test"], 500),
        "dev": random.sample(data_toxic["dev"], 500)
    }
    
    return data_benign_sample, data_toxic_sample


def load_dataset(
    dataset_name = "dynhate",
    dataset_root_path = "toxigen"
):
    # Dataset: Dynamic hate
    if dataset_name == "dynhate":
        data = pd.read_csv(f"{dataset_root_path}/dynhate.csv")
        data = data[["text", "label", "split"]].dropna()
        text = data["text"].tolist()
        label = data["label"].tolist()
        split = data["split"].tolist()
        
        data_benign = {
            "train": [],
            "test": [],
            "dev": []
        }
        
        data_toxic = {
            "train": [],
            "test": [],
            "dev": []
        }
        
        for i in range(len(text)):
            if label[i] == "hate":
                data_toxic[split[i]].append(text[i])
            else:
                data_benign[split[i]].append(text[i])
                
    # Dataset: Toxigen
    elif dataset_name == "toxigen":
        data = pd.read_csv(f"{dataset_root_path}/annotated_test.csv")
        data_benign_test = data[data["toxicity_human"] < 3]["text"].tolist()
        data_toxic_test = data[data["toxicity_human"] > 3]["text"].tolist()

        data = pd.read_csv(f"{dataset_root_path}/annotated_train.csv")
        data_benign_train = data[data["toxicity_human"] < 3]["text"].tolist()
        data_toxic_train = data[data["toxicity_human"] > 3]["text"].tolist()
        
        data_benign = {
            "train": data_benign_train,
            "test": data_benign_test
        }
        
        data_toxic = {
            "train": data_toxic_train,
            "test": data_toxic_test
        }
    
    # Dataset: ADS RAI + GAI
    elif dataset_name == "RAI":
        toxic_cate = [
            "Hate and Intolerance",
            "Violence",
            "Sexual Content",
            "Bullying",
            "Religion & Folklore"
        ]
        toxic_data = pd.read_csv(f"{dataset_root_path}/RAI5k.csv")
        toxic_data = toxic_data[[
            "responsible_ai_prompt_generation_maincategory",
            "responsible_ai_prompt_generation_prompt"]].dropna()
        text = toxic_data["responsible_ai_prompt_generation_prompt"].tolist()
        label = toxic_data["responsible_ai_prompt_generation_maincategory"].tolist()
        
        data_toxic_test = []
        for i in range(len(label)):
            for cate in toxic_cate:
                if cate in label[i]:
                    data_toxic_test.append(text[i])

                    
        benign_data = pd.read_csv(f"{dataset_root_path}/GAI9k.csv")
        benign_data = benign_data[[
            "general_demonstrations_prompt_generation_prompt",
            "general_demonstrations_verification_harmless"]].dropna()
        text = benign_data["general_demonstrations_prompt_generation_prompt"].tolist()
        label = benign_data["general_demonstrations_verification_harmless"].tolist()
        
        data_benign_test = []
        for i in range(len(label)):
            if json.loads(label[i])["harmlessYes"] == True:
                data_benign_test.append(text[i])
                    
                    
        data_benign = {
            "train": [],
            "test": data_benign_test
        }
        
        data_toxic = {
            "train": [],
            "test": data_toxic_test
        }
    
    elif dataset_name == "sbic_sample":
        return sample_text_from_SBIC(dataset_root_path)
    
    elif dataset_name == "dynhate_sample":
        return sample_text_from_dynhate(dataset_root_path)
        
        
    return data_benign, data_toxic


def generate_prompt_fst5(text_batch, context="", question="", decoder_only=False):
    if question == "":
        # Prompt question
        question = "Does this sentence contain toxic content? Answer yes or no, and explain your answer."
    
    # Generate prompt from template
    prompt_batch = []
    for text in text_batch:
        if not decoder_only:
            prompt_batch += [
                f"""{context}
                ### Human: {text} {question}
                ### Assistant: """
            ]
        else:
            prompt_batch += [
                f"""{context}
                <human>: "{text}" {question}
                <assistant>: """
            ]
    
    # Return generated prompts
    return prompt_batch



def generate_prompt_fst5_rationale(text_batch, context="", is_benign=False):
    # Prompt question
    question = "Why does this sentence contain toxic content?"
    
    # Generate prompt from template
    prompt_batch = []
    for text in text_batch:
        if is_benign:
            prompt_batch += [
                f"""{context}
                ### Human: {text} Why does this sentence not contain toxic content?"
                ### Assistant: """
            ]
        else:
            prompt_batch += [
                f"""{context}
                ### Human: {text} Why does this sentence contain toxic content?"
                ### Assistant: """
            ]
    # Return generated prompts
    return prompt_batch


def extract_label_from_response(response):
    response = re.sub(r'[^a-zA-Z0-9\s]', '', response)
    response = response.split("Assistant ")[-1]
    response = response[:3].lower()
    if response.startswith("yes"):
        return 1
    else:
        return 0
    
    
def save_checkpoints(
    context, 
    model_name,
    dataset_name,
    dataset_type,
    data_benign, 
    all_benign_response,
    all_benign_score,
    data_toxic,
    all_toxic_response,
    all_toxic_score,
    version=""
):
    
    # Save response results
    with open(f"./results/prediction_{model_name}_{dataset_name}_{dataset_type}{version}.json", "w") as json_file:
        json.dump(
            {"benign": 
             {
                 "text": data_benign, 
                 "output": all_benign_response, 
                 "score": all_benign_score
             },
             "toxic": 
             {
                 "text": data_toxic, 
                 "output": all_toxic_response, 
                 "score": all_toxic_score
             },
             "context": context
            },
            json_file
        )