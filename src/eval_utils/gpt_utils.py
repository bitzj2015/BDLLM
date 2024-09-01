import os
import re
import time
import json
import openai
import subprocess
from tqdm import tqdm
import numpy as np
from eval_utils.data_utils import load_dataset
from eval_utils.context import context_by_type
from eval_utils.gpt_utils import *
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer


def generate_prompt(text, context=""):
    question = f"""Sentence: {text}
    Does this sentence contain toxic content? 
    First, answer in "Yes" or "No".
    Second, rate the toxicity level out of 100.
    Third, explain your rationale briefly.
    An example of output is given below:
    Yes.
    Toxicity level: 10/100.
    Rationale: ...
    """

    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": question}
    ]
    return messages


def generate_prompt_by_rationale(text, context="", toxic_type=""):
    question = f"""Rationale: {text}
    Which type of toxic content does the sentence mentioned in rationale contain:
    1) biased and discrimination content;
    2) hate and intolerance content;
    3) sexual content;
    4) violent content;
    5) bullying content;
    6) others.
    Output the type name only, such as "1) biased and discrimination content".
    """

    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": question}
    ]
    return messages


def get_response_from_chatgpt(
    prompt, 
    prompt_id, 
    api_key, 
    model="gpt-3.5-turbo", 
    temperature=0, 
    response_root_path="", 
    prompt_root_path="",
    max_trials=3
):
    openai.api_key = api_key
    
    trial = 0
    while trial < max_trials:
        time.sleep(1)

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=temperature
            )
        except:
            trial += 1
            continue
        
        with open(f"{response_root_path}/{prompt_id}.json", "w") as json_file:
            json.dump(response, json_file)
            
        with open(f"{prompt_root_path}/{prompt_id}.json", "w") as json_file:
            json.dump({"prompt": prompt}, json_file)
        
        break


def get_response_for_root_node(
    dataset_name,
    dataset_root_path,
    dataset_type,
    response_root_path,
    prompt_root_path,
    context,
    api_key
):
    data_benign, data_toxic = load_dataset(
        dataset_name = dataset_name,
        dataset_root_path = dataset_root_path
    )
    data_benign = data_benign[dataset_type]
    data_toxic = data_toxic[dataset_type]
        

    for i in tqdm(range(len(data_toxic))):
        prompt = generate_prompt(data_toxic[i], context)
        prompt_id = f"toxic_{dataset_type}_{i}"
        if os.path.isfile(f"{response_root_path}/{prompt_id}.json"):
            continue
            
        else:
            get_response_from_chatgpt(
                prompt=prompt, 
                prompt_id=prompt_id, 
                api_key=api_key, 
                response_root_path=response_root_path,
                prompt_root_path=prompt_root_path
            )
                        
    for i in tqdm(range(len(data_benign))):
        prompt = generate_prompt(data_benign[i], context)
        prompt_id = f"benign_{dataset_type}_{i}"
        if os.path.isfile(f"{response_root_path}/{prompt_id}.json"):
            continue

        else:
            get_response_from_chatgpt(
                prompt=prompt, 
                prompt_id=prompt_id, 
                api_key=api_key, 
                response_root_path=response_root_path,
                prompt_root_path=prompt_root_path
            )
            
    return            


def parse_response_for_root_node(
    dataset_name,
    dataset_root_path,
    dataset_type,
    response_root_path,
    prompt_root_path,
    parsed_response_root_path,
    response_leaf_path=None,
    conf_th=90,
    version=""
):

    # Analyze first pass
    checkers = {
        "hate": "hate and intolerance", 
        "bias": "biased and discrimination", 
        "discrimination": "biased and discrimination", 
        "racism": "biased and discrimination", 
        "racist": "biased and discrimination",  
        "stereotype": "biased and discrimination",
        "stereotyping": "biased and discrimination",
        "sexual": "sexual content",
        "violent": "violent content",
        "violence": "violent content",
        "harmful": "bullying content",
        "aggressive": "bullying content",
        "disrespect": "bullying content",
        "rude": "bullying content",
        "implies": "implicit content",
        "promotes": "implicit content",
        "imply": "implicit content",
        "promote": "implicit content"
    }

    y_true = []
    y_pred = []

    valid_samples = 0
    total_samples = 0

    rationale_data = {}

    data_benign, data_toxic = load_dataset(
        dataset_name = dataset_name,
        dataset_root_path = dataset_root_path
    )
    data_benign = data_benign[dataset_type]
    data_toxic = data_toxic[dataset_type]

    prompt = {
        "benign": data_benign,
        "toxic": data_toxic
    }

    for filename in os.listdir(response_root_path):
        if not filename.endswith(".json"):
            continue
            
        tags = filename.split(".")[0].split("_")
        index = tags[-1]
        label = tags[0]
                
        with open(f"{response_root_path}/{filename}", "r") as json_file:
            response_raw = json.load(json_file)['choices'][0]['message']['content']

            response_split = response_raw.split("\n")
            response = []
            for r in response_split:
                if len(r) == 0:
                    continue
                response.append(r)
                

        pred = response[0]
        score = 0

        if len(response) == 3:
            find_score = False
            patterns = [
                r"(\w+)/(\w+)", r"(\d+) out of 100", 
                r"Toxicity level: (\d+)", r"Second: (\d+)", r"2. (\d+)",
                r"(\d+)"]
            for pattern in patterns:
                match = re.search(pattern, response[1])
                if match:
                    score = match.group(1)
                    if score == "N":
                        score = 0
                    else:
                        score = int(score)
                    find_score = True
                    break
                else:
                    continue

            if not find_score:
                total_samples += 1
                continue

        else:
            total_samples += 1
            continue
                    
        pred = pred.split(" ")[-1]        
        flag = 1
        cate = "None"

        if score >= conf_th or score == 0:
            flag = 1
        elif response_leaf_path != None:
            flag = 1
            try:
                with open(f"{response_leaf_path}/{filename}", "r") as json_file:
                    response_raw = json.load(json_file)['choices'][0]['message']['content']
                    response_split = response_raw.split("\n")
                    response = []
                    for r in response_split:
                        if len(r) == 0:
                            continue
                        response.append(r)
                    prev_pred = pred
                    pred = response[0].split(" ")[-1]
                    match = re.search(r"(\w+)/(\w+)", response[1])
                    score = match.group(1)
                    cate = response[1].split(" level: ")[0].strip()
            except:
                total_samples += 1
                continue
        else:
            flag = 0
                

        if pred.startswith("Yes") and flag:
            y_pred.append(1)
            if label == "benign":
                y_true.append(0)
            else:
                y_true.append(1)

        elif pred.startswith("No") and flag:
            y_pred.append(0)
            if label == "benign":
                y_true.append(0)
            else:
                y_true.append(1)  
                
        else:
            total_samples += 1
            continue
            
        valid_samples += 1
        total_samples += 1
        
        if response[-1].startswith("Rationale: "):
            response[-1] = response[-1][11:]
            
        elif response[-1].startswith("Explanation: "):
            response[-1] = response[-1][13:]
            
        if cate != "None":
            toxic_type = cate
        else:
            toxic_type = "None"
            for key in checkers.keys():
                if key in response[2]:
                    toxic_type = checkers[key]
                    break
            
        rationale_data["_".join(tags)] = {
            "prompt": prompt[label][int(index)],
            "response": response_raw,
            "pred": pred,
            "rationale": response[-1].split("Rationale: ")[-1],
            "correct": int(y_true[-1] == y_pred[-1]),
            "toxic type": toxic_type,
            "score": score
        }
        

    report = classification_report(y_true, y_pred, digits=4)
    print(f"In total, the percentile of valid samples we have is {valid_samples / total_samples}")
    print(report)

    with open(f"{parsed_response_root_path}/rationale_data{version}.json", "w") as json_file:
        json.dump(rationale_data, json_file)


def get_cate_for_second_pass(
    parsed_response_root_path,
    response_leaf_cate_path,
    prompt_leaf_cate_path,
    api_key,
    conf_th=90,
    rewrite=False
):

    with open(f"{parsed_response_root_path}/rationale_data.json", "r") as json_file:
        data = json.load(json_file)

    total_samples = 0
    for prompt_id in tqdm(data.keys()):
        if data[prompt_id]["score"] < conf_th and data[prompt_id]["score"] > 0:
            prompt = generate_prompt_by_rationale(data[prompt_id]["rationale"])
            total_samples += 1

            if os.path.isfile(f"{response_leaf_cate_path}/{prompt_id}.json") and not rewrite:
                continue

            else:
                get_response_from_chatgpt(
                    prompt=prompt, 
                    prompt_id=prompt_id, 
                    api_key=api_key, 
                    response_root_path=response_leaf_cate_path,
                    prompt_root_path=prompt_leaf_cate_path
                )
                
    print(f"In total, we have {total_samples} for second pass.")
    
    

def get_rationale_type(response_leaf_cate_path, prompt_id, context_by_type):
    with open(f"{response_leaf_cate_path}/{prompt_id}.json") as json_file:
        cate_data = json.load(json_file)['choices'][0]['message']['content']

    toxic_type = cate_data
    
    if "none" in toxic_type.lower():
        return "others"

    for key in context_by_type.keys():
        if key in toxic_type.lower():
            toxic_type = key
            break
                    
    return toxic_type

        
def generate_prompt_by_context_type(text, context="", toxic_type="", examples=""):
    question = f"""Sentence: "{text}".
    Does this sentence contain {toxic_type} content? 
    First, answer in "Yes" or "No".
    Second, rate the {toxic_type} level out of 100.
    Lastly, explain your rationale briefly.
    The output format is given below:
    Answer: ...
    {toxic_type} level: .../100.
    Rationale: ...
    """

    messages=[
        {"role": "system", "content": context + examples},
        {"role": "user", "content": question}
    ]
    return messages



def get_response_for_leaf_node(
    parsed_response_root_path,
    parsed_demo_response_root_path,
    response_leaf_cate_path,
    response_leaf_path,
    prompt_leaf_path,
    api_key,
    conf_th=90,
    use_example=False,
    use_rationale=False,
    num_samples=3,
    demo_sim_map=None,
    data_demo=None,
    rewrite=False
):

    with open(f"{parsed_response_root_path}/rationale_data.json", "r") as json_file:
        data = json.load(json_file)
    
    total_samples = 0
    for prompt_id in tqdm(data.keys()):
        if data[prompt_id]["score"] < conf_th and data[prompt_id]["score"] > 0:
            toxic_type = get_rationale_type(response_leaf_cate_path, prompt_id, context_by_type)

            tags = prompt_id.split("_")
            examples = ""
            
            if use_example:
                idx = int(tags[-1])   
                neg_s_all = []
                pos_s_all = []
                neg_r_all = []
                pos_r_all = []
                
                label_type = tags[0]
                for n in range(num_samples):
                    neg_idx = demo_sim_map[label_type]["benign"][idx][-n-1]
                    
                    if data_demo[f"benign_demo_{neg_idx}"]["correct"] and \
                        data_demo[f"benign_demo_{neg_idx}"]["score"] >= 90:
                        neg_s_all.append(data_demo[f"benign_demo_{neg_idx}"]["prompt"])
                        neg_r_all.append(data_demo[f"benign_demo_{neg_idx}"]["rationale"])
                    
                    if len(neg_s_all) == 3:
                        break

                for n in range(num_samples):
                    pos_idx = demo_sim_map[label_type]["toxic"][idx][-n-1]
                        
                    if data_demo[f"toxic_demo_{pos_idx}"]["correct"] and \
                        data_demo[f"toxic_demo_{pos_idx}"]["score"] >= 90:
                        pos_s_all.append(data_demo[f"toxic_demo_{pos_idx}"]["prompt"])
                        pos_r_all.append(data_demo[f"toxic_demo_{pos_idx}"]["rationale"])
                    
                    if len(pos_s_all) == 3:
                        break
                    
                examples = "Examples:"
                for n in range(min(len(neg_s_all), len(pos_s_all))):
                    neg_s = neg_s_all[n]
                    pos_s = pos_s_all[n]
                    neg_r = neg_r_all[n]
                    pos_r = pos_r_all[n]

                    if use_rationale:
                        examples += f"""
                        Sentence: "{neg_s}".
                        This sentence does not contain {toxic_type} content. Rationale: {neg_r}
                        Sentence: "{pos_s}".
                        This sentence contains {toxic_type} content. Rationale: {pos_r}
                        """
                    else:
                        examples += f"""
                        Sentence: "{neg_s}".
                        This sentence does not contain {toxic_type} content.
                        Sentence: "{pos_s}".
                        This sentence contains {toxic_type} content.
                        """
                     
            prompt = generate_prompt_by_context_type(
                text = data[prompt_id]["prompt"],
                context = context_by_type[toxic_type],
                toxic_type = toxic_type,
                examples = examples
            )


            if os.path.isfile(f"{response_leaf_path}/{prompt_id}.json") and not rewrite:
                continue

            else:
                get_response_from_chatgpt(
                    prompt=prompt, 
                    prompt_id=prompt_id, 
                    api_key=api_key, 
                    response_root_path=response_leaf_path,
                    prompt_root_path=prompt_leaf_path
                )


def get_demo_by_sim(
    dataset_name,
    dataset_root_path,
    demo_type="dev",
    input_type="test",
    num_sample=10,
    rationale_path=""
):

    data_benign, data_toxic = load_dataset(
        dataset_name = dataset_name,
        dataset_root_path = dataset_root_path
    )

    data_benign_test = data_benign[input_type]
    data_toxic_test = data_toxic[input_type]

    data_benign_demo = data_benign[demo_type]
    data_toxic_demo = data_toxic[demo_type]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    benign_test_embeddings = model.encode(data_benign_test)
    toxic_test_embeddings = model.encode(data_toxic_test)
    benign_demo_embeddings = model.encode(data_benign_demo)
    toxic_demo_embeddings = model.encode(data_toxic_demo)
    
    benign_test_embeddings = np.array([list(emb) for emb in benign_test_embeddings])
    toxic_test_embeddings = np.array([list(emb) for emb in toxic_test_embeddings])
    benign_demo_embeddings = np.array([list(emb) for emb in benign_demo_embeddings])
    toxic_demo_embeddings = np.array([list(emb) for emb in toxic_demo_embeddings])
    
    benign_test_sim_neg = np.matmul(benign_test_embeddings, benign_demo_embeddings.T)
    benign_test_sim_pos = np.matmul(benign_test_embeddings, toxic_demo_embeddings.T)
    benign_test_sim_neg_idx = np.argsort(benign_test_sim_neg, axis=-1)[:, -num_sample:]
    benign_test_sim_pos_idx = np.argsort(benign_test_sim_pos, axis=-1)[:, -num_sample:]

    toxic_test_sim_neg = np.matmul(toxic_test_embeddings, benign_demo_embeddings.T)
    toxic_test_sim_pos = np.matmul(toxic_test_embeddings, toxic_demo_embeddings.T)
    toxic_test_sim_neg_idx = np.argsort(toxic_test_sim_neg, axis=-1)[:, -num_sample:]
    toxic_test_sim_pos_idx = np.argsort(toxic_test_sim_pos, axis=-1)[:, -num_sample:]
    
    demo_sim_map = {
        "benign": {
            "benign": [],
            "toxic": []
        },
        "toxic": {
            "benign": [],
            "toxic": []
        }
    }
    
    
    for i in range(len(toxic_test_sim_neg_idx)):
        demo_sim_map["toxic"]["benign"] += [list(toxic_test_sim_neg_idx[i])]
        
    for i in range(len(toxic_test_sim_pos_idx)):
        demo_sim_map["toxic"]["toxic"] += [list(toxic_test_sim_pos_idx[i])]
          
    for i in range(len(benign_test_sim_neg_idx)):
        demo_sim_map["benign"]["benign"] += [list(benign_test_sim_neg_idx[i])]
        
    for i in range(len(benign_test_sim_pos_idx)):
        demo_sim_map["benign"]["toxic"] += [list(benign_test_sim_pos_idx[i])]
        
    data_demo = {}
    use_rationale = 0
    
    if rationale_path != "":
        with open(rationale_path, "r") as json_file:
            rationale_data = json.load(json_file)
        use_rationale = 1
            
    for i in range(len(data_benign_demo)):
        rationale = ""
        correct = 1
        if use_rationale:
            rationale = rationale_data[f"benign_{demo_type}_{i}"]["rationale"]
            correct = rationale_data[f"benign_{demo_type}_{i}"]["correct"]
            score = rationale_data[f"benign_{demo_type}_{i}"]["score"]
            
        data_demo[f"benign_demo_{i}"] = {
            "prompt": data_benign_demo[i],
            "rationale": rationale,
            "correct": correct,
            "score": int(score)
        }
        
    for i in range(len(data_toxic_demo)):
        rationale = ""
        correct = 1
        if use_rationale:
            rationale = rationale_data[f"toxic_{demo_type}_{i}"]["rationale"]
            correct = rationale_data[f"toxic_{demo_type}_{i}"]["correct"]
            score = rationale_data[f"toxic_{demo_type}_{i}"]["score"]
            
        data_demo[f"toxic_demo_{i}"] = {
            "prompt": data_toxic_demo[i],
            "rationale": rationale,
            "correct": correct,
            "score": int(score)
        }
        
    return demo_sim_map, data_demo