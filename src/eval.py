import subprocess
import json
from sklearn.metrics import roc_auc_score, classification_report
from eval_utils.model_utils import *
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Argument Parser')

# Add string arguments
parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
parser.add_argument('--model_name', type=str, help='Name of the model')
parser.add_argument('--method', type=str, help='if using dtot', default="")
parser.add_argument('--version', type=str, help='version number if needed', default="")
parser.add_argument('--re_infer', action='store_true', help='Whether to use rerun inference only')
parser.add_argument('--score_th', type=float, help='Threshold for calculating F1/accuracy', default=0.5)

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
model_name = args.model_name
re_infer = args.re_infer
score_th = args.score_th
method = args.method
version = args.version

subprocess.run(['mkdir', "-p", "./results"])

eval_params = {
    "model_name": model_name, #"outputs/fastchat-t5-ft-full-dynhate-5ep-v1-all-gpt_ep_2", #"lmsys/fastchat-t5-3b-v1.0", #"outputs/fastchat-t5-ft-full-5ep-v1-all-gpt-base-good_ep_1", #"outputs/fastchat-t5-ft-full-5ep-new-base_ep_3", #"outputs/flan-t5-ft-bn-10ep_ep_6",
    "score_th": score_th,
    "max_new_tokens": 1,
    "batch_size": 5,
    "dataset_root_path": "./dataset",
    "dataset_name": dataset_name,
    "dataset_type": "test",
    "run_infer": re_infer,
    "method": method,
    "version": version
}

if eval_params["run_infer"]:
    if "bert" in eval_params["model_name"]:
        # Evaluate bert-based models
        bert_model, _ = load_model(eval_params["model_name"])

        bert_eval(
            bert_model,
            model_name=eval_params["model_name"].split("/")[-1],
            dataset_root_path = eval_params["dataset_root_path"],
            dataset_name = eval_params["dataset_name"], 
            dataset_type = eval_params["dataset_type"]
        )

    else:
        # Evaluate t5-based models
        context = f"Toxic content includes hate speech, biased or discriminatory content,"\
                f"sexual content, violent content, bullying content, and other implicit toxic content ."\
                f"Toxic content can be hateful, aggressive, disrespectful, rude, and unreasonable."
                
        question = "Does this sentence contain toxic content? Answer in 'Yes' or 'No', and then output your rationale."

        t5_model, t5_tokenizer = load_model(eval_params["model_name"])
        t5_model.eval()

        t5_eval(
            t5_model,
            t5_tokenizer,
            model_name = eval_params["model_name"].split("/")[1],
            dataset_root_path = eval_params["dataset_root_path"],
            dataset_name = eval_params["dataset_name"], 
            dataset_type = eval_params["dataset_type"],
            context = context,
            question = question,
            batch_size = eval_params["batch_size"],
            max_new_tokens = eval_params["max_new_tokens"],
            re_infer=True,
            save_results=True,
            version=eval_params["version"]
        )


# Analyze classification results and generate auc score
model_name = eval_params["model_name"].split("/")[-1]
dataset_name = eval_params["dataset_name"]
dataset_type = eval_params["dataset_type"]
score_th = eval_params["score_th"]

with open(f"./results/prediction_{model_name}_{dataset_name}_{dataset_type}{method}{version}.json", "r") as json_file:
    data = json.load(json_file)

all_benign_response = data["benign"]["output"]
all_benign_score = data["benign"]["score"]
all_toxic_response = data["toxic"]["output"]
all_toxic_score = data["toxic"]["score"]

y_true = []
y_pred = []
y_score = []

for i in range(len(all_benign_score)):
    if all_benign_score[i] >= 0.:
        y_true += [0]
        if "bert" in model_name:
            y_pred += [int(all_benign_response[i])]
        else:
            y_pred += [extract_label_from_response(all_benign_response[i])]
        
        if y_pred[-1] == 0:
            score = 1 - all_benign_score[i]
        else:
            score = all_benign_score[i]

        if score < score_th:
            y_pred[-1] = 0
        else:
            y_pred[-1] = 1
        
        y_score.append(score)

        
for i in range(len(all_toxic_score)):
    if  all_toxic_score[i] >= 0.:
        y_true += [1]
        if "bert" in model_name:
            y_pred += [int(all_toxic_response[i])]
        else:
            y_pred += [extract_label_from_response(all_toxic_response[i])]
        
        if y_pred[-1] == 0:
            score = 1 - all_toxic_score[i]
        else:
            score = all_toxic_score[i]
            
        if score < score_th:
            y_pred[-1] = 0
        else:
            y_pred[-1] = 1
            
        y_score.append(score)
            

report = classification_report(y_true, y_pred, digits=4)
print(report)
print("Auc score: ", roc_auc_score(y_true, y_score))