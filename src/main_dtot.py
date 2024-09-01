import torch
import json
import subprocess
import transformers
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from eval_utils.model_utils import *
from eval_utils.context import context_by_type
from eval_utils.gpt_utils import get_demo_by_sim
from sklearn.metrics import roc_auc_score, classification_report

subprocess.run(['mkdir', "-p", "./results"])

def search_second_level(
    model, 
    tokenizer,
    context,
    text,
    context_by_type,
    candidate_inputs,
    batch_size = 1,
    num_candidates = 5,
    max_new_tokens = 5, 
    examples = [],
    use_example = True,
    use_rationale = False
):
    model.eval()
    input_texts = []
    for t in text:
        input_texts += [f"""{context}
                        ### Human: "{t}" Does this sentence contain toxic content? 
                        Answer in "Yes" or "No", and then output your rationale.
                        ### Assistant: """,
                    ]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")
    
    encoder_output = model.encoder(
        input_ids = inputs.input_ids.cuda(), 
        attention_mask = inputs.attention_mask.cuda()
    )[0]

    decoder_start_ids = torch.as_tensor(
        [[model.generation_config.decoder_start_token_id] for _ in range(batch_size * num_candidates)],
        dtype=torch.int64,
        device="cuda",
    )

    encoder_output = torch.stack([encoder_output for _ in range(num_candidates)], dim=1)
    encoder_output = encoder_output.reshape(
        -1,
        encoder_output.size(-2), 
        encoder_output.size(-1)
    )
    past_key_values = None

    prob = []
    prob_mask = []
    
    for i in range(candidate_inputs.input_ids.size(1)):
        if i == 0:
            out = model.decoder(
                input_ids=decoder_start_ids,
                encoder_hidden_states=encoder_output,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            token = candidate_inputs.input_ids[:, i: i + 1]
            token = torch.stack([token for _ in range(batch_size)], dim=0).reshape(-1,1)
            logits = torch.softmax(model.lm_head(out[0]), axis=-1)
            prob.append(torch.diag(logits[:, 0, token.reshape(-1)]).cpu())
            
        else:
            out = model.decoder(
                input_ids=token.cuda(),
                encoder_hidden_states=encoder_output,
                use_cache=True,
                past_key_values=past_key_values,
            )
            token = candidate_inputs.input_ids[:, i: i + 1]
            token = torch.stack([token for _ in range(batch_size)], dim=0).reshape(-1,1)
            logits = torch.softmax(model.lm_head(out[0]), axis=-1)
            prob.append(torch.diag(logits[:, 0, token.reshape(-1)]).cpu())

    prob = torch.stack(prob, dim=1)
    prob = prob / torch.norm(prob, dim=0).unsqueeze(0)
    prob = torch.sum(prob, dim=-1)
    prob = prob.reshape(batch_size, num_candidates)
    prob_ids = torch.argmax(prob, dim=-1)

    context_type = list(context_by_type.keys())
    input_texts = []
    for i in range(len(text)):
        ct = context_type[prob_ids[i].item()]
        
        if use_rationale:
            example_str = "Examples:"
            for (neg_s, neg_r, pos_s, pos_r) in examples:
                example_str += f"""
                Sentence: "{neg_s}".
                This sentence does not contain {ct} content. Rationale: {neg_r}.
                Sentence: "{pos_s}".
                This sentence contains {ct} content. Rationale: {pos_r}.
                """
                break
                
        elif use_example:
            example_str = "Examples:"
            for (neg_s, neg_r, pos_s, pos_r) in examples:
                example_str += f"""
                Sentence: "{neg_s}".
                This sentence does not contain {ct} content.
                Sentence: "{pos_s}".
                This sentence contains {ct} content.
                """
        
        else:
            example_str = ""
            
        input_texts += [f"""{context_by_type[ct]}
                        {example_str}
                        ### Human: "{text[i]}". Does this sentence contain {ct} content? Answer in "Yes" or "No", and then output your rationale.
                        ### Assistant: """,
                    ]
        
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")
        
    logits_processor = prepare_logits_processor(
        temperature=0.7, repetition_penalty=1.2, top_p=1, top_k=-1
    )
    outputs = model.generate(
        **inputs, 
        do_sample=False,
        max_new_tokens=max_new_tokens, 
        logits_processor=logits_processor,
        return_dict_in_generate=True, 
        output_scores=True
    )
        
    # Get the scores for each token generated with Greedy Search
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    
    conf_score = torch.prod(torch.exp(transition_scores), 1)

    # Convert output ids into tokens
    response_batch = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    
    # Return generated responses
    return response_batch, conf_score.cpu().numpy().tolist()


eval_params = {
    "model_name": "lmsys/fastchat-t5-3b-v1.0", #"lmsys/fastchat-t5-3b-v1.0", #"outputs/fastchat-t5-ft-bn-3ep-v2_ep_0",
    "score_th": 0.5,
    "max_new_tokens": 10,
    "batch_size": 5,
    "dataset_root_path": "./dataset",
    "dataset_name": "dynhate_sample",
    "dataset_type": "test",
    "use_example": True,
    "use_rationale": True,
    "num_tokens_for_conf": 10,
    "search_th": 90,
    "num_sample": 3,
    "method": "_dtot"
}
model_name = eval_params["model_name"] #"outputs/fastchat-t5-ft-full-10ep-aug_ep_2" #"lmsys/fastchat-t5-3b-v1.0" #"outputs/fastchat-t5-ft-full-5ep_ep_2"

prefix = "Yes, this sentence contains "
candidates = [f"{prefix}hate content.",
              f"{prefix}biased content.",
              f"{prefix}sexual content.",
              f"{prefix}violent content.",
              f"{prefix}bullying content.",
              f"{prefix}implicit content."
]

context = f"Toxic content includes hate speech, biased or discriminatory content,"\
          f"sexual content, violent content, bullying content, and other implicit toxic content ."\
          f"Toxic content can be hateful, aggressive, disrespectful, rude, and unreasonable."

question = "Does this sentence contain toxic content? Answer in 'Yes' or 'No', and then output your rationale."
tokenizer = T5Tokenizer.from_pretrained(model_name)
model =  T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

candidate_inputs = tokenizer(candidates, return_tensors="pt", padding=True)


re_infer = False

data_benign, all_benign_response, all_benign_score, \
data_toxic, all_toxic_response, all_toxic_score = t5_eval(
    model,
    tokenizer,
    model_name = eval_params["model_name"].split("/")[1],
    dataset_root_path = eval_params["dataset_root_path"],
    dataset_name = eval_params["dataset_name"], 
    dataset_type = eval_params["dataset_type"],
    context = context,
    question = question,
    batch_size = eval_params["batch_size"],
    max_new_tokens = eval_params["max_new_tokens"],
    num_tokens_for_conf = eval_params["num_tokens_for_conf"],
    save_results=False,
    re_infer=re_infer
)

use_example = eval_params["use_example"]
use_rationale=eval_params["use_rationale"]
dataset_name = eval_params["dataset_name"]
dataset_type = eval_params["dataset_type"]
score_th = eval_params["score_th"]
search_th = eval_params["search_th"]
num_sample = eval_params["num_sample"]
dataset_root_path = eval_params["dataset_root_path"]
demo_type="train"

if use_example:
    rationale_path = ""
    if use_rationale:
        rationale_path = f"./parsed_response/second_pass/{dataset_name}/{demo_type}/rationale_data.json"
        
    demo_sim_map, data_demo = get_demo_by_sim(
        dataset_name=dataset_name,
        dataset_root_path=dataset_root_path,
        demo_type=demo_type,
        rationale_path=rationale_path
        
    )
    
y_true = []
y_pred = []
y_score = []

for label_type in ["benign", "toxic"]:
    if label_type == "benign":
        all_score = all_benign_score
        all_response = all_benign_response
        all_data = data_benign
        
    else:
        all_score = all_toxic_score
        all_response = all_toxic_response
        all_data = data_toxic
        
    for i in tqdm(range(len(all_score))):
        if all_score[i] < search_th:
            text = [all_data[i]]
            
            examples = []
            
            if use_example:
                neg_s_all = []
                pos_s_all = []
                neg_r_all = []
                pos_r_all = []
                        
                for n in range(num_sample):
                    neg_idx = demo_sim_map[label_type]["benign"][i][-n-1]
                        
                    if data_demo[f"benign_demo_{neg_idx}"]["correct"]:
                        neg_s_all.append(data_demo[f"benign_demo_{neg_idx}"]["prompt"])
                        neg_r_all.append(data_demo[f"benign_demo_{neg_idx}"]["rationale"])

                for n in range(num_sample):
                    pos_idx = demo_sim_map[label_type]["toxic"][i][-n-1]
                    
                    if data_demo[f"toxic_demo_{pos_idx}"]["correct"]:
                        pos_s_all.append(data_demo[f"toxic_demo_{pos_idx}"]["prompt"])
                        pos_r_all.append(data_demo[f"toxic_demo_{pos_idx}"]["rationale"])
                        
                for n in range(min(len(neg_s_all), len(pos_s_all))):
                    neg_s = neg_s_all[n]
                    pos_s = pos_s_all[n]
                    neg_r = neg_r_all[n]
                    pos_r = pos_r_all[n]
                    examples += [(neg_s, neg_r, pos_s, pos_r)]
                
            response, score = search_second_level(
                model, 
                tokenizer,
                context,
                text,
                context_by_type,
                candidate_inputs,
                batch_size = 1,
                num_candidates = len(candidates),
                max_new_tokens = 1,
                examples = examples,
                use_example = use_example,
                use_rationale = False
            )
            all_response[i] = response[0]
            all_score[i] = score[0]

        if label_type == "benign":
            y_true += [0]
        else:
            y_true += [1]
        y_pred += [extract_label_from_response(all_response[i])]
        
        if y_pred[-1] == 0:
            score = 1 - all_score[i]
        else:
            score = all_score[i]

        if score < score_th:
            y_pred[-1] = 0
        else:
            y_pred[-1] = 1
        
        y_score.append(score)


report = classification_report(y_true, y_pred, digits=4)
print(report)
print("Auc score: ", roc_auc_score(y_true, y_score))

save_checkpoints(
    context, 
    model_name.split("/")[-1],
    dataset_name,
    dataset_type,
    data_benign, 
    all_benign_response,
    all_benign_score,
    data_toxic,
    all_toxic_response,
    all_toxic_score,
    version=eval_params["method"]
)