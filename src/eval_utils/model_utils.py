import torch
from transformers import pipeline
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics import classification_report
from tqdm import tqdm
from .data_utils import load_dataset, save_checkpoints, extract_label_from_response, generate_prompt_fst5
from fastchat.serve.inference import prepare_logits_processor


def load_model(model_name="bert", ):
    if model_name == "bert":
        classifier = pipeline("text-classification", model="tomh/toxigen_roberta")
        return classifier, None
        
    elif "bert" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
        return classifier, None
        
    else:
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
        model =  T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto") 
        
    return model, tokenizer
        
        
def bert_eval(
    bert_model,
    model_name,
    dataset_root_path = "toxigen",
    dataset_name = "toxigen", 
    dataset_type = "test"
):
    data_benign, data_toxic = load_dataset(
        dataset_name = dataset_name,
        dataset_root_path = dataset_root_path
    )
    data_benign = data_benign[dataset_type]
    data_toxic = data_toxic[dataset_type]
    
    all_benign_response = []
    all_benign_score = []
    for text in tqdm(data_benign):
        ret = bert_model(text)[0]
        label = int(ret["label"].split("_")[-1])
        all_benign_response.append(label)
        all_benign_score.append(float(ret["score"]))
        

    all_toxic_response = []
    all_toxic_score = []
    for text in tqdm(data_toxic):
        ret = bert_model(text)[0]
        label = int(ret["label"].split("_")[-1])
        all_toxic_response.append(label)
        all_toxic_score.append(float(ret["score"]))

    save_checkpoints(
        context="", 
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        data_benign=data_benign, 
        all_benign_response=all_benign_response,
        all_benign_score=all_benign_score,
        data_toxic=data_toxic,
        all_toxic_response=all_toxic_response,
        all_toxic_score=all_toxic_score
    )
            
    y_true = [0 for _ in all_benign_response] + [1 for _ in all_toxic_response]
    y_pred = all_benign_response + all_toxic_response

    report = classification_report(y_true, y_pred)

    with open(f"./results/report_{model_name}_{dataset_name}_{dataset_type}.txt", 'w') as text_file:
        text_file.write(report)

    return 
        
        
def generate_response_fst5(
    model, 
    tokenizer, 
    text_batch, 
    context="", 
    question="",
    max_new_tokens=3,
    num_tokens_for_conf=10,
    decoder_only=False
):
    # Convert a batch of text into a batch of prompt
    prompt_batch = generate_prompt_fst5(text_batch, context=context, question=question, decoder_only=decoder_only)

    # Get input ids
    if decoder_only:
        inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, return_token_type_ids=False).to("cuda")
    else:
        inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True).to("cuda")

    logits_processor = prepare_logits_processor(
        temperature=0.7, repetition_penalty=1.2, top_p=1, top_k=-1
    )
    # Get LLM outputs (below use parameters from FastChat-T5)
    if decoder_only:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            return_dict_in_generate=True, 
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id
        )
        input_length = inputs.input_ids.size(-1) + 1
        sequences = outputs.sequences[:, input_length:]
        scores = outputs.scores[1:]
    else:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            return_dict_in_generate=True, 
            output_scores=True
        )
        sequences = outputs.sequences
        scores = outputs.scores

    # Get the scores for each token generated with Greedy Search
    transition_scores = model.compute_transition_scores(
        sequences, scores, normalize_logits=True
    )
    
    conf_score = torch.prod(torch.exp(transition_scores[:,:num_tokens_for_conf]), 1)

    # Convert output ids into tokens
    response_batch = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    
    
    # Return generated responses
    return response_batch, conf_score.float().cpu().numpy().tolist()


def t5_eval(
    t5_model,
    t5_tokenizer,
    model_name = "fst5",
    dataset_root_path = "toxigen",
    dataset_name = "toxigen", 
    dataset_type = "test",
    context = "",
    question = "",
    batch_size = 20,
    max_new_tokens = 3,
    save_results=True,
    num_tokens_for_conf=10,
    re_infer=False,
    decoder_only=False,
    version=""
):
    
    # Load toxigen dataset
    data_benign, data_toxic = load_dataset(
        dataset_name = dataset_name,
        dataset_root_path = dataset_root_path
    )
    data_benign = data_benign[dataset_type]
    data_toxic = data_toxic[dataset_type]

    if not re_infer:
        with open(f"./results/prediction_{model_name}_{dataset_name}_{dataset_type}{version}.json", 'r') as json_file:
            data = json.load(json_file)

        all_benign_response = data["benign"]["output"]
        all_benign_score = data["benign"]["score"]
        all_toxic_response = data["toxic"]["output"]
        all_toxic_score = data["toxic"]["score"]
            
        return data_benign, all_benign_response, all_benign_score, \
            data_toxic, all_toxic_response, all_toxic_score
            
    # Generate response on toxic text
    num_batch = len(data_toxic) // batch_size + 1
    all_toxic_response = []
    all_toxic_score = []
    all_benign_response = []
    all_benign_score = []
    
    for i in tqdm(range(num_batch)):
        # Get data batch
        sid = i * batch_size
        eid = (i + 1) * batch_size
        text_batch = data_toxic[sid: eid]
        
        if len(text_batch) == 0:
            break
            
        response, score = generate_response_fst5(
            t5_model, 
            t5_tokenizer, 
            text_batch, 
            context=context,
            question=question,
            max_new_tokens=max_new_tokens,
            decoder_only=decoder_only
        )
        all_toxic_response += response
        all_toxic_score += score
        
        if i % 5 == 0 or i == num_batch - 1:
            if save_results:
                save_checkpoints(
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
                    version
                )
    if save_results:
        save_checkpoints(
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
            version
        )

    # Generate response on benign text
    num_batch = len(data_benign) // batch_size + 1
    for i in tqdm(range(num_batch)):
        sid = i * batch_size
        eid = (i + 1) * batch_size
        text_batch = data_benign[sid: eid]
        
        if len(text_batch) == 0:
            break
        
        response, score = generate_response_fst5(
            t5_model, 
            t5_tokenizer, 
            text_batch, 
            context=context,
            question=question,
            max_new_tokens=max_new_tokens,
            decoder_only=decoder_only
        )
        all_benign_response += response
        all_benign_score += score

        if i % 5 == 0 or i == num_batch - 1:
            if save_results:
                save_checkpoints(
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
                    version
                )
    if save_results:
        save_checkpoints(
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
            version
        )
        
    y_true = [0 for _ in all_benign_response] + [1 for _ in all_toxic_response]
    y_pred = [
        extract_label_from_response(r) for r in all_benign_response
    ] + [
        extract_label_from_response(r) for r in all_toxic_response
    ]

    report = classification_report(y_true, y_pred, digits=4)

    if save_results:
        with open(f"./results/report_{model_name}_{dataset_name}_{dataset_type}{version}.txt", 'w') as text_file:
            text_file.write(report)
            
    return data_benign, all_benign_response, all_benign_score, \
        data_toxic, all_toxic_response, all_toxic_score
    
        