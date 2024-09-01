import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss


def train(
    epoch, 
    tokenizer, 
    model, 
    dataloader, 
    optimizer, 
    lr_scheduler,
    accelerator,
    log_interval=10,
    logger=None,
    rationale_loss_w=-1
):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    
    for i, data in enumerate(dataloader, 0):
        target_ids = data['target_ids']
        lm_labels = target_ids.clone().detach()
        lm_labels[target_ids == tokenizer.pad_token_id] = -100
        input_ids = data['source_ids']
        attention_mask = data['source_mask']
        
        decoder_input_ids = torch.cat(
            (
                input_ids[:, -1:].contiguous() * 0 + model.config.decoder_start_token_id, 
                target_ids[:, :-1].contiguous()
            ), 
            dim = -1
        )

        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            decoder_input_ids = decoder_input_ids, 
            labels = lm_labels
        )
        
        if rationale_loss_w == -1:
            loss = outputs[0]
            loss_binary = 0
            loss_rationale = 0
            
        else:
            logits = outputs.logits
            
            loss_binary = loss_fct(
                logits[:, 0, :], 
                lm_labels[:, 0]
            )
            loss_rationale = loss_fct(
                logits[:, 1:, :].reshape(-1, logits.size(-1)), 
                lm_labels[:, 1:].reshape(-1)
            )
            loss = loss_binary + rationale_loss_w * loss_rationale
            
        if i % log_interval == 0:
            if not logger:
                print(f"[Training] Epoch: {epoch}, step: {i}, loss: {loss.item()} ({loss_binary},{loss_rationale})")
            else:
                logger.info(f"[Training] Epoch: {epoch}, step: {i}, loss: {loss.item()} ({loss_binary},{loss_rationale})")
        
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        

def extract_label_from_response(response):
    response = response.lower()
    if response.startswith("yes"):
        return 1
    else:
        return 0
    
    
def validate(
    epoch, 
    tokenizer, 
    model, 
    dataloader,
    log_interval=10,
    logger=None
):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            y = data['target_ids'].to(dtype = torch.long)
            input_ids = data['source_ids'].to(dtype = torch.long)
            attention_mask = data['source_mask'].to(dtype = torch.long)

            generated_ids = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask, 
                max_new_tokens=1, 
                repetition_penalty=1.2, 
                temperature=0.7
            )
            
            preds = [tokenizer.decode(
                g, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(
                t, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True)for t in y]
            
            if i % log_interval == 0:
                if not logger:
                    print(f'[Eval] Completed: {i}')
                else:
                    logger.info(f'[Eval] Completed: {i}')

            predictions += preds
            actuals += target

    y_pred = [extract_label_from_response(r) for r in predictions]
    y_true = [extract_label_from_response(r) for r in actuals]
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return report
        