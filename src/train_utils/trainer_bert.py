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
        input_ids = data['source_ids']
        attention_mask = data['source_mask']

        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            labels = target_ids
        )
        
        loss = outputs.loss
            
        if i % log_interval == 0:
            if not logger:
                print(f"[Training] Epoch: {epoch}, step: {i}, loss: {loss.item()}")
            else:
                logger.info(f"[Training] Epoch: {epoch}, step: {i}, loss: {loss.item()}")
        
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
    
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
            input_ids = data['source_ids'].to(dtype = torch.long)
            attention_mask = data['source_mask'].to(dtype = torch.long)
            target_ids = data['target_ids'].to(dtype = torch.long)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predicted labels
            _, preds = torch.max(logits, dim=1)
            
            if i % log_interval == 0:
                if not logger:
                    print(f'[Eval] Completed: {i}')
                else:
                    logger.info(f'[Eval] Completed: {i}')

            predictions += preds.cpu().numpy().tolist()
            actuals += target_ids.cpu().numpy().tolist()

    report = classification_report(actuals, predictions, output_dict=True)
    
    return report
        