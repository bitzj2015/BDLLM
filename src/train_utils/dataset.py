import torch
from torch.utils.data import Dataset


class PromptResponseDataset(Dataset):
    """
    Creating a custom dataset which consists of prompt and response pairs,
    for finetuning the model

    """

    def __init__(
        self, 
        tokenizer, 
        source_len, 
        target_len, 
        source_text, 
        target_text
    ):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = target_text
        self.source_text = source_text

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = self.source_text[index]
        target_text = self.target_text[index]

        source = self.tokenizer.batch_encode_plus(
            [source_text], 
            max_length=self.source_len, 
            pad_to_max_length=True, 
            truncation=True, 
            padding="max_length", 
            return_tensors='pt'
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text], 
            max_length=self.summ_len, 
            pad_to_max_length=True, 
            truncation=True, 
            padding="max_length", 
            return_tensors='pt'
        )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
    
    
class PromptResponseBinaryDataset(Dataset):
    """
    Creating a custom dataset which consists of prompt and response pairs,
    for finetuning the model

    """

    def __init__(
        self, 
        tokenizer, 
        source_len, 
        target_len, 
        source_text, 
        target_text
    ):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = target_text
        self.source_text = source_text

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = self.source_text[index]
        target = self.target_text[index]

        source = self.tokenizer.batch_encode_plus(
            [source_text], 
            max_length=self.source_len, 
            pad_to_max_length=True, 
            truncation=True, 
            padding="max_length", 
            return_tensors='pt'
        )
        target = torch.Tensor([target]).cuda()

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target.squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long)
        }
        
        
def generate_prompt_fst5(text, context="", question=None):
    # Prompt question
    if not question:
        question = "Does this sentence contain toxic content? Answer in 'Yes' or 'No', and then output your rationale."
    
    # Generate prompt from template
    prompt = f"""{context}\n### Human: "{text}" {question}\n### Assistant: """

    # Return generated prompts
    return prompt