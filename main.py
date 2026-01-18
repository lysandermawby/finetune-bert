#!/usr/bin/env python
"""finetuning a custom BERT model based on personal text data, stored in data.txt"""

import click
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.optim as optim
from typing import List
from tqdm import tqdm


def load_BERT():
    """load the BERT model and tokenizer from Huggingface"""
    bert_model_name = "google-bert/bert-base-uncased" # using default BERT
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModelForMaskedLM.from_pretrained(bert_model_name)

    return tokenizer, model


def get_device():
    """find the device available from torch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def mask_tokens(input_ids: torch.Tensor, tokenizer, mask_prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """apply a mask to the tokens"""
    # mask_prob = probability that a random token is masked
    labels = input_ids.clone()

    prob_matrix = torch.full(input_ids.shape, mask_prob)

    # avoid masking special tokens for the tokenizer
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
        special_tokens_mask |= (input_ids == special_id)

    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # setting which tokens to mask
    masked_indices = torch.bernoulli(prob_matrix).bool()

    # set unmasked tokens to not have loss calculated on them
    labels[~masked_indices] = -100

    # replace masked tokens with [MASK], or the equivalent token for this tokenizer
    input_ids[masked_indices] = tokenizer.mask_token_id
    
    return input_ids, labels


def forward_pass(text: str, tokenizer, model, device: str) -> torch.Tensor:
    """performing a single forward pass of the model"""
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')

    input_ids = encoding["input_ids"].clone()
    attention_mask = encoding["attention_mask"]

    # apply masking
    masked_input_ids, labels = mask_tokens(input_ids, tokenizer)

    # move to device
    masked_input_ids = masked_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(
        input_ids = masked_input_ids,
        attention_mask = attention_mask,
        labels=labels
    )

    # returning a torch tensor object
    return outputs.loss


def load_data(data_path: Path) -> List:
    """load the data in the data file into a python string"""
    if not data_path.exists():
        raise ValueError(f"Provided data path {data_path} does not exist")

    data_lst = []
    with open(data_path, "r") as f:
        data_lst = [line.strip() for line in f if line.strip()] # concatenating all of the text 
    
    return data_lst


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('--data', help='data file name', default='data.txt', type=str)
@click.option('--epochs', help="number of training epochs", default=3, type=int)
@click.option('--lr', help="learning rate", default=5e-5, type=float)
@click.option('--accumulation-steps', help='number of samples to accumulate before weight update', default=8, type=int)
@click.option('--output', help='output directory for saved model', default='finetuned-bert', type=str)
def main(data, epochs, lr, accumulation_steps, output):
    data_path = Path(data)
    data_lst = load_data(data_path)
    # assert len(data_lst) > accumulation_steps, "Error: As the length of the provided data is below the accumulation_steps, no loss propagation will occur"

    # find the available device
    device = get_device()

    tokenizer, model = load_BERT()

    # move model to device
    model.to(device)

    # initialising optimiser
    optimizer = optim.AdamW(model.parameters(), lr = lr)

    # main training loop
    model.train()
    epoch_pbar = tqdm(range(epochs), desc="Finetuning BERT...", dynamic_ncols=True)
    for epoch in epoch_pbar:
        total_loss = 0.0
        accumulated_loss = 0.0

        text_pbar = tqdm(enumerate(data_lst), total=len(data_lst), desc=f"Epoch: {epoch+1}", leave=False, unit="text", dynamic_ncols=True)
        for i, text in text_pbar:
            # performing forward pass and calculating loss
            loss = forward_pass(text, tokenizer, model, device) 

            # scale loss by the number of accumulation steps to stabilise learning rate
            scaled_loss = loss / accumulation_steps

            # backward pass without step
            scaled_loss.backward()

            accumulated_loss += loss.item()
            total_loss += loss.item()

            # update steps after accumulating enough gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                text_pbar.set_postfix({"batch_loss": f"{accumulated_loss / accumulation_steps:.4f}"})
                accumulated_loss = 0.0

        # handle remaining samples at end of epoch
        if len(data_lst) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_lst)
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})


    # save model
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
