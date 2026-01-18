#!/usr/bin/env python
"""finetuning a custom BERT model based on personal text data, stored in data.txt"""

import click
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.optim as optim
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def plot_loss(loss_store: List[List[float]], output_path: Path, bin_size: int = 100) -> None:
    """plotting the loss over time during training"""

    # flatten with epoch tracking
    records = []
    step = 0
    for epoch_idx, epoch_losses in enumerate(loss_store):
        for loss_val in epoch_losses:
            records.append({
                'step': step,
                'epoch': epoch_idx,
                'loss': loss_val,
                'bin': step // bin_size
            })
            step += 1

    df = pd.DataFrame(records)

    # finding the central x values of bins. Should be an irrelevant plotting artefact with enough samples
    df['bin_midpoint'] = (df['bin'] * bin_size) + (bin_size // 2)

    fig, ax = plt.subplots(figsize=(12, 6))

    # seaborn computes a 95% CI for each value
    sns.lineplot(data=df, x='bin_midpoint', y='loss', ax=ax)

    # epoch boundaries (in bin units)
    epoch_boundaries = df.groupby('epoch')['step'].min().tolist()
    for boundary in epoch_boundaries[1:]:
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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


def forward_pass_batched(texts: List[str], tokenizer, model, device:str) -> torch.Tensor:
    """performing a forward pass of the model batching across texts"""
    encoding = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True  # pad to longest in batch, not max_length
    )

    input_ids = encoding["input_ids"].clone()
    attention_mask = encoding["attention_mask"]

    # apply masking
    masked_input_ids, labels = mask_tokens(input_ids, tokenizer)

    # check if there are any tokens to predict
    if (labels != -100).sum() == 0:
        return None

    masked_input_ids = masked_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(
        input_ids=masked_input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

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
@click.option('--batch-size', help='batch size for training', default=16, type=int)
def main(data, epochs, lr, accumulation_steps, output, batch_size):
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

    # storing loss data
    loss_store = [] # list of lists representing loss throughout epochs

    # main training loop
    model.train()
    epoch_pbar = tqdm(range(epochs), desc="Finetuning BERT...", dynamic_ncols=True)
    for epoch in epoch_pbar:
        total_loss = 0.0
        accumulated_loss = 0.0
        num_batches = 0
        samples_in_batch = 0

        loss_store.append([]) # list for this epoch's loss

        text_pbar = tqdm(
            range(0, len(data_lst), batch_size),
            desc=f"Epoch: {epoch+1}",
            leave=False,
            unit="batch",
            dynamic_ncols=True
        )
        for index in text_pbar:
            # performing forward pass and calculating loss
            texts = data_lst[index : index + batch_size]
            loss = forward_pass_batched(texts, tokenizer, model, device) 

            # skip if no valid masked tokens
            if loss is None or torch.isnan(loss):
                continue

            # incrementing the number of batches
            num_batches += 1

            # storing loss for later use
            loss_store[epoch].append(loss.item())  

            # scale loss by the number of accumulation steps to stabilise learning rate
            scaled_loss = loss / accumulation_steps

            # backward pass without step
            scaled_loss.backward()

            accumulated_loss += loss.item()
            total_loss += loss.item()
            samples_in_batch += 1

            # update steps after accumulating enough gradients
            if samples_in_batch == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()

                text_pbar.set_postfix({"batch_loss": f"{accumulated_loss / accumulation_steps:.4f}"})
                accumulated_loss = 0.0
                samples_in_batch = 0

        # handling the remaining samples
        if samples_in_batch > 0:
            optimizer.step()
            optimizer.zero_grad()


        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})


    # save model
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir) # saved for convenience, not changed during the training process
    print(f"Model saved to {output_dir}")

    # save loss over time
    loss_file = output_dir / "loss.txt"
    with open(loss_file, "w") as f:
        for epoch_idx, loss_lst in enumerate(loss_store):
            for loss_val in loss_lst:
                f.write(f"{epoch_idx},{loss_val}\n")

    # plot this loss
    loss_file_name = "loss.png"
    plot_loss(loss_store, output_dir / loss_file_name)
    print(f"Loss saved to {loss_file_name}")


if __name__ == "__main__":
    main()
