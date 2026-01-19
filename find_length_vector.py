#!/usr/bin/env python
"""finding the vector corresponding to the length of the underlying text"""

import click
from pathlib import Path
from transformers import AutoTokenizer, BertModel
import torch
from typing import List, Tuple
# import safetensors
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_device():
    """finding the device available"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def load_data(data_path: Path) -> List:
    """load the data in the data file into a python string"""
    if not data_path.exists():
        raise ValueError(f"Provided data path {data_path} does not exist")

    data_lst = []
    with open(data_path, "r") as f:
        data_lst = [line.strip() for line in f if line.strip()] # concatenating all of the text 
    
    return data_lst    


def load_model_to_torch(model: str, device: str) -> torch.Tensor:
    """loading the model from a safetensors file"""
    model_path = Path(model)

    loaded_model = BertModel.from_pretrained(
        model_path.parent,  # Directory containing the safetensors file
        device_map=device
    )

    return loaded_model


def find_save_dir(model_path: Path) -> Path:
    """find the appropriate directroy to save the length vector to"""
    model_path = Path(model_path)
    if model_path.parent:
        directory = model_path.parent
    else:
        directory = Path(".")
    return directory


def fit_probe_train(embeddings_full: np.array, lengths_full: np.array) -> Tuple[int]:
    """fit the linear length probe on a train set and evaluate on a test set"""
    emb_train, emb_test, len_train, len_test = train_test_split(
        embeddings_full, lengths_full, test_size=0.2, random_state=0
    )

    reg = LinearRegression()
    reg.fit(emb_train, len_train)

    r_squared_train = reg.score(emb_train, len_train)
    r_squared_test = reg.score(emb_test, len_test)

    return r_squared_train, r_squared_test


def find_length_direction(lengths: np.array, embeddings: np.array) -> np.array:
    """get the length direction as a linear dir"""
    scaler = StandardScaler()
    lengths_scaled = scaler.fit_transform(lengths.reshape(-1, 1)).flatten()

    reg = LinearRegression()
    reg.fit(embeddings, lengths_scaled)

    length_direction = reg.coef_

    # normalising to a unit vector
    length_direction = length_direction / np.linalg.norm(length_direction)
    
    return length_direction


def embed_text(texts: List[str], tokenizer: torch.Tensor, model: BertModel, device: str = 'cpu') -> np.array:
    """Generate embedding for a list of texts"""
    # Tokenize and prepare input
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling over token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().numpy()


def find_lengths(texts: List[str]) -> np.array:
    """find the lengths of text inputs"""
    lengths = []
    for text in texts:
        lengths.append(len(text))
    return np.array(lengths)


def remove_length_direction(embeddings_full: np.array, length_direction: np.array) -> np.array:
    """adjusting the embeddings to remove the length direction"""
    projections = embeddings_full @ length_direction
    adjusted = embeddings_full - np.outer(projections, length_direction)
    norms = np.linalg.norm(adjusted, axis=1, keepdims=True)
    return adjusted / norms



@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('--data', help='data file', type=str, default="data.txt")
@click.option('--model-path', help='model to analyse', default="finetuned-bert-remote/model.safetensors")
def main(data, model_path):
    """main script logic"""

    bert_model_name = "google-bert/bert-base-uncased" # default BERT for tokenizer

    # device
    device = get_device()

    # load the relevant data
    data_lst = load_data(Path(data))

    # load model and tokenizer
    model_torch = load_model_to_torch(model_path, device) # actually an instance of transformers.models.bert.modeling_bert.BertModel
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # compute embeddings
    batch_size = 16
    embed_pbar = tqdm(range(0, len(data_lst), batch_size), desc='Calculating embeddings for text', dynamic_ncols=True)
    embeddings_full = None
    lengths_full = None
    for idx in embed_pbar:
        texts = data_lst[idx : idx + batch_size]
        embeddings = embed_text(texts, tokenizer, model_torch, device)
        lengths = find_lengths(texts)

        # updating array of all embeddings
        if embeddings_full is None:
            embeddings_full = embeddings
        else:
            embeddings_full = np.concatenate((embeddings_full, embeddings), axis=0)

        # updating array of all lengths
        if lengths_full is None:
            lengths_full = lengths
        else:
            lengths_full = np.concatenate((lengths_full, lengths), axis=0)

    length_direction = find_length_direction(lengths_full, embeddings_full)

    # find directory to save length_direction to
    save_dir = find_save_dir(model_path)
    length_direction_file = save_dir / "length_direction.npy"
    print(f"Saving length embedding direction to {length_direction_file}")
    np.save(length_direction_file, length_direction)

    r_squared = LinearRegression().fit(embeddings_full, lengths_full).score(embeddings_full, lengths_full)
    print(f"R squared of length prediction: {r_squared:.4f}")
    print("")

    r_squared_train, r_squared_test = fit_probe_train(embeddings_full, lengths_full)
    print(f"R squared of train set for length prediction: {r_squared_train:.4f}")
    print(f"R squared of test set for length prediction: {r_squared_test:.4f}")

    # to adjust the embeddings
    # corrected_embeddings = remove_length_direction(embeddings_full, length_direction)
    # print(corrected_embeddings)


if __name__ == "__main__":
    main()
