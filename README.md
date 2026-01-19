# Finetune BERT

Finetuning a custom BERT embedding model based on a local set of text data, through the `./main.py` file.

Finds the vector encoding length in the embedding space, and saves it to a numpy array through the `./find_length_vector.py` file.

## Quick Start

This project assumes that [uv package management](https://docs.astral.sh/uv/getting-started/installation/) is installed.

To install dependencies, run the following command.

``` bash
uv sync
```

### Data

This assumes that there is data in a `data.txt` file which is structured with text split across many lines. Performance will degrade considerably if most lines are over around 1500 words.

The use of batching, gradient accumulation, and binning in data visualisation all make the assumption that there are a reasonably large (~10,000 lines) number of samples to finetune on.

## Finetuning

To run the finetuning pipeline and produce a custom BERT model, run the following script.

```bash
uv run python main.py
```

This will download the BERT model and tokenizer from Huggingface, and start finetuning on the data in the data.txt file using MLM (masked language modeling).

By default, this saves the finetuned mode in the `finetuned-bert/` directory with the name `model.safetensors`.
A plot can be seen at `loss.png` which shows the declining loss of your model over time.

## Length Embedding

To find the direction in BERT embedding space (at the final layer) which corresponds to the length or your sample, run the following script.

```bash
uv run python find_length_vector.py
```

This will save a `length_direction.npy` file to the directory containing your finetuned model, which corresponds to a numpy array (of 784 dimensions if you used a standard BERT). This is the direction in the BERT embedding space most strongly associated with the length of the input sequence. 

When tested on a large dataset (~850k lines), the R squared value for the length prediction was `0.9366`, implying the most of the variance is captured by the simple linear model used.

You can utilise the `find_length_vector.remove_length_direction` function to return embeddings which should not contain any length information.
