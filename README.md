# Finetune BERT

Finetuning a custom BERT embedding model based on my set of text data, stored in `data.txt`.

Assumes that data is split across different lines. Note that performance will degrade if most lines are over around 1500 words (or 512 tokens).

The use of batching, gradient accumulation, and binning in data visualisation all make the assumption that there are a reasonably large (~10,000 lines) number of samples to finetune on.
