from datasets import Dataset as DDataset

import pandas as pd
import os

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    # SentenceTransformerModelCardData,
    
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

# https://sbert.net/docs/sentence_transformer/training_overview.html

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_val = pd.read_csv("data/val.csv")

# https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss

# OPTION 1: current sentence is positive (i == j), all others are negatives (i != j)
# train_dataset = DDataset.from_dict(
#     {
#         "anchor": df_train["text_ru"].to_list(),
#         "positive": df_train["text_lez"].to_list(),
#     }
# )
# etc..


# OPTION 2: current sentence is positive, some others (yours) are negatives
# NOTE (https://huggingface.co/intfloat/multilingual-e5-large): Use "query: " prefix for symmetric tasks such as semantic similarity, bitext mining, paraphrase retrieval.
train_dataset = DDataset.from_dict(
    {
        "anchor": ["query: " + text for text in df_train["anchor"].to_list()],
        "positive": ["query: " + text for text in df_train["positive"].to_list()],
        "negative": ["query: " + text for text in df_train["negative"].to_list()]
    }
)

test_dataset = DDataset.from_dict(
    {
        "anchor": ["query: " + text for text in df_test["anchor"].to_list()],
        "positive": ["query: " + text for text in df_test["positive"].to_list()],
        "negative": ["query: " + text for text in df_test["negative"].to_list()]
    }
)

val_dataset = DDataset.from_dict(
    {
        "anchor": ["query: " + text for text in df_val["anchor"].to_list()],
        "positive": ["query: " + text for text in df_val["positive"].to_list()],
        "negative": ["query: " + text for text in df_val["negative"].to_list()]
    }
)

def main():
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir="models/multilingual-e5-large-tuned",
        num_train_epochs=10,
        per_device_train_batch_size=14,
        per_device_eval_batch_size=14,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="steps",
        eval_steps=4432,
        save_strategy="steps",
        save_steps=4432,
        save_total_limit=2,
        logging_steps=100,
        run_name="multilingual-e5-large-tuned",  # Will be used in W&B if `wandb` is installed
        save_on_each_node=True,
        dataloader_drop_last=False,
        load_best_model_at_end=True,
        logging_dir="./tb_logs",
    )


    val_evaluator = TripletEvaluator(
        anchors  = val_dataset["anchor"],
        positives = val_dataset["positive"],
        negatives = val_dataset["negative"],
        name="e5-val",
    )
    # val_evaluator(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=val_evaluator,
    )
    trainer.train()

    test_evaluator = TripletEvaluator(
        anchors = test_dataset["anchor"],
        positives = test_dataset["positive"],
        negatives = test_dataset["negative"],
        name="e5-test",
    )
    test_evaluator(model)

    model.save_pretrained("models/multilingual-e5-large-tuned/final")


if __name__ == "__main__":
    main()
