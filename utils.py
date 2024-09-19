import pandas as pd
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_data(df_path: str = "data/bible.csv", df_with_negatives_path: str = "data/bible_with_neg.csv", save_dir: str = "data"):
    df = pd.read_csv(df_path)
    df_with_negatives = pd.read_csv(df_with_negatives_path)

    anchor, positive, negative = [], [], []
    for _, item in tqdm(df_with_negatives.iterrows(), total = len(df)):
        src_text = item["text_ru"] # anchor
        tgt_text = item["text_lez"] # positive translation
        neg_texts = [df.iloc[int(neg_idx)]["text_lez"] for neg_idx in ast.literal_eval(item["negative_id"])] # negative translations

        for neg_text in neg_texts:
            anchor.append(src_text)
            positive.append(tgt_text)
            negative.append(neg_text)

    # Шаг 1: Разделить данные на train и temp (сначала отделить 90% для train)
    anchor_train, anchor_temp, positive_train, positive_temp, negative_train, negative_temp = train_test_split(
        anchor, positive, negative, test_size=0.1, random_state=42
    )

    # Шаг 2: Разделить temp данные на val и test (оставшиеся 10% делим пополам по 5% на каждый)
    anchor_val, anchor_test, positive_val, positive_test, negative_val, negative_test = train_test_split(
        anchor_temp, positive_temp, negative_temp, test_size=0.5, random_state=42
    )


    df_train = pd.DataFrame({
        'anchor': anchor_train,
        'positive': positive_train,
        'negative': negative_train
    })

    df_val = pd.DataFrame({
        'anchor': anchor_val,
        'positive': positive_val,
        'negative': negative_val
    })

    df_test = pd.DataFrame({
        'anchor': anchor_test,
        'positive': positive_test,
        'negative': negative_test
    })

    save_dir = Path(save_dir)

    df_train.to_csv(save_dir / 'train.csv', index=False)
    df_val.to_csv(save_dir / 'val.csv', index=False)
    df_test.to_csv(save_dir / 'test.csv', index=False)


if __name__ == "__main__":
    split_data()