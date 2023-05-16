import csv
import json
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from numpy import dot
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


def convert_embedding_to_cache():
    # precomputed_embedding_cache_path = (
    #     "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
    # )
    embedding_path = ".cache/snli_embedding_cache.pkl"
    embedding_cache = pd.read_pickle(embedding_path)
    with open(".cache/snli_embedding.csv", "w", newline="") as f:
        f.write("record\n")
        for k, v in embedding_cache.items():
            d = {"sentence": k[0], "embeddings": v}
            row = json.dumps(d)
            f.write(row)
            f.write("\n")


def embeddings_by_sentence() -> dict:
    mapping = {}
    id_counter = 0
    num_lines_read = 0
    with open("../.cache/snli_embedding.csv") as f:
        for line in f:
            if num_lines_read == 0:
                num_lines_read += 1
                continue
            data = json.loads(line)
            mapping[data["sentence"]] = (data["embeddings"], id_counter)
            id_counter += 1
    return mapping


def prepare_optimizing_lib(mapping) -> List[Tuple]:
    num_pairs_to_consider = 1000
    num_positive = 1000
    num_negaitve = 1000
    num_lines_read = 0
    embeding_pairs = []
    with open("../lib/retrieval/.data/snli_1.0/snli_1.0_train.txt") as file:
        for line in file:
            if num_positive == 0 and num_negaitve == 0:
                break

            if num_lines_read != 0:
                parts = line.split("\t")
                label = parts[0]
                sentence_1 = parts[5]
                sentence_2 = parts[6]
                if label != "entailment" and label != "contradiction":
                    continue
                item_id_1 = mapping.get(sentence_1)
                item_id_2 = mapping.get(sentence_2)
                if item_id_1 is None or item_id_2 is None:
                    continue
                item_labels = 1.0
                if label == "entailment":
                    item_labels = 1.0
                else:
                    item_labels = -1.0
                if item_labels == 1:
                    if num_positive == 0:
                        continue
                    else:
                        num_positive -= 1
                if item_labels == -1:
                    if num_negaitve == 0:
                        continue
                    else:
                        num_negaitve -= 1

                embeding_pairs.append(
                    {
                        "text_1": sentence_1,
                        "text_1_embedding": item_id_1[0],
                        "text_2": sentence_2,
                        "text_2_embedding": item_id_2[0],
                        "label": item_labels,
                    }
                )
            num_lines_read += 1
    return embeding_pairs


def dataframe_of_negatives(dataframe_of_positives: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of negative pairs made by combining elements of positive pairs."""
    texts = set(dataframe_of_positives["text_1"].values) | set(
        dataframe_of_positives["text_2"].values
    )
    all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
    positive_pairs = set(
        tuple(text_pair)
        for text_pair in dataframe_of_positives[["text_1", "text_2"]].values
    )
    negative_pairs = all_pairs - positive_pairs
    df_of_negatives = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    df_of_negatives["label"] = -1
    return df_of_negatives


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
    accuracies = []
    for threshold_thousandths in range(-1000, 1000, 1):
        threshold = threshold_thousandths / 1000
        total = 0
        correct = 0
        for cs, ls in zip(cosine_similarity, labeled_similarity):
            total += 1
            if cs > threshold:
                prediction = 1
            else:
                prediction = -1
            if prediction == ls:
                correct += 1
        accuracy = correct / total
        accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
    return a, standard_error


def embedding_multiplied_by_matrix(
    embedding: List[float], matrix: torch.tensor
) -> np.array:
    embedding_tensor = torch.tensor(embedding).float()
    modified_embedding = embedding_tensor @ matrix
    modified_embedding = modified_embedding.detach().numpy()
    return modified_embedding


# compute custom embeddings and new cosine similarities
def apply_matrix_to_embeddings_dataframe(matrix: torch.tensor, df: pd.DataFrame):
    for column in ["text_1_embedding", "text_2_embedding"]:
        df[f"{column}_custom"] = df[column].apply(
            lambda x: embedding_multiplied_by_matrix(x, matrix)
        )
    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(
            row["text_1_embedding_custom"], row["text_2_embedding_custom"]
        ),
        axis=1,
    )


def optimize_matrix(
    df,
    modified_embedding_length: int = 2048,  # in my brief experimentation, bigger was better (2048 is length of babbage encoding)
    batch_size: int = 100,
    max_epochs: int = 100,
    learning_rate: float = 100.0,  # seemed to work best when similar to batch size - feel free to try a range of values
    dropout_fraction: float = 0.0,  # in my testing, dropout helped by a couple percentage points (definitely not necessary)
    print_progress: bool = True,
    save_results: bool = True,
) -> torch.tensor:
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    """Return matrix optimized to minimize loss on training data."""
    run_id = random.randint(0, 2**31 - 1)  # (range is arbitrary)

    # convert from dataframe to torch tensors
    # e is for embedding, s for similarity label
    def tensors_from_dataframe(
        df: pd.DataFrame,
        embedding_column_1: str,
        embedding_column_2: str,
        similarity_label_column: str,
    ) -> Tuple[torch.tensor]:
        e1 = np.stack(np.array(df[embedding_column_1].values))
        e2 = np.stack(np.array(df[embedding_column_2].values))
        s = np.stack(np.array(df[similarity_label_column].astype("float").values))

        e1 = torch.from_numpy(e1).float()
        e2 = torch.from_numpy(e2).float()
        s = torch.from_numpy(s).float()

        return e1, e2, s

    e1_train, e2_train, s_train = tensors_from_dataframe(
        df[df["dataset"] == "train"], "text_1_embedding", "text_2_embedding", "label"
    )
    e1_test, e2_test, s_test = tensors_from_dataframe(
        df[df["dataset"] == "test"], "text_1_embedding", "text_2_embedding", "label"
    )

    # create dataset and loader
    dataset = torch.utils.data.TensorDataset(e1_train, e2_train, s_train)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # define model (similarity of projected embeddings)
    def model(embedding_1, embedding_2, matrix, dropout_fraction=dropout_fraction):
        e1 = torch.nn.functional.dropout(embedding_1, p=dropout_fraction)
        e2 = torch.nn.functional.dropout(embedding_2, p=dropout_fraction)
        modified_embedding_1 = e1 @ matrix  # @ is matrix multiplication
        modified_embedding_2 = e2 @ matrix
        similarity = torch.nn.functional.cosine_similarity(
            modified_embedding_1, modified_embedding_2
        )
        return similarity

    # define loss function to minimize
    def mse_loss(predictions, targets):
        difference = predictions - targets
        mse_loss = torch.sum(difference * difference) / difference.numel()
        # print(f"[mse_loss] num_elements: {difference.shape}")
        return mse_loss

    # initialize projection matrix
    embedding_length = len(df["text_1_embedding"].values[0])
    matrix = torch.rand(embedding_length, modified_embedding_length, requires_grad=True)

    epochs, types, losses, accuracies, matrices = [], [], [], [], []
    for epoch in range(1, 1 + max_epochs):
        # iterate through training dataloader
        for a, b, actual_similarity in train_loader:
            # generate prediction
            predicted_similarity = model(a, b, matrix)
            # get loss and perform backpropagation
            loss = mse_loss(predicted_similarity, actual_similarity)
            loss.backward()
            # update the weights
            with torch.no_grad():
                matrix -= matrix.grad * learning_rate
                # set gradients to zero
                matrix.grad.zero_()
        # calculate test loss
        test_predictions = model(e1_test, e2_test, matrix)
        test_loss = mse_loss(test_predictions, s_test)
        print(f"test_loss: {test_loss}")

        # compute custom embeddings and new cosine similarities
        apply_matrix_to_embeddings_dataframe(matrix, df)

        # calculate test accuracy
        for dataset in ["test"]:
            data = df[df["dataset"] == dataset]
            a, se = accuracy_and_se(data["cosine_similarity_custom"], data["label"])

            # record results of each epoch
            epochs.append(epoch)
            types.append(dataset)
            losses.append(loss.item() if dataset == "train" else test_loss.item())
            accuracies.append(a)
            matrices.append(matrix.detach().numpy())

            # optionally print accuracies
            if print_progress is True:
                print(
                    f"Epoch {epoch}/{max_epochs}: {dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}"
                )


def run_through_example():
    mapping = embeddings_by_sentence()

    def get_embedding_with_cache(text: str):
        return mapping[text][0]

    embedding_pairs = prepare_optimizing_lib(mapping)
    df = pd.DataFrame(embedding_pairs)
    # print(df)
    test_fraction = 0.5  # 0.5 is fairly arbitrary
    random_seed = 123  # random seed is arbitrary, but is helpful in reproducibility
    # train_df, test_df = train_test_split(
    #     df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
    # )
    train_df = df.iloc[:900, :]
    test_df = df.iloc[900:, :]
    train_df.loc[:, "dataset"] = "train"
    test_df.loc[:, "dataset"] = "test"

    negatives_per_positive = (
        1  # it will work at higher values too, but more data will be slower
    )
    # generate negatives for training dataset
    train_df_negatives = dataframe_of_negatives(train_df)
    train_df_negatives["dataset"] = "train"
    # generate negatives for test dataset
    test_df_negatives = dataframe_of_negatives(test_df)
    test_df_negatives["dataset"] = "test"
    # sample negatives and combine with positives
    train_df = pd.concat(
        [
            train_df,
            # train_df_negatives.sample(
            #     n=len(train_df) * negatives_per_positive, random_state=random_seed
            # ),
        ]
    )
    test_df = pd.concat(
        [
            test_df,
            # test_df_negatives.sample(
            #     n=len(test_df) * negatives_per_positive, random_state=random_seed
            # ),
        ]
    )

    df = pd.concat([train_df, test_df])
    # print(df)

    for column in ["text_1", "text_2"]:
        df[f"{column}_embedding"] = df[column].apply(get_embedding_with_cache)

    df["cosine_similarity"] = df.apply(
        lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
        axis=1,
    )
    # print(df)

    for dataset in ["train", "test"]:
        data = df[df["dataset"] == dataset]
        a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
        print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")

    results = []
    max_epochs = 30
    dropout_fraction = 0.2
    for batch_size, learning_rate in [(10, 10), (100, 100)]:
        result = optimize_matrix(
            df,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            dropout_fraction=dropout_fraction,
            save_results=False,
        )
        results.append(result)


if __name__ == "__main__":
    # convert_embedding_to_cache()
    run_through_example()
