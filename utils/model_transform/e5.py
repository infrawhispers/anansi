from pathlib import Path
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # item = ~attention_mask[..., None].bool()
    # print(f"atten_mask_shape : {item.shape}")
    # print(f"last_hidden_state_shape: {last_hidden_states.shape}")
    # print(item)
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # print(f"last_hidden: {last_hidden.sum(dim=1)}")
    # print(f"attention_mask_sum: {attention_mask.sum(dim=1)}")
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



def handle_e5_export(model_name: str, output_folder: Path):
    tokenizer = AutoTokenizer.from_pretrained(f"intfloat/{model_name}")
    model = AutoModel.from_pretrained(f"intfloat/{model_name}")
    input_texts = [
        "query: how much protein should a female eat",
        "query: summit define",
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    # print(batch_dict)
    input_names = ["INPUT_IDS", "TOKEN_TYPE_IDS", "ATTENTION_MASK"]
    torch.onnx.export(
        model,
        (
            {
                "input_ids": batch_dict["input_ids"],
                "token_type_ids": batch_dict["token_type_ids"],
                "attention_mask": batch_dict["attention_mask"],
            },
        ),
        output_folder.joinpath(f"{model_name}.onnx"),
        export_params=True,
        input_names=input_names,
        dynamic_axes={
            "INPUT_IDS": {1: "input_ids_len", 0: "batch_size_len"},
            "TOKEN_TYPE_IDS": {1: "input_ids_len", 0: "batch_size_len"},
            "ATTENTION_MASK": {1: "input_ids_len", 0: "batch_size_len"},
        },
    )

def evaluate_e5_export(model_name: str, output_folder: Path):
    tokenizer = AutoTokenizer.from_pretrained(f"intfloat/{model_name}")
    model = AutoModel.from_pretrained(f"intfloat/{model_name}")
    input_texts = [
        "query: how much protein should a female eat",
        "query: summit define",
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**batch_dict)
   
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    print(embeddings)
    # print(batch_dict)

    ort_sess = ort.InferenceSession(
        ".onnx_output/onnx_e5_large/model.onnx",
        # str(output_folder.joinpath(f"{model_name}.onnx")),
        providers=["CPUExecutionProvider"],
    )
    # print(batch_dict["input_ids"].numpy())
    # print(batch_dict["attention_mask"].numpy())
    result = ort_sess.run(
        [],
        {
            "input_ids": batch_dict["input_ids"].numpy(),
            "token_type_ids": batch_dict["token_type_ids"].numpy(),
            "attention_mask": batch_dict["attention_mask"].numpy(),
        },
    )
    # print(result[0].shape)
    # print(result[0])
    embeddings_onnx = average_pool(torch.from_numpy(result[0]), batch_dict['attention_mask'])
    print(embeddings_onnx)



def handle_e5(action: str, model_name: str, output_folder: Path):
    if action == "export":
        handle_e5_export(model_name, output_folder)
    if action == "evaluate":
        evaluate_e5_export(model_name, output_folder)
