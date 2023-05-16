import argparse
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from e5 import handle_e5


def handle_instructor_evaluate(model_name: str, output_folder: Path):
    logging.info(f"handling instructor evaluation for model: {model_name}")
    from InstructorEmbedding import INSTRUCTOR

    model = INSTRUCTOR(f"hkunlp/{model_name}")
    sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments is the best thing that we can possibly do"
    instruction = "Represent the Science title:"
    model_input = [
        [instruction, sentence],
        [
            instruction,
            sentence + " and the cat jumped over the bag going forward into the night",
        ],
    ]
    existing_output = model.encode(model_input)
    model_input_tokens = model.tokenize(model_input)
    ort_sess = ort.InferenceSession(
        str(output_folder.joinpath(f"{model_name}.onnx")),
        providers=["CUDAExecutionProvider"],
    )
    result = ort_sess.run(
        [],
        {
            "INSTRUCTION": model_input_tokens["input_ids"].numpy(),
            "TEXT": model_input_tokens["attention_mask"].numpy(),
            "CTX": model_input_tokens["context_masks"].numpy(),
        },
    )
    assert existing_output.shape == result[-1].shape
    assert np.allclose(result[-1], existing_output, atol=1e-04), "the diff between the PyTORCH model and ONNX is > 1e-04"
    

def handle_instructor_export(model_name: str, output_folder: Path):
    """
    exports the instructor model to onnx using torch.onnx, the file is written to cwd
    """
    logging.info(f"handling instructor export for model: {model_name}")
    from InstructorEmbedding import INSTRUCTOR

    model = INSTRUCTOR(f"hkunlp/{model_name}")
    sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    instruction = "Represent the Science title:"
    # we choose two to demonstrate that we want to batch things out to the underlying
    # onnx model
    model_input = [
        [instruction, sentence],
        [instruction, sentence + "with the usage of high energy photons and cats"],
    ]
    input_tokens = model.tokenize(model_input)
    input_names = ["INSTRUCTION", "TEXT", "CTX"]
    torch.onnx.export(
        model,
        (
            {
                "input_ids": input_tokens["input_ids"],
                "attention_mask": input_tokens["attention_mask"],
                "context_masks": input_tokens["context_masks"],
            },
            "liam",
        ),
        f"{model_name}.onnx",
        export_params=True,
        input_names=input_names,
        dynamic_axes={
            "INSTRUCTION": {1: "input_ids_len", 0: "batch_size_len"},
            "TEXT": {1: "input_ids_len", 0: "batch_size_len"},
            "CTX": {0: "batch_size_len"},
        },
    )


def handle_instructor(action: str, model_name: str, output_folder: Path):
    if action == "export":
        handle_instructor_export(model_name, output_folder)
    if action == "evaluate":
        handle_instructor_evaluate(model_name, output_folder)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="TorchToONNX",
        description="transforms common pytorch modules into their onnxruntime equivalents",
        epilog="check out https://github.com/infrawhispers/anansi for addn details",
    )
    parser.add_argument("-o", "--output_folder", default=".onnx_output")
    parser.add_argument("-m", "--model_name", default="instructor-large")
    parser.add_argument("-a", "--action", default="export")
    args = parser.parse_args()
    if not Path(args.output_folder).is_dir():
        logging.info(
            f"the path at: {args.output_folder} does not exist, please select a folder that exists"
        )
        exit(1)

    if "instructor-" in args.model_name:
        handle_instructor(args.action, args.model_name, Path(args.output_folder))
    if "e5-" in args.model_name:
        handle_e5(args.action, args.model_name, Path(args.output_folder))
