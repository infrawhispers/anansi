import numpy as np
import onnxruntime as ort
import torch
from InstructorEmbedding import INSTRUCTOR


def export_model_to_onnx(output_name: str):
    """
    exports the instructor model to onnx using torch.onnx, the file is written to cwd
    """
    model = INSTRUCTOR("hkunlp/instructor-large")
    sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    instruction = "Represent the Science title:"
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
        output_name,
        export_params=True,
        input_names=input_names,
        dynamic_axes={
            "INSTRUCTION": {1: "input_ids_len", 0: "batch_size_len"},
            "TEXT": {1: "input_ids_len", 0: "batch_size_len"},
            "CTX": {0: "batch_size_len"},
        },
    )


def evaluate_onnx_model(model_name: str):
    model = INSTRUCTOR("hkunlp/instructor-large")
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
    ort_sess = ort.InferenceSession(model_name)
    result = ort_sess.run(
        [],
        {
            "INSTRUCTION": model_input_tokens["input_ids"].numpy(),
            "TEXT": model_input_tokens["attention_mask"].numpy(),
            "CONTEXT_MASK": model_input_tokens["context_masks"].numpy(),
        },
    )
    assert existing_output.shape == result[-1].shape
    num_mismatches = 0
    # TODO(infrawhispers) - there is away to do this with numpy right??
    for i in range(0, existing_output.shape[0]):
        for j in range(0, existing_output.shape[1]):
            diff = abs(existing_output[i][j] - result[-1][i][j])
            if diff > 0.01:
                num_mismatches += 1
    print(f"num_mismatches: {num_mismatches}")


if __name__ == "__main__":
    onnx_model_name = "instructor_w_ctx_mask_v2.onnx"
    export_model_to_onnx(onnx_model_name)
    # evaluate_onnx_model(onnx_model_name)
