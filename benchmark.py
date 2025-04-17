from transformers import AutoTokenizer, AutoModelForCausalLM
import onnxruntime as ort
import torch
import numpy as np
import time

# Use a small causal LM that still allocates large buffers
model_name = "sshleifer/tiny-gpt2"  # swap to bigger if needed

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Hello, my name is", return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"],),
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence"}},
    opset_version=13
)

# Load into ONNX Runtime
session = ort.InferenceSession("model.onnx")

# Run many inference steps with different lengths
for i in range(100):
    seq_len = 128 + (i % 64)
    batch = np.random.randint(0, 10000, (4, seq_len)).astype(np.int64)
    ort_inputs = {"input_ids": batch}
    outputs = session.run(None, ort_inputs)
    time.sleep(0.01)
