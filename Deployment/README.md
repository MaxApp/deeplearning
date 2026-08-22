# Model Optimize and Deployment

While a model is well trained, the next progress is to use it in practical scenarios. Your model not only able to run on your own machine but also able to run cross different platforms by convert it to `ONNX` format.

## Save checkpoints during traning

Training is a time consuming progress, you won't want to get from the start everytime you want to modify code or encounter an issue. We could use pytorch save methods to store model states and training parameters in order to resume them at any point.

```python
def save_checkpoint(epoch, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # save scheduler state if exists
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        # save random seed for reproducible
        'rng_state': torch.get_rng_state(),
    }
    torch.save(checkpoint, path)
```

## Save Model

After training finished, we could save all the parameters to disk. Certainly reload them is trivial.

```python
torch.save(model.state_dict(), "my_model.pt")
model.load_state_dict(torch.load("my_model.pt", weights_only=True))
```

## Lightning and MLFlow tool

#### lightning_and_mlflow.py

We could refine our code with `Lightning` framework to seperate training, testing, validating progress and code could be easily written and managed.

`MLFlow` could help us logging, managing model state while training with various parameters. It could record those metrics to let you make comparison between trials.

## ONNX Deployment

Make your model run cross platforms by `onnx runtime`. Save and convert your model to `.onnx` file, then you can even use it as `TensorFlow` model.

```python
model.eval()
# Export to ONNX format
torch.onnx.export(
         model, # PyTorch model
         dummy_input, # Input tensor
         "model.onnx", # Output file name
         export_params = True,
         opset_version = 11, # ONNX version
         do_constant_folding = True, # Optimize constant folding
         input_names = ['input'], # Name inputs
         output_names = ['output'], # Name outputs
         dynamic_axes = { 
            # Support variable batch size
             'input': {0: 'batch_size'}, 
             'output': {0: 'batch_size'}
             }
)
```

Inference in ONNX runtime environment.

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})
prediction = outputs[0]
```

## Model Compression

Optimize your model to use less space, less memory and run faster. There're two kinds of methods. `Pruning` and `Quantization`.

#### Pruning

* Unstructured pruning
* Structured pruning

#### Quantization

