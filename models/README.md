Place optional line-guide ONNX models in this directory.

Expected default name:
- `line_guide.onnx`

A lightweight reference model can be generated locally:

```powershell
python tools\create_line_guide_sobel_onnx.py models\line_guide.onnx
python tools\line_guide_benchmark.py --model models\line_guide.onnx --output docs\line_guide_measurements.json
```

This creates a small Sobel-based ONNX graph. It is useful as a local default or as
a sanity check for ONNX Runtime integration. A learned line-art model can replace
the file when better quality is needed.

Supported model input:
- 3-channel RGB or 1-channel luma input.
- NCHW, NHWC, CHW, or HWC shape.
- float input is fed as 0.0 to 1.0 RGB.
- uint8 input is fed as 0 to 255 RGB.

Supported model output:
- HxW, 1xHxW, HxWx1, 1x1xHxW, or 1xHxWx1 confidence map.
- 2-channel output uses channel 1 as the line class.
- 3 or 4-channel output uses the strongest channel response.
- Values are normalized into a 0.0 to 1.0 confidence map, not a binary image.

Search order:
- `FD6_MODELS_DIR`
- packaged executable directory `models`
- current working directory `models`
- repository `models`

Packaged builds look for external models next to the executable under `models`.
Source runs also check this directory. `FD6_MODELS_DIR` can point to another model directory.
When a relative model path is typed into the settings screen, FD6 also checks the
source image directory, the current working directory, and the repository root
before the default model directories.

The normal package does not include ONNX Runtime. Use an ONNX-enabled build or install
the optional source dependency before expecting `line_guide.onnx` to run.
On Windows, `onnxruntime-directml` is supported through `DmlExecutionProvider`
when CUDA provider dependencies are not available.

Optional per-model settings may be stored next to the model file as `line_guide.json`
or `line_guide.onnx` sidecar JSON. Supported keys:

- `input.colorOrder`: `rgb`, `bgr`, or `luma`
- `input.valueRange`: `0_1`, `0_255`, or `minus1_1`
- `input.mean` / `input.std`
- `output.index`
- `output.channel`
- `output.activation`: `auto`, `sigmoid`, `softmax`, or `none`
- `output.invert`
- `maxResolution`
