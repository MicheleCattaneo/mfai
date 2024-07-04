# MFAI: Météo-France's AI Python package

![Unit Tests](https://github.com/meteofrance/mfai/actions/workflows/tests.yml/badge.svg)

**MFAI** is a Python package that provides the following features:
- A variety of PyTorch Neural Network architectures (CNN, Vision Transformers, ...) adapted to our needs, tested on our projects and datasets. For each architecture, we provide the reference to the original paper and source code if applicable and also the modifications we made.
- Per architecture schema validated settings using [dataclasses-json](https://github.com/lidatong/dataclasses-json)
- A NamedTensor class to handle multi-dimensional data with named dimensions and named features

# Table of contents

- [Neural Network Architectures](#neural-network-architectures)
    - deeplabv3+
    - halfunet
    - unet
    - segformer
    - swinunetr
    - unetr++
- [NamedTensors](#namedtensors)
- [Installation](#installation)
- [Usage](#usage)
    - [Instanciate a model](#instanciate-a-model)
    - [Export to onnx](#export-to-onnx)
    - [NamedTensors](#namedtensors-example)
- [Running tests](#tests)

# Neural Network Architectures

Each model we provide is a subclass of [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and can be used in a PyTorch training loop. It has three **critical** class attributes:
- **settings_kls**: a class that defines the settings of the model (number of filters, kernel size, ...). It is used to instanciate the model with a specific configuration.
- **onnx_supported**: a boolean that indicates if the model can be exported to onnx. Our CI validates that the model can be exported to onnx and reloaded for inference.
- **input_spatial_dims**: a tuple that describes the spatial dimensions of the input tensor supported by the model. A model that supports 2D spatial data will have **(2,)** as value. A model that supports 2d or 3d spatial data will have **(2, 3)** as value.

The Python interface contract for our model is enforced using [Python ABC](https://docs.python.org/3/library/abc.html) and in our case [ModelABC](mfai/torch/models/base.py#L1) class.

```python
@dataclass_json
@dataclass(slots=True)
class HalfUNetSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"
    absolute_pos_embed: bool = False

class HalfUNet(ModelABC, nn.Module):
    settings_kls = HalfUNetSettings
    onnx_supported = True
    input_spatial_dims = (2,)
```

Currently we support the following neural network architectures:

| Model  | Research Paper  | Input Shape    | ONNX exportable ? | Notes | Use-Cases at MF | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
| [deeplabv3+](mfai/torch/models/deeplabv3.py#L1) | [arxiv link](https://arxiv.org/abs/1802.02611) | (Batch, features, Height, Width)    | Yes | As a very large receptive field versus U-Net, Half-Unet, ... | Front Detection, Nowcasting | Theo Tournier / Frank Guibert |
| [halfunet](mfai/torch/models/half_unet.py#L1) | [researchgate link](https://www.researchgate.net/publication/361186968_Half-UNet_A_Simplified_U-Net_Architecture_for_Medical_Image_Segmentation) | (Batch, features, Height, Width)    | Yes | In prod/oper on [Espresso](https://www.mdpi.com/2674-0494/2/4/25) V2 with 128 filters and standard conv blocks instead of ghost | Satellite channels to rain estimation |  Frank Guibert |
| [unet](mfai/torch/models/unet.py#L1) | [arxiv link](https://arxiv.org/pdf/1505.04597.pdf) | (Batch, features, Height, Width)    | Yes | Vanilla U-Net | Radar image cleaning |  Theo Tournier / Frank Guibert |
| [segformer](mfai/torch/models/segformer.py#L1) | [arxiv link](https://arxiv.org/abs/2105.15203)   | (Batch, features, Height, Width) | Yes | On par with u-net like on Deepsyg (MF internal), added an upsampling stage. Adapted from [Lucidrains' github](https://github.com/lucidrains/segformer-pytorch) | Segmentation tasks | Frank Guibert |
| [swinunetr](mfai/torch/models/swinunetr.py#L1) | [arxiv link](https://arxiv.org/abs/2201.01266)   | (Batch, features, Height, Width)  | Yes | 2D Swin  Unet transformer (Pangu and archweather uses customised 3D versions of Swin Transformers). Plugged in from [MONAI](https://github.com/Project-MONAI/MONAI/). The decoders have been modified to use Bilinear2D + Conv2d instead of Conv2dTranspose to remove artefacts/checkerboard effects | Segmentation tasks  |  Frank Guibert |
| [unetr++](mfai/torch/models/unetrpp.py#L1) | [arxiv link](https://arxiv.org/abs/2212.04497)  | (Batch, features, Height, Width) or  (Batch, features, Height, Width, Depth) | Yes | Vision transformer with a reduced GFLOPS footprint adapted from [author's github](https://github.com/Amshaker/unetr_plus_plus). Modified to work both with 2d and 3d inputs | Front Detection | Frank Guibert |

# NamedTensors

PyTorch provides an experimental feature called [**named tensors**](https://pytorch.org/docs/stable/named_tensor.html), at this time it is subject to change so we don't use it. That's why we provide our own implementation.

NamedTensors are a way to give names to dimensions of tensors and to keep track of the names of the physical/weather parameters along the features dimension.

The [**NamedTensor**](../py4cast/datasets/base.py#L38) class is a wrapper around a PyTorch tensor, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...). See the implementation [here](../py4cast/datasets/base.py#L38) and usage for plots [here](../py4cast/observer.py)


# Installation

```bash
pip install mfai
```

# Usage

## Instanciate a model

Our [unit tests](tests/test_models.py#L39) provides an example of how to use the models in a PyTorch training loop. Our models are instanciated with 2 mandatory positional arguments: **in_channels** and **out_channels** respectively the number of input and output channels/features of the model. A third **input_shape** parameter is either mandatory (**UNETR++** or **HalfUNet with absolute pos embedding**) or optional for the other models. It describes the shape of the input tensor along its spatial dimensions.

The last parameter is an instance of the model's settings class and is a keyword argument with a default value set to the default settings.



Here is an example of how to instanciate the UNet model with a 3 channels input (like an RGB image) and 1 channel output with its default settings:

```python
from mfai.torch.models import UNet
unet = UNet(in_channels=3, out_channels=1)
```

**_FEATURE:_** Once instanciated, the model (subclass of **nn.Module**) can be used like any standard [PyTorch model](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html).

In order to instanciate a HalfUNet model with a 2 channels inputs, 2 channels outputs and a custom settings (128 filters, ghost module):

```python
from mfai.torch.models import HalfUNet
halfunet = HalfUNet(in_channels=2, out_channels=2, settings=HalfUNet.settings_kls(num_filters=128, use_ghost=True))
```

Finally, to instanciate a model with the mandatory **input_shape** parameter, here is an example with the UNETR++ model working on 2d spatial data (256x256) with 3 channels input and 1 channel output:

```python
from mfai.torch.models import UNETRPP
unetrpp = UNETRPP(in_channels=3, out_channels=1, input_shape=(256, 256))
```

**_FEATURE:_**  Each model has its settings class available under the **settings_kls** attribute.

You can use the **load_from_settings_file** function to instanciate a model with its settings from a json file:

```python
from pathlib import Path
from mfai.torch.models import load_from_settings_file
model = load_from_settings_file(
    "HalfUNet",
    2,
    2,
    Path(".") / "mfai" / "config" / "models" / "halfunet128.json",
)
```

**_FEATURE:_**  Use the **load_from_settings_file** to have the strictest validation of the settings.

## Export to onnx

Our tests [illustrate how to export and later reload a model to/from onnx](tests/test_models.py#L91). Here is an example of how to export a model to onnx:

```python
from mfai.torch import export_to_onnx, onnx_load_and_infer

# Export the model to onnx assuming we have just trained a 'model'
export_to_onnx(model, "model.onnx")

# Load the onnx model and infer
output_tensor = onnx_load_and_infer("model.onnx", input_tensor)
```

Check the code of [onnx_load_and_infer](mfai/torch/__init__.py#L35) if you would like to load the model once and make multiple inferences.

## NamedTensors example

We use **NamedTensor** instances to keep the link between our torch tensors and our physical/weather feature names (for plotting, for specific losses weights on given features, ...).

Some examples of NamedTensors usage, here for gridded data on a 256x256 grid:

```python

tensor = torch.rand(4, 256, 256, 3)

nt = NamedTensor(
    tensor,
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

print(nt.dim_size("lat"))
# 256

nt2 = NamedTensor(
    torch.rand(4, 256, 256, 1),
    names=["batch", "lat", "lon", "features"],
    feature_names=["q"],
)

# concat along the features dimension
nt3 = nt | nt2

# index by feature name
nt3["u"]

# Create a new NamedTensor with the same names but different data (useful for autoregressive models)
nt4 = NamedTensor.new_like(torch.rand(4, 256, 256, 4), nt3)

# Flatten in place the lat and lon dimensions and rename the new dim to 'ngrid'
# this is typically to feed our gridded data to GNNs
nt3.flatten_("ngrid", 1, 2)

# str representation of the NamedTensor yields useful statistics
>>> print(nt)
--- NamedTensor ---
Names: ['batch', 'lat', 'lon', 'features']
Tensor Shape: torch.Size([4, 256, 256, 3]))
Features:
┌────────────────┬─────────────┬──────────┐
│ Feature name   │         Min │      Max │
├────────────────┼─────────────┼──────────┤
│ u              │ 1.3113e-06  │ 0.999996 │
│ v              │ 8.9407e-07  │ 0.999997 │
│ t2m            │ 5.06639e-06 │ 0.999995 │

```


# Running Tests

Our tests are written using [pytest](https://docs.pytest.org). We check that:
- The models can be instantiated with their default parameters, trained on a toy problem, onnx exported and reloaded for inference.
- The NamedTensor class can be instantiated and used to manipulate data and metadata.

```bash
docker build . -f Dockerfile -t mfai
docker run -it --rm mfai python -m pytest tests
```
# Contributing

We welcome contributions to this package. Our guidelines are the following:

- Submit a PR with a clear description of the changes and the motivation behind them.
- Make sure the current tests pass and add new tests if necessary to cover the new features. Our CI will fail with a **test coverage below 80%**.
- Make sure the code is formatted with [ruff](https://docs.astral.sh/ruff/)

# Acknowledgements

This package is maintained by the DSM/LabIA team at Météo-France. We would like to thank the authors of the papers and codes we used to implement the models (see [above links](#neural-network-architectures) to **arxiv** and **github**) and the authors of the libraries we use to build this package (see our [**requirements.txt**](requirements.txt)).
