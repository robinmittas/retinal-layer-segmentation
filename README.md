# retina-segmentation

This repository contains code to segment 3 retinal layers (plus background and outer borders): ILM-OPL-HFL, ONL, and BMEIS-OB-RPE. It reproduces results from the paper:

> **Deep Learning based retinal layer segmentation in optical coherence tomography scans of patients with inherited retinal diseases**
> *Klinische Monatsblätter für Augenheilkunde* — https://www.thieme-connect.de/products/ejournals/pdf/10.1055/a-2227-3742.pdf

2D B-scans from Heidelberg Spectralis OCT volumes are used as input. The codebase also includes ETDRS grid analysis and thickness map computation.

---

## Setup

### Prerequisites

- Python 3.10 or 3.11
- A CUDA-capable GPU is strongly recommended for training

### 1. Clone the repository

```bash
git clone https://github.com/robinmittas/retinal-layer-segmentation.git
cd retinal-layer-segmentation
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install the package and dependencies

```bash
pip install --upgrade pip
pip install -e .
```

This installs the `retinal_seg` package in editable mode along with all dependencies listed in `requirements.txt`.

### 4. Verify the installation

```python
from retinal_seg.model.unet import build_unet
model = build_unet(img_height=496, img_width=512, n_classes=4)
model.summary()
```

---

## Usage

### Train a model

```python
from retinal_seg.data.loader import build_datasets
from retinal_seg.model.unet import build_unet
from retinal_seg.config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, BATCH_SIZE, EPOCHS

# x_y_map: dict mapping OCT .npy paths → segmentation .npy paths
train_ds, val_ds, test_ds = build_datasets(x_y_map, batch_size=BATCH_SIZE)

model = build_unet(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    img_channels=IMG_CHANNELS,
    n_classes=NUM_CLASSES,
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

### Compute an ETDRS thickness report

```python
from retinal_seg.analysis.etdrs import ETDRSUtils

utils = ETDRSUtils("path/to/volume.npy")
stats = utils.get_etdrs_stats()
# stats is a flat dict with keys like 'C0_thickness_mean', 'S1_neurosensory_retina', …
```

See `notebooks/` for end-to-end training, transfer learning, domain adaptation, and ETDRS analysis examples.

---

# Colab Notebooks
You can also run the code within these Colab notebooks. Here, we have included links to two Notebooks, which depict our final model setup. Please adjust the Runtime type to GPU to have a faster computation time.
- https://colab.research.google.com/drive/12wnQltPpyxaw5SCx9InbtBulSAwEfPdO Notebook to train the initial model on healthy individuals
- https://colab.research.google.com/drive/1ujpqwPge-6quofOe8VgFs4Ci9GNspajt Notebook to train the initial model on diseased individuals without using prior weights or information
- https://colab.research.google.com/drive/1gQkKExlk2yxnA8xzY03C4VMcjVNizmbj#scrollTo=j-nO5ZBTgD4U Notebook to train the initial model on diseased individuals by reusing prior weights of healthy model and freezing all layers except Batchnorm-layers
- https://colab.research.google.com/drive/1jz9xNO-1lm5vLg3vLqVNCZgqLv3mYTbs Notebook to load the trained models and segment images

# Results
![Figure 3](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/327fb8e7-c4c0-44b1-8e1d-e74a3c5a75b1)
(a) Exemplary slide in the middle of an OCT scan in a healthy patient with the ONL-’’hill’’
(b) Exemplary first slide of an OCT scan without ONL-’’hill’’. The same color codes for
retinal layers will be used within this work.
OCT = optical coherence tomography; ONL = outer nuclear layer
![Figure 4](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/9f4c4062-ecb1-4c84-99a8-800a9fb7dbe1)
Slide containing the ‘’ONL’’ hill (a) Prediction of the model. (b) Ground truth. (c) Input
slide.
ONL = outer nuclear layer
![Figure 5](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/cadec0c6-5387-47df-bea5-fc8a25bc6afd)
Regular slide. (a) Prediction of the model. (b) Ground truth. (c) Input slide.
![Figure 6](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/d2f44fe2-5f0c-4a32-b637-65cea0f6243b)
Trained model on the IRD dataset. (a) Prediction of the model. (b) Ground truth. (c) Input
slide.
IRD = inherited retinal disease
![Figure 7](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/e0b9dc14-fb1d-4dbe-8d1d-cbdfeb5461f9)
Trained model on the IRD dataset where ONL hill is visible. (a) Prediction of the model.
(b) Ground truth. (c) Input slide.
IRD = inherited retinal disease; ONL = outer nuclear layer
![Figure 8](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/8a0b74ee-501f-438f-9dd1-520c17cc2d91)
Trained model on a IRD dataset where ONL layer is almost vanished. (a) Prediction of
the model. (b) Ground truth. (c) Input slide.
IRD = inherited retinal disease; ONL = outer nuclear layer
![Figure 9](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/6fa7e90b-7e12-4bf0-baab-eaf10750ea2b)
Trained model on a IRD dataset: Model has problems when quality is too low. White
frame indicates some wrongly segmented pixels. (a) Prediction of the model. (b) Ground
This article is protected by copyright. All rights reserved.
Accepted Manuscript
truth. (c) Input slide.
IRD = inherited retinal disease
![Figure 10](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/1e3543cb-f9b5-4585-a34f-54794237ad73)
ETDRS regions.
![Figure 11](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/404ac3a2-19b8-4eab-a411-726c9dda74c5)
Thickness maps based on predicted segmentation of all retinal layers. (a) Thickness
map of a healthy individual. (b) Thickness map of an IRD patient. Interestingly, the
diseased individual has a similar full-retinal thickness.
Healthy: C0 Average thickness: 274µm; S2 Average thickness 301µm; S1 Average
thickness 354µm; N1 Average thickness: 352µm; N2 Average thickness: 325µm; I1
Average thickness: 352µm; I2 Average thickness: 302µm; T1 Average thickness:
348µm; T2 Average thickness: 294µm
Diseased: C0 Average thickness: 281µm; S2 Average thickness 270µm; S1 Average
thickness 359µm; N1 Average thickness: 369µm; N2 Average thickness: 310µm; I1
Average thickness: 366µm; I2 Average thickness: 279µm; T1 Average thickness:
343µm; T2 Average thickness: 278
IRD = inherited retinal disease
![Figure 12](https://github.com/robinmittas/retinal-layer-segmentation/assets/64377780/1001649d-78c9-4479-a51a-ec62875cef41)
Thickness maps based on predicted segmentation of ONL layer. (a) Thickness map of a
healthy individual. (b) Thickness map of an IRD patient. Evidently, the IRD patient has a
thinner ONL thickness, especially at the non-central regions.
This article is protected by copyright. All rights reserved.
Accepted Manuscript
Healthy: C0 Average thickness: 109µm; S2 Average thickness 80µm; S1 Average
thickness 88µm; N1 Average thickness: 94µm; N2 Average thickness: 75µm; I1 Average
thickness: 88µm; I2 Average thickness: 68µm; T1 Average thickness: 89µm; T2 Average
thickness: 76 µm
Diseased: C0 Average thickness: 115µm; S2 Average thickness 37µm; S1 Average
thickness 81µm; N1 Average thickness: 94µm; N2 Average thickness: 48µm; I1 Average
thickness: 92µm; I2 Average thickness: 42µm; T1 Average thickness: 89µm; T2 Average
thickness: 51 µmro
IRD = inherited retinal disease




## Acknowledgments
This repository contains code snippets from the following two repositories:
- https://github.com/yu02019/BEN (Model definitions)
- https://github.com/theislab/LODE/tree/master/ (Thickness map calculations)
