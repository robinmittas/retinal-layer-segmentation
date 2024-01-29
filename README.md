# retina-segmentation
This repository consists of code to segment 3 retinal layers (and the background, and outer borders), i.e. ILM-OPL-HFL, ONL, and BMEIS-OB-RPE. To see how to train the model, refer to the `notebooks` folders.
We used 2D slides of OCT scans as Input from healthy patients with the eventual goal of segmenting the layers of IRD (inherited retinal diseases) patients. 
Finally, the repository contains code to calculate Thickness maps and ETDRS grid regions and average thicknesses. 

# Colab Notebooks
You can also run the code within these Colab notebooks. Here, we have included links to two Notebooks, which depict our final model setup. Please adjust the Runtime type to GPU to have a faster computation time.
- https://colab.research.google.com/drive/12wnQltPpyxaw5SCx9InbtBulSAwEfPdO Notebook to train the initial model on healthy individuals
- https://colab.research.google.com/drive/1ujpqwPge-6quofOe8VgFs4Ci9GNspajt Notebook to train the initial model on diseased individuals without using prior weights or information
- https://colab.research.google.com/drive/1gQkKExlk2yxnA8xzY03C4VMcjVNizmbj#scrollTo=j-nO5ZBTgD4U Notebook to train the initial model on diseased individuals by reusing prior weights of healthy model and freezing all layers except Batchnorm-layers
- https://colab.research.google.com/drive/1jz9xNO-1lm5vLg3vLqVNCZgqLv3mYTbs Notebook to load the trained models and segment images

## Acknowledgments
This repository contains code snippets from the following two repositories:
- https://github.com/yu02019/BEN (Model definitions)
- https://github.com/theislab/LODE/tree/master/ (Thickness map calculations)

