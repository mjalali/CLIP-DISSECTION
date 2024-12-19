# CLIP-DISSECTION
A GitHub repository accompanying a "Dissecting CLIP: Decomposition with a Schur Complement-based Approach" paper

## Initializing SCE
To compute SCE score presented in the paper, initialize SCE with the following:
```python
from SCE.metric.SCE import SCE_Evaluator
from SCE.datasets.ImageFilesDataset import ImageFilesDataset

sigma = 3.5
fe = 'clip'

result_name = 'your_result_name'

img_pth = 'path_to_images'
text_pth = 'path_to_text.txt'

with open(text_pth, 'r') as f:
    prompts = f.readlines()
image_dataset = ImageFilesDataset(img_pth, name=result_name, extension='PNG')

SCE = SCE_Evaluator(logger_path='./logs', batchsize=64, sigma=sigma, eta=0, num_samples=num_samples, result_name=result_name, rff_dim=2500, save_visuals_path=f'visuals_{result_name}')
SCE.set_schur_feature_extractor(fe, save_path='./save')
```
In this snippet, parameter _sigma_ controls the bandwidth of the Gaussian Kernel and _fe_ allows to choose a specific feature extractor. In this repository we provide an implementation for CLIP, but other feature extractors may be used. We note that to access T2I and I2T evaluations, the feature extractor should support encoding of both text and image domains. 

## Computing SCE Score
To compute the SCE score of the paired text-image dataset, use the following function:
```python
# Get SCE Scores
img_generator_diversity, text_prompt_diversity = SCE.rff_schur_vendi(prompts, image_dataset)
```
The function returns decoupled diversity that comes from the text source and image source. 
