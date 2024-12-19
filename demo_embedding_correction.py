from SCE.metric.SCE import SCE_Evaluator
from SCE.datasets.ImageFilesDataset import ImageFilesDataset

#%%
if __name__ == '__main__':
    sigma = 3.5
    fe = 'clip'
    
    result_name = 'your_result_name'
    
    # Initialise reference embeddings to construct matrix Gamma*
    # In this example, "prompts" and "image_dataset" variables will serve to construct Gamma*
    img_pth = 'path_to_images'
    text_pth = 'path_to_text.txt'
    
    with open(text_pth, 'r') as f:
        prompts = f.readlines()
    image_dataset = ImageFilesDataset(img_pth, name=result_name, extension='PNG')
    
    
    num_samples = len(prompts)
    assert len(prompts) == len(image_dataset.files)
    
    
    SCE = SCE_Evaluator(logger_path='./logs', batchsize=64, sigma=sigma, eta=0, num_samples=num_samples, result_name=result_name, rff_dim=2500, save_visuals_path=f'visuals_{result_name}')
    SCE.set_schur_feature_extractor(fe, save_path='./save')
    SCE.set_feature_extractor(fe, save_path='./save')
    
    # Initialise images/texts to correct
    img_pth_to_correct = 'path_to_correction_images'
    text_pth_to_correct = 'path_to_correction_text.txt'
    
    with open(text_pth_to_correct, 'r') as f:
        prompts_to_correct = f.readlines()
    image_dataset_to_correct = ImageFilesDataset(img_pth_to_correct, name=result_name, extension='PNG')
    
    # Correct embeddings in T2I tasks (remove features from an image given a text description)
    corrected_t2i_embedding = SCE.corrected_embedding_t2i(prompts_to_correct, image_dataset_to_correct, prompts, image_dataset)
    
    # Correct embeddings in I2T tasks (remove features from a text caption given an image)
    corrected_i2t_embedding = SCE.corrected_embedding_i2t(prompts_to_correct, image_dataset_to_correct, prompts, image_dataset)
    
    
    





