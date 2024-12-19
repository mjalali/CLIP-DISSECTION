from SCE.metric.SCE import SCE_Evaluator
from SCE.datasets.ImageFilesDataset import ImageFilesDataset

#%%
if __name__ == '__main__':
    sigma = 3.5
    fe = 'clip'
    
    result_name = 'your_result_name'
    
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
    

    
    # Cluster Results
    SCE.rff_schur_clustering_modes_of_dataset(prompts, image_dataset)
    
    # Get SCE Scores
    img_generator_diversity, text_prompt_diversity = SCE.rff_schur_vendi(prompts, image_dataset)

    
    
    





