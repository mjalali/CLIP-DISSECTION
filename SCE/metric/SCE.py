import torch
import torchvision.datasets as dset
import pandas as pd
from argparse import ArgumentParser, Namespace
from .algorithm_utils import *
from os.path import join
from SCE.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from SCE.features.SWAVFeatureExtractor import SWAVFeatureExtractor
from SCE.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from SCE.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from SCE.features.PixelFeatureExtractor import MonoPixelFeatureExtractor, ColorPixelFeatureExtractor
from SCE.features.BERTFeatureExtractor import BERTFeatureExtractor

from SCE.schur_features.ImageCLIPFeatureExtractor import ImageCLIPFeatureExtractor
from SCE.schur_features.TextCLIPFeatureExtractor import TextCLIPFeatureExtractor

import time
import logging
import sys

def get_logger(filepath='./logs/novelty.log'):
    '''
        Information Module:
            Save the program execution information to a log file and output to the terminal at the same time
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

class SCE_Evaluator():
    def __init__(self, logger_path : str, sigma : float, eta : float, result_name: str, num_samples: int = 5000, batchsize: int = 128, rff_dim: int = 0, normalise: bool = False, save_visuals_path: str = 'visuals'):
        self.logger_path = logger_path
        self.sigma = sigma
        self.eta = eta
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.rff_dim = rff_dim
        self.normalise = normalise
        self.save_visuals_path = save_visuals_path

        self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.result_name = '{}_num_{}_sigma_{}_eta_{}'.format(result_name, num_samples, sigma, eta)
        self.save_feats_name = '{}_num_{}'.format(result_name, num_samples)


        self.feature_extractor = None
        self.schur_image_feature_extractor = None
        self.schur_text_feature_extractor = None
        self.name_feature_extractor = None
        self.running_logger = None

        self.init_running_logger()
        self.running_logger.info("SCE Evaluator Initialized.")
        
    
    def init_running_logger(self):
        self.running_logger = get_logger(join(self.logger_path, 'run_{}_{}.log'.format(self.result_name, self.current_time)))

    
    def set_feature_extractor(self, name: str, save_path):
        self.save_path = save_path
        
        if name.lower() == 'inception':
            self.feature_extractor = InceptionFeatureExtractor(save_path, logger=self.running_logger)
        # elif name.lower() == 'dinov2': # CHANGE BACK LATER
        elif 'dinov2' in name.lower():
            print('BEGIN DINOv2')
            self.feature_extractor = DINOv2FeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'clip':
            self.feature_extractor = CLIPFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'swav':
            self.feature_extractor = SWAVFeatureExtractor(save_path, logger=self.running_logger)    
        elif name.lower() =='colored_pixel':
            print('BEGIN COLORED PIXEL')
            self.feature_extractor = ColorPixelFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() =='mono_pixel':
            self.feature_extractor = MonoPixelFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'bert':
            self.feature_extractor = BERTFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() in ['gemini', 'gpt_large', 'gpt_small']:
            self.feature_extractor = -1
        else:
            raise NotImplementedError(
                f"Cannot get feature extractor '{name}'. Expected one of ['inception', 'dinov2', 'clip']"
            )
        self.name_feature_extractor = name.lower()
        self.running_logger.info("Initialized feature-extractor network: {}".format(self.name_feature_extractor))
    
    def set_schur_feature_extractor(self, name: str, save_path=None):
        self.save_path = save_path
        
        
        if name.lower() == 'clip':
            self.schur_image_feature_extractor = ImageCLIPFeatureExtractor(save_path, logger=self.running_logger)
            self.schur_text_feature_extractor = TextCLIPFeatureExtractor(save_path, logger=self.running_logger)
        else:
            raise NotImplementedError(
                f"Cannot get feature extractor '{name}'. Expected one of ['clip']"
            )
        self.name_feature_extractor = name.lower()
        self.running_logger.info("Initialized feature-extractor network: {}".format(self.name_feature_extractor))
        
        try:
            self.A_star = torch.load(os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt'))
        except:
            self.A_star = None
        
    def schur_clustering_modes_of_dataset(self,
                                          prompts,
                                          test_dataset: torch.utils.data.Dataset,
                                          paired_test_feats = None):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur_cov',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise,
                         kernel='gaussian'
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, image_test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=False, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, text_test_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=False, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        visualise_schur_image_modes(image_test_feats, test_dataset, text_test_feats, prompts, args)
        
        
    def rff_schur_clustering_modes_of_dataset(self,
                                          prompts,
                                          image_dataset: torch.utils.data.Dataset,
                                          recompute=False,
                                          paired_test_feats = None):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur_rff',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise,
                         kernel='gaussian'
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(image_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, image_test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(image_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, text_test_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        visualise_schur_image_modes_rff(image_test_feats, image_dataset, image_test_idxs, text_test_feats, prompts, args)
        
    
    def rff_clustering_modes_of_dataset(self,
                                        test_dataset: torch.utils.data.Dataset):
        
        assert self.rff_dim > 0
        
        args = Namespace(num_samples=self.num_samples, 
                          batchsize=self.batchsize, 
                          sigma=self.sigma, 
                          rff_dim=self.rff_dim,
                          logger=self.running_logger,
                          backbone=self.name_feature_extractor,
                          visual_name=self.result_name,
                          current_time=self.current_time,
                          path_save_visual=f'./{self.save_visuals_path}/modes_original_rff',
                          num_visual_mode=5,
                          num_img_per_mode=50,
                          resize_img_to=224,
                          normalise = self.normalise,
                          kernel='gaussian'
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))

        if self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_feature_extractor(name='clip', logger=self.running_logger)
        
        with torch.no_grad():
            self.running_logger.info("Calculating test feats:")
            test_feats, test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}".format(len(test_feats)))
        visualize_mode_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args)
        

    def rff_schur_vendi(self,
                        prompts,
                        test_dataset: torch.utils.data.Dataset,
                        paired_test_feats = None,
                        recompute=False,
                        kernel='gaussian'):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise,
                         kernel = kernel
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, image_test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, text_test_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        # print(text_test_feats.shape, image_test_feats.shape)
        vendi_complement, vendi_complement_difference = rff_schur_vendi_from_feats(text_test_feats, image_test_feats, args, K=None)
        
        return vendi_complement, vendi_complement_difference
    
    
    def corrected_embedding_t2i(self,
            prompts_to_embed,
            test_dataset_to_embed,
            prompts,
            test_dataset: torch.utils.data.Dataset,
            paired_test_feats = None,
            recompute = False,
            recompute_reference = False,
            kernel='cosine'):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise,
                         kernel=kernel
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, _ = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute_reference, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, _ = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute_reference, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        with torch.no_grad():
            self.running_logger.info("Calculating image test feats:")
            image_feats_to_correct, _ = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset_to_embed, 
                                                                    name = 'sample_test_' + self.save_feats_name, 
                                                                    recompute=recompute, 
                                                                    num_samples=len(prompts_to_embed), 
                                                                    batchsize=args.batchsize)
            self.running_logger.info("Calculating text test feats:")
            text_feats_to_correct, _ = self.schur_text_feature_extractor.get_features_and_idxes(prompts_to_embed, 
                                                                    name = 'sample_test_' + self.save_feats_name, 
                                                                    recompute=recompute, 
                                                                    num_samples=len(prompts_to_embed), 
                                                                    batchsize=args.batchsize)
                
        
        embed, A_star = get_corrected_embedding(image_feats_to_correct, text_feats_to_correct, image_test_feats, text_test_feats, args, A_star = self.A_star)

        if not os.path.isfile(os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt')):
            self.A_star = A_star
            torch.save(A_star, os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt'))
        
        return embed
    
    
    def corrected_embedding_i2t(self,
            prompts_to_embed,
            test_dataset_to_embed,
            prompts,
            test_dataset: torch.utils.data.Dataset,
            paired_test_feats = None,
            recompute = False,
            kernel='cosine'):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise,
                         kernel=kernel
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, image_test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, text_test_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        with torch.no_grad():
            self.running_logger.info("Calculating image test feats:")
            image_feats_to_correct, image_feats_to_correct_idxs = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset_to_embed, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=recompute, 
                                                                    num_samples=len(prompts_to_embed), 
                                                                    batchsize=args.batchsize)
            self.running_logger.info("Calculating text test feats:")
            text_feats_to_correct, text_feats_to_correct_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts_to_embed, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=recompute, 
                                                                    num_samples=len(prompts_to_embed), 
                                                                    batchsize=args.batchsize)
                
        
        #test(text_test_feats, image_test_feats, args)
        image_feature = image_test_feats[0]
        text_feature = text_test_feats[0]
        
        print(image_feats_to_correct.shape, text_feats_to_correct.shape)
        # embed = get_corrected_embedding(, text_feats_to_correct, image_test_feats, text_test_feats, args)
        embed = get_corrected_embedding(text_feats_to_correct, image_feats_to_correct, text_test_feats, image_test_feats, args)
        print(embed.shape)
        return embed
    
    def corrected_embedding_t2t(self,
            test_prompts_to_embed,
            test_prompts_to_cancel,
            prompts,
            prompts_to_cancel,
            paired_test_feats = None,
            recompute = False,
            kernel='cosine'):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise,
                         kernel=kernel
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(prompts_to_cancel)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating text test to cancel feats:")
                ds_text_feats_to_cancel, _ = self.schur_text_feature_extractor.get_features_and_idxes(prompts_to_cancel, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                ds_text_test_feats, _ = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        with torch.no_grad():
            self.running_logger.info("Calculating text test to cancel feats:")
            text_feats_to_cancel, _ = self.schur_text_feature_extractor.get_features_and_idxes(test_prompts_to_cancel, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=recompute, 
                                                                    num_samples=len(test_prompts_to_embed), 
                                                                    batchsize=args.batchsize)
            self.running_logger.info("Calculating text test feats:")
            text_feats, _ = self.schur_text_feature_extractor.get_features_and_idxes(test_prompts_to_embed, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=recompute, 
                                                                    num_samples=len(test_prompts_to_embed), 
                                                                    batchsize=args.batchsize)
                
        
        embed, A_star = get_corrected_embedding(text_feats, text_feats_to_cancel, ds_text_test_feats, ds_text_feats_to_cancel, args)
        print(os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt'))
        if not os.path.isfile(os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt')):
            self.A_star = A_star
            torch.save(A_star, os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt'))
        return embed

            
        
            
            
        
        
        
        
        
        
        
        

        



