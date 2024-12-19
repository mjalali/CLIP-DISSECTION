import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.linalg import eigh, eigvalsh, eigvals
from torch.distributions import Categorical
from torchvision.utils import save_image
import os
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

def normalized_gaussian_kernel(x, y, sigma, batchsize):
    '''
    calculate the kernel matrix, the shape of x and y should be equal except for the batch dimension

    x:      
        input, dim: [batch, dims]
    y:      
        input, dim: [batch, dims]
    sigma:  
        bandwidth parameter
    batchsize:
        Batchify the formation of kernel matrix, trade time for memory
        batchsize should be smaller than length of data

    return:
        scalar : mean of kernel values
    '''
    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx*batchsize:min((batchidx+1)*batchsize, y.shape[0])]
        res = torch.norm(x.unsqueeze(1)-y_slice, dim=2, p=2).pow(2)
        res = torch.exp((- 1 / (2*sigma*sigma)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

    return total_res

def cosine_kernel(x, y):
    total_res = torch.zeros((x.shape[0], y.shape[0]), device=x.device)
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            total_res[i][j] = torch.nn.functional.cosine_similarity(x[i], y[j], dim=0)
    return total_res / np.sqrt(x.shape[0] * y.shape[0])

def cov_through_cholesky_decomposition(K, args):
    try:
        V = torch.linalg.cholesky(K, upper=True)
        V = V.real
    except:
        args.logger.info('Cholesky failed, trying Cholesky_ex with perturbation...')
        # K += epsilon * torch.eye(K.size(0))
        V, info = torch.linalg.cholesky_ex(K, upper=True)
        
        degree = -12
        while not torch.all(info == 0):
            epsilon = 10**(degree)
            V, info = torch.linalg.cholesky_ex(K + epsilon * torch.eye(K.size(0)), upper=True)
            V = V.real
            degree += 1
        args.logger.info(f'Perturbed with 1e{degree}')
    
    cov = V @ V.T
    
    return cov, V

def rff_schur_complement(x, y, args):
    assert x.shape == y.shape
    n = x.shape[0]
    
    if args.kernel == 'gaussian':
        x_cov, omegas, x_feature = cov_rff(x, args.rff_dim, args.sigma, args.batchsize, args.normalise)
        y_cov, _, y_feature = cov_rff(y, args.rff_dim, args.sigma, args.batchsize, args.normalise, omegas)
        
        
        cov_yy = y_feature.T @ y_feature / n
        cov_xy = x_feature.T @ y_feature / n
        cov_xx = x_feature.T @ x_feature / n
    elif args.kernel == 'cosine':
        cov_xx = cosine_kernel(x.T, x.T)
        cov_yy = cosine_kernel(y.T, y.T)
        cov_xy = cosine_kernel(x.T, y.T)
    
    epsilon = 1e-12
    cov_xx = cov_xx + torch.eye(cov_xx.shape[0]).to(cov_xx.device) * epsilon
    cov_xx = torch.linalg.pinv(cov_xx)
    
    complement = cov_yy - cov_xy.T @ cov_xx @ cov_xy
    complement_difference = cov_xy.T @ cov_xx @ cov_xy
    
    if args.kernel == 'gaussian':
        features = {'x_feature': x_feature, 'y_feature': y_feature}
    else:
        features = None
    
    return complement, complement_difference, features

def cov_schur_complement(x, y, args):
    # y is images and x is text
    # basically computes conditioning on features x
    
    
    k_yy = normalized_gaussian_kernel(y, y, args.sigma, args.batchsize)
    k_xy = normalized_gaussian_kernel(x, y, args.sigma, args.batchsize)
    k_xx = normalized_gaussian_kernel(x, x, args.sigma, args.batchsize)

    top_row = torch.cat((k_xx, k_xy), dim=1)
    bottom_row = torch.cat((k_xy.T, k_yy), dim=1)
    K = torch.cat((top_row, bottom_row), dim=0)
    
    n = int(k_yy.size(0))
    
    
    _, V = cov_through_cholesky_decomposition(K, args)
    
    phi_x = V[:, :n]
    phi_y = V[:, n:]
    
    
    cov_xx = phi_x @ phi_x.T
    cov_yy = phi_y @ phi_y.T
    
    cov_xy = phi_x @ phi_y.T
    cov_yx = phi_y @ phi_x.T
    
    try:
        cov_xx = torch.linalg.inv(cov_xx)
    except:
        args.logger.info('True Inverse doesnt exist, using Pseudo...')
        epsilon = 1e-12
        cov_xx = cov_xx + torch.eye(n) * epsilon
        cov_xx = torch.linalg.pinv(cov_xx)

    complement = cov_yy - cov_xy.T @ cov_xx @ cov_xy
    
    return complement, cov_yx @ cov_xx @ cov_xy, (phi_x, phi_y)


def get_corrected_embedding(image_feature, text_feature, all_image_features, all_text_features, args, A_star=None):
    assert args.kernel in ['cosine', 'gaussian']
    if args.kernel == 'gaussian':
        assert args.sigma != None
    
    text_feature.to('cuda:0')
    image_feature.to('cuda:0')
    
    if A_star is not None:
        corrected_embedding = image_feature - text_feature @ A_star
        
        return corrected_embedding, A_star
    
    
    if args.kernel == 'cosine':
        cov_xx = cosine_kernel(all_text_features.T, all_text_features.T)
        epsilon = 1e-12
        cov_xx = cov_xx + torch.eye(cov_xx.shape[0]) * epsilon
        cov_xx = torch.linalg.pinv(cov_xx)
        cov_xy = cosine_kernel(all_text_features.T, all_image_features.T)
        
        A_star = cov_xx @ cov_xy
    elif args.kernel == 'gaussian':
        text_feature = text_feature.unsqueeze(0)
        text_feature = get_phi(text_feature, args.rff_dim, args.sigma, presign_omeaga=None)
        
        image_feature = image_feature.unsqueeze(0)
        image_feature = get_phi(image_feature, args.rff_dim, args.sigma, presign_omeaga=None)
        
        n = all_text_features.shape[0]
        
        x_cov, omegas, x_feature = cov_rff(all_text_features, args.rff_dim, args.sigma, args.batchsize, args.normalise)
        y_cov, _, y_feature = cov_rff(all_image_features, args.rff_dim, args.sigma, args.batchsize, args.normalise, omegas)
        
        cov_xx = x_feature.T @ x_feature / n
        epsilon = 1e-12
        cov_xx = cov_xx + torch.eye(cov_xx.shape[0]) * epsilon
        cov_xx = torch.linalg.pinv(cov_xx)

        cov_xy = x_feature.T @ y_feature / n
        A_star = cov_xx @ cov_xy
        A_star = A_star.to(text_feature.device)
        assert A_star.get_device() == text_feature.get_device()
    
    corrected_embedding = image_feature - text_feature @ A_star
    
    return corrected_embedding, A_star
    

def visualise_schur_image_modes(image_test_feats, image_test_dataset, text_test_feats, text_test_dataset, args):
    nrow = 2
    
    K_sc, K_inv_part, (phi_x, phi_y) = cov_schur_complement(text_test_feats, image_test_feats, args)

    eigenvalues_inv_part, eigenvectors_inv_part = torch.linalg.eigh(K_inv_part)
    eigenvalues_inv_part = eigenvalues_inv_part.real
    eigenvectors_inv_part = eigenvectors_inv_part.real
    
    eigenvalues_sc, eigenvectors_sc = torch.linalg.eigh(K_sc)
    eigenvalues_sc = eigenvalues_sc.real
    eigenvectors_sc = eigenvectors_sc.real
    
    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)
    
    m, max_id = eigenvalues_sc.topk(args.num_visual_mode)
    
    args.logger.info(f'Computed Eigenvalues as follows: {m}')

    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))

    for i in range(args.num_visual_mode):
        top_eigenvector = eigenvectors_sc[:, max_id[i]]
        
        # scores, idxes = top_eigenvector.topk(args.num_img_per_mode)
        
        # save_folder_name = os.path.join(root_dir, 'top{}'.format(i+1))
        # os.makedirs(save_folder_name)
        # summary = []
        
        top_eigenvector = top_eigenvector.unsqueeze(axis=-1)
        s_value = (phi_y.T @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name, exist_ok=True)
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        summary = []

        for j, idx in enumerate(topk_id.cpu()):
            
            top_imgs = transform(image_test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_top{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_top{i+1}.png'.format(j)), nrow=nrow)
    
    # BOTTOM
    for i in range(args.num_visual_mode):
        top_eigenvector = eigenvectors_sc[:, max_id[i]]
        
        top_eigenvector = top_eigenvector.unsqueeze(axis=-1)
        s_value = (phi_y.T @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        # topk_id = s_value.topk(args.num_img_per_mode)[1]
        _, bottomk_id = torch.topk(-s_value, args.num_img_per_mode, largest=True)
        
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'bottom{}'.format(i+1))
        os.makedirs(save_folder_name, exist_ok=True)
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        summary = []

        for j, idx in enumerate(bottomk_id.cpu()):
            top_imgs = transform(image_test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)

        
    save_rff_stats(eigenvalues_sc, eigenvalues_inv_part, root_dir, args)
    save_cdf(eigenvalues_sc, root_dir, args)

def visualise_schur_image_modes_rff(image_test_feats, image_test_dataset, image_test_idxs, text_test_feats, text_test_dataset, args):
    nrow = 2
        
    args.logger.info('Computing K from scratch...')
    K_sc, K_inv_part, features = rff_schur_complement(text_test_feats, image_test_feats, args)
    
    eigenvalues_inv_part, eigenvectors_inv_part = torch.linalg.eigh(K_inv_part)
    eigenvalues_inv_part = eigenvalues_inv_part.real
    eigenvectors_inv_part = eigenvectors_inv_part.real
    
    eigenvalues_sc, eigenvectors_sc = torch.linalg.eigh(K_sc)
    eigenvalues_sc = eigenvalues_sc.real
    eigenvectors_sc = eigenvectors_sc.real
    
    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)

    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))
    os.makedirs(root_dir, exist_ok=True)
    
    save_rff_stats(eigenvalues_sc, eigenvalues_inv_part, root_dir, args)
    
    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)
    
    # top eigenvalues
    m, max_id = eigenvalues_sc.topk(args.num_visual_mode)
    
    image_feature = features['y_feature']
    
    args.logger.info(f'Computed Eigenvalues as follows: {m}')

    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors_sc[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (image_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        summary = []

        for j, idx in enumerate(image_test_idxs[topk_id.cpu()]):
            idx = idx.int()
            top_imgs = transform(image_test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_top{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_top{i+1}.png'.format(j)), nrow=nrow)
    
    for i in range(args.num_visual_mode):
        
        top_eigenvector = eigenvectors_sc[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (image_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        _, bottomk_id = torch.topk(-s_value, args.num_img_per_mode, largest=True)

        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'bottom{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(image_test_idxs[bottomk_id.cpu()]):
            idx = idx.int()
            top_imgs = transform(image_test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)

        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
    
    
def visualize_mode_by_eigenvectors_spectral(test_feats, test_dataset, test_idxs, args):
    nrow = 2
    args.logger.info('Start compute covariance matrix')
    # x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)
    K = normalized_gaussian_kernel(test_feats, test_feats, args.sigma, args.batchsize)
    
    # cov, _ = cov_through_cholesky_decomposition(K, args)

    # eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)
    
    m, max_id = eigenvalues.topk(args.num_visual_mode)
    
    args.logger.info(f'Computed Eigenvalues as follows: {m}')

    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))
    

    for i in range(args.num_visual_mode):
        top_eigenvector = eigenvectors[:, max_id[i]]
        
        scores, idxes = top_eigenvector.topk(args.num_img_per_mode)
        
        save_folder_name = os.path.join(root_dir, 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        summary = []

        for j, idx in enumerate(idxes):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_top{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_top{i+1}.png'.format(j)), nrow=nrow)
    
    # Bottom CLasses
    for i in range(args.num_visual_mode):
        top_eigenvector = eigenvectors[:, max_id[i]]
        
        # scores, idxes = top_eigenvector.topk(args.num_img_per_mode)
        scores, idxes = torch.topk(-top_eigenvector, args.num_img_per_mode, largest=True)
        
        save_folder_name = os.path.join(root_dir, 'bottom{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(idxes):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
    
    # Plot and Save Eigenvalues
    save_stats(eigenvalues, root_dir, args)
    save_cdf(eigenvalues, root_dir, args)
    
def save_stats(eigenvalues, save_path, args):
    m, max_id = eigenvalues.topk(args.num_visual_mode)
    
    plt.bar(np.arange(1, len(m)+1, 1), m.cpu().numpy())
    plt.xticks(np.arange(1, len(m)+1, 1))
    
    plt.xlabel('Modes')
    plt.ylabel('Eigenvalue')

    pth = os.path.join(save_path, 'eigs.png')
    plt.savefig(pth, dpi=75)
    
    if not args.normalise:
        plt.clf()
        plt.bar(np.arange(2, len(m)+1, 1), m.cpu().numpy()[1:], color='orange')
        plt.xticks(np.arange(2, len(m)+1, 1))

        plt.xlabel('Modes')
        plt.ylabel('Eigenvalue')
        pth = os.path.join(save_path, 'eigs_without_first.png')
        plt.savefig(pth, dpi=75)
    
    schur_vendi = schur_vendi_from_eigs(eigenvalues, args)
    
    pth_textfile = os.path.join(save_path, 'stats.txt')
    with open(pth_textfile, 'w') as f:
        f.write(f'Eigenvalues of SC = {list(m.cpu())}\n')
        f.write(f'Sum of Eigenvalues of SC = {torch.sum(eigenvalues)}\n')
        f.write(f'Schur VENDI of SC = {schur_vendi}\n')

def save_rff_stats(eigenvalues_complement_difference, eigenvalues_complement, save_path, args):
    # SCHUR COMPLEMENT DIFFERENCE - IMAGE COMPONENT
    m_complement_difference, _ = eigenvalues_complement_difference.topk(args.num_visual_mode)
    plt.bar(np.arange(1, len(m_complement_difference)+1, 1), m_complement_difference.cpu().numpy())
    plt.xticks(np.arange(1, len(m_complement_difference)+1, 1))
    
    plt.xlabel('Modes')
    plt.ylabel('Eigenvalue')

    pth = os.path.join(save_path, 'eigs_complement_difference.png')
    plt.savefig(pth, dpi=75)
    
    # SCHUR COMPLEMENT - TEXT COMPONENT
    m_complement, _ = eigenvalues_complement.topk(args.num_visual_mode)
    plt.bar(np.arange(1, len(m_complement)+1, 1), m_complement.cpu().numpy())
    plt.xticks(np.arange(1, len(m_complement)+1, 1))
    
    plt.xlabel('Modes')
    plt.ylabel('Eigenvalue')

    pth = os.path.join(save_path, 'eigs_complement.png')
    plt.savefig(pth, dpi=75)
    
    
    schur_vendi_complement_difference = schur_vendi_from_eigs(eigenvalues_complement_difference, args)
    schur_vendi_complement = schur_vendi_from_eigs(eigenvalues_complement, args)
    
    pth_textfile = os.path.join(save_path, 'stats.txt')
    with open(pth_textfile, 'w') as f:
        f.write(f'Eigenvalues of SC-diff (image) = {list(m_complement_difference.cpu())}\n')
        f.write(f'Sum of Eigenvalues of SC-diff (image) = {torch.sum(eigenvalues_complement_difference)}\n')
        f.write(f'Schur VENDI of SC-diff (image) = {schur_vendi_complement_difference}\n\n\n')
        
        f.write(f'Eigenvalues of SC (text) = {list(m_complement.cpu())}\n')
        f.write(f'Sum of Eigenvalues of SC (text) = {torch.sum(eigenvalues_complement)}\n')
        f.write(f'Schur VENDI of SC (text) = {schur_vendi_complement}\n')

def save_cdf(eigenvalues, save_folder_name, args):
    eigenvalues = torch.sort(eigenvalues, descending=True)[0].cpu().numpy()
    
    # Compute the CDF by taking the cumulative sum of the probabilities
    cdf = np.cumsum(eigenvalues)
    cdf = np.insert(cdf, 0, 0)  # Insert a zero at the beginning to start the CDF from (0,0)

    # Plotting the CDF as an area under the curve
    plt.figure(figsize=(8, 5))
    plt.fill_between(range(len(cdf)), cdf, color="skyblue", alpha=0.4)  # Filling the area under the curve
    plt.plot(cdf, linestyle='-', color='b')  # Line plot of the CDF
    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Eigenvalue ID')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.xticks(range(0, len(cdf), 500))  # Adjust x-axis ticks to match the array index
    plt.xlim(0, len(cdf) - 1)  # Adjust x-axis limits to ensure it covers the full range of indices
    plt.ylim(0, 1)  # Adjust y-axis limits to ensure it starts at 0 and ends at 1
    
    pth = os.path.join(save_folder_name, 'cdf.png')
    plt.savefig(pth, dpi=75)
    
def visualize_mode_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args):
    nrow = 2
    args.logger.info('Start compute covariance matrix')
    x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)

    test_idxs = test_idxs.to(dtype=torch.long)

    args.logger.info('Start compute eigen-decomposition')
    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    args.logger.info(f'Sum of Eigenvalues: {torch.sum(eigenvalues)}')

    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)
    
    # top eigenvalues
    m, max_id = eigenvalues.topk(args.num_visual_mode)
    
    
    args.logger.info(f'Computed Eigenvalues as follows: {m}')

    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (x_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        summary = []

        for j, idx in enumerate(test_idxs[topk_id.cpu()]):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_top{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_top{i+1}.png'.format(j)), nrow=nrow)
    
    for i in range(args.num_visual_mode):
        
        top_eigenvector = eigenvectors[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (x_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        _, bottomk_id = torch.topk(-s_value, args.num_img_per_mode, largest=True)

        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'bottom{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(test_idxs[bottomk_id.cpu()]):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
    
    # Plot and Save Eigenvalues
    save_stats(eigenvalues, root_dir, args)
    save_cdf(eigenvalues, root_dir, args)
    
def get_phi(x, feature_dim, std, presign_omeaga=None):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]
    
    return batched_rff

def cov_rff2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2

    return cov, batched_rff.squeeze()

def cov_diff_rff(x, y, feature_dim, std, batchsize=16):
    assert len(x.shape) == len(y.shape) == 2 # [B, dim]

    B, D = x.shape
    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')

    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)
    y_cov, y_feature = cov_rff2(y, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)

    return x_cov, y_cov, omegas, x_feature, y_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim], [B, 2 * feature_dim]

def cov_rff(x, feature_dim, std, batchsize=16, normalise=True, presign_omegas = None):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    
    if presign_omegas is None:
        omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omegas

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas, normalise=normalise)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]

def non_fourier_scores(x_feats, args, alpha=2, t=100):
    K = normalized_gaussian_kernel(x_feats, x_feats, args.sigma, args.batchsize)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    results = calculate_stats(eigenvalues)
    
    
    return results

def fourier_scores(test_feats, args, alpha=2):
    args.logger.info('Start compute covariance matrix')
    if args.kernel == 'gaussian':
        x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)
    elif args.kernel == 'cosine':
        x_cov = cosine_kernel(test_feats.T, test_feats.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    results = calculate_stats(eigenvalues)
    return results

def calculate_stats(eigenvalues, alpha=2):
    results = {}
    epsilon = 1e-10  # Small constant to avoid log of zero

    # Ensure eigenvalues are positive and handle alpha = 1 case
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    
    log_eigenvalues = torch.log(eigenvalues)
    
    entanglement_entropy = -torch.sum(eigenvalues * log_eigenvalues)# * 100
    vendi = torch.exp(entanglement_entropy)
    results['VENDI-1.0'] = np.around(vendi.item(), 2)
    
    for alpha in [1.5, 2, 2.5, 'inf']:
        if alpha != 'inf':
            entropy = (1 / (1-alpha)) * torch.log(torch.sum(eigenvalues**alpha))
            f_rke = torch.exp(entropy)
        else:
            f_rke = 1 / torch.max(eigenvalues)

        results[f'VENDI-{alpha}'] = np.around(f_rke.item(), 2)
    return results

def save_eigenvals(test_feats, where_to, fkea_score, args):
    
    args.logger.info('Start compute covariance matrix')
    if fkea_score:
        x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)
    else:
        x_cov = normalized_gaussian_kernel(test_feats, test_feats, args.sigma, args.batchsize)

    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    

    os.makedirs(where_to, exist_ok=True)
    torch.save(eigenvalues, os.path.join(where_to, f'n_{test_feats.shape[0]}_sigma_{args.sigma}.pt'))
    
def schur_vendi_from_eigs(eigenvalues, args):
    epsilon = 1e-10  # Small constant to avoid log of zero
    
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    
    eig_sum = torch.sum(eigenvalues)
    
    log_eigenvalues = torch.log(torch.div(eigenvalues, eig_sum))
    
    entanglement_entropy = -torch.sum(eigenvalues * log_eigenvalues)# * 100
    vendi = torch.exp(entanglement_entropy)
    
    return vendi.item()

def rff_sce_from_feats(text_test_feats, image_test_feats, args, K=None):
    
    if K is None:
        complement, complement_difference, _ = rff_schur_complement(text_test_feats, image_test_feats, args)
    
    epsilon = 1e-10  # Small constant to avoid log of zero
    
    eigenvalues_complement_difference, _ = torch.linalg.eigh(complement_difference)
    eigenvalues_complement_difference = eigenvalues_complement_difference.real
    
    vendi_complement_difference = schur_vendi_from_eigs(eigenvalues_complement_difference, args) # text part
    
    eigenvalues_complement, _ = torch.linalg.eigh(complement)
    eigenvalues_complement = eigenvalues_complement.real
    
    vendi_complement = schur_vendi_from_eigs(eigenvalues_complement, args) # image part
    
    
    return vendi_complement, vendi_complement_difference


    
    










