import torch
import wandb
import einops
import torch.nn.functional as F
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing


def peak_signal_to_noise_ratio(x_hat_logits, x_true):
    mse = torch.mean(torch.square(F.sigmoid(x_hat_logits) - x_true))
    max_value = torch.Tensor([1.]).to(mse.device)
    return 20 * torch.log10(max_value) - 10 * torch.log10(mse)


def discretize_unique(z):
    """

    :param z: (num_samples, num_dims)
    :return:
    """
    ret = np.zeros_like(z, dtype=np.int32)
    for i in range(z.shape[1]):
        unique, labels = np.unique(z[:, i], return_inverse=True)
        ret[:, i] = labels
    return ret


def compute_entropy(z):
    """

    :param z: (num_samples, num_dims)
    :return:
    """
    ret = np.zeros(z.shape[1])
    if z.dtype in [np.int32, np.int64, torch.int32, torch.int64]:
        for i in range(z.shape[1]):
            ret[i] = sklearn.metrics.mutual_info_score(z[:, i], z[:, i])
    else:
        for i in range(z.shape[1]):
            ret[i] = sklearn.feature_selection.mutual_info_regression(z[:, i][:, None], z[:, i])
    return ret


def log_reconstruction_metrics(aux, step, use_wandb: bool = False):
    num_samples = 16
    true = einops.rearrange(aux['x_true'][:num_samples], 'b c h w -> h (b w) c')
    predicted = einops.rearrange(F.sigmoid(aux['x_hat_logits'][:num_samples]), 'b c h w -> h (b w) c')
    # predicted = einops.rearrange(aux['x_hat_logits'][:num_samples], 'b c h w -> h (b w) c')
    absolute_diff = torch.abs(true - predicted)
    image = torch.cat((true, predicted, absolute_diff), dim=0)
    psnr = peak_signal_to_noise_ratio(aux['x_hat_logits'], aux['x_true'])

    if use_wandb:
        wandb.log({
            'reconstructions': wandb.Image(image.detach().cpu().numpy()),
            'ae/psnr': psnr.mean().item()
        }, step=step)


def compute_pairwise_mutual_information(latents, sources, estimator: str, bins=20, normalization='sources'):
    ret = np.zeros((latents.shape[1], sources.shape[1]))
    sources = discretize_unique(sources)
    source_entropies = compute_entropy(sources)

    latent_ranges = torch.max(latents, dim=0) - torch.min(latents, dim=0)

    latent_mask = latent_ranges > torch.max(latent_ranges) / 8
    latents = discretize_unique(latents)
    latent_entropies = compute_entropy(latents)

    for i in range(latents.shape[1]):
        for j in range(sources.shape[1]):
            if estimator == 'discrete-discrete':
                ret[i, j] = sklearn.metrics.mutual_info_score(latents[:, i], sources[:, j])
            else:
                raise NotImplementedError

    if normalization == 'sources':
        ret /= source_entropies[None, :]
    else:
        raise NotImplementedError

    ret = torch.nan_to_num(ret, nan=0.)
    return ret, latent_mask


def compute_mutual_information_ratio(pairwise_mutual_information, latent_mask, info_metric):
    pairwise_mutual_information = pairwise_mutual_information * latent_mask[:, None]
    num_sources = pairwise_mutual_information.shape[1]
    num_active_latents = torch.sum(latent_mask)

    if info_metric == 'modularity':
        preferences = torch.max(pairwise_mutual_information, dim=1) / torch.sum(pairwise_mutual_information, dim=1)
        mask = torch.isfinite(preferences)
        mir = (torch.sum(preferences[mask]) / torch.sum(mask) - 1 / num_sources) / (1 - 1 / num_sources)
    elif info_metric == 'compactness':
        preferences = torch.max(pairwise_mutual_information, dim=0) / torch.sum(pairwise_mutual_information, dim=0)
        mir = (torch.sum(preferences) / num_sources - 1 / num_active_latents) / (1 - 1 / num_active_latents)
    else:
        raise NotImplementedError
    return mir


def compute_mutual_information_gap(pairwise_mutual_information, latent_mask, info_metric):
    """

    :param pairwise_mutual_information: (num_latents, num_sources)
    :param latent_mask: (num_latents,)
    :param per: 'latent' -> modularity, 'source' -> compactness
    :return:
    """

    if info_metric == 'modularity':
        mig = np.zeros(pairwise_mutual_information.shape[0])
        sorted_pairwise_mutual_information = np.sort(pairwise_mutual_information, axis=1)
        for i in range(pairwise_mutual_information.shape[0]):
            mig[i] = (sorted_pairwise_mutual_information[i, -1] - sorted_pairwise_mutual_information[i, -2])
        mig = mig[latent_mask]
    elif info_metric == 'compactness':
        mig = np.zeros(pairwise_mutual_information.shape[1])
        sorted_pairwise_mutual_information = np.sort(pairwise_mutual_information, axis=0)
        for i in range(pairwise_mutual_information.shape[1]):
            mig[i] = (sorted_pairwise_mutual_information[-1, i] - sorted_pairwise_mutual_information[-2, i])
    else:
        raise NotImplementedError
    return torch.mean(mig)


def process_latents(latents):
    if latents.dtype in [np.int64, np.int32, torch.int64, torch.int32]:
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        latents = one_hot_encoder.fit_transform(latents)
    elif latents.dtype in [np.float32, np.float64, torch.float32, torch.float64]:
        standardizer = preprocessing.StandardScaler()
        latents = standardizer.fit_transform(latents)
    else:
        raise ValueError(f'latents.dtype {latents.dtype} not supported')
    return latents


def log_disentanglement_metrics(model, aux, step, use_wandb: bool = False):
    pairwise_mutual_infos = {}
    mask_hats = {}
    normalizations = ['sources']

    for normalization in normalizations:
        estimator = 'discrete-discrete'
        name = f'normalization={normalization}/{estimator}'
        pairwise_mutual_infos[name], mask_hats[name] = compute_pairwise_mutual_information(
            latents=aux['z_hat'],
            sources=aux['z_true'],
            estimator=estimator,
            normalization=normalization
        )

    for k, v in pairwise_mutual_infos.items():
        fig, ax = plt.subplots(figsize=(6, aux['z_hat'].shape[1] ** 0.8))
        sns.heatmap(v, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1,
                    yticklabels=[f'z{i}' for i in range(v.shape[0])], xticklabels=[f's{i}' for i in range(v.shape[1])],
                    annot_kws={'fontsize': 8})
        for i, label in enumerate(ax.get_yticklabels()):
            if mask_hats[k][i] == 0:
                label.set_color('red')
        fig.tight_layout()
        plt.show()
        if use_wandb:
            wandb.log({f'pairwise_mutual_information/{k}': wandb.Image(fig)}, step=step)
        plt.close()

        for info_metric in ['modularity', 'compactness']:
            if use_wandb:
                wandb.log({
                    f'{info_metric}/{k}/ratio': compute_mutual_information_ratio(v, mask_hats[k], info_metric),
                    f'{info_metric}/{k}/gap': compute_mutual_information_gap(v, mask_hats[k], info_metric)

                }, step=step)

