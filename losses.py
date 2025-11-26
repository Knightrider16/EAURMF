import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Contrastive_loss(nn.Module):
    def __init__(self, tau):
        super(Contrastive_loss,self).__init__()
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        # Compute cosine similarity between Samples
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        
        # checking, may break
        # reflection similarity (similarity between same modality)
        # FIXED: corrected sim(z1, z2) -> (z1, z1)
        refl_sim = f(self.sim(z1, z1))
        
        # similarity between different modalities
        between_sim = f(self.sim(z1, z2))
        
        # infoNCE loss
        ince_loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        return ince_loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool=True):
        l1 = self.semi_loss(z1,z2)
        l2 = self.semi_loss(z2,z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
    
def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = x.size(0)

    xx = x.unsqueeze(1)
    yy = y.unsqueeze(0)
    L2 = ((xx - yy) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernels = [torch.exp(-L2 / (bandwidth * (kernel_mul ** i))) for i in range(kernel_num)]
    return sum(kernels)


def MMD_loss(p, q):
    Kpp = gaussian_kernel(p, p)
    Kqq = gaussian_kernel(q, q)
    Kpq = gaussian_kernel(p, q)
    return Kpp.mean() + Kqq.mean() - 2 * Kpq.mean()

def wasserstein_distance(mu1, logvar1, mu2, logvar2):
    """Wasserstein-2 distance between two Gaussians"""
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # W2 distance = ||mu1 - mu2||^2 + ||sqrt(var1) - sqrt(var2)||^2
    mean_diff = torch.sum((mu1 - mu2) ** 2, dim=1)
    var_diff = torch.sum((torch.sqrt(var1) - torch.sqrt(var2)) ** 2, dim=1)
    
    return (mean_diff + var_diff).mean()


def JS_divergence(mu1, logvar1, mu2, logvar2):
    """
    Jensen-Shannon divergence between two Gaussian distributions.
    
    JS divergence is symmetric and bounded [0, log(2)], making it more stable than KL.
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
    
    Args:
        mu1, logvar1: mean and log-variance of first distribution
        mu2, logvar2: mean and log-variance of second distribution
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # Mixture distribution parameters (average of the two Gaussians)
    mu_m = 0.5 * (mu1 + mu2)
    var_m = 0.5 * (var1 + var2 + mu1**2 + mu2**2) - mu_m**2
    logvar_m = torch.log(var_m + 1e-8)  # Add epsilon for numerical stability
    
    # KL(P||M)
    kl_p_m = 0.5 * (logvar_m - logvar1 + (var1 + (mu1 - mu_m)**2) / (var_m + 1e-8) - 1)
    
    # KL(Q||M)
    kl_q_m = 0.5 * (logvar_m - logvar2 + (var2 + (mu2 - mu_m)**2) / (var_m + 1e-8) - 1)
    
    # JS divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)
    
    return js_div.sum(dim=1).mean()


def multimodalloss_js(
        txt_img_logits, 
        txt_logits, 
        tgt,
        img_logits,
        txt_mu, txt_logvar,
        img_mu, img_logvar,
        mu, logvar, 
        z    
    ):
    """
    Multimodal loss function with Jensen-Shannon divergence for cross-modal alignment.
    
    Uses JS divergence instead of KL for more stable training and symmetric distance.
    """
    
    # Standard KL divergence losses (regularization to prior N(0,1))
    txt_kl_loss = KL_normal(txt_mu, txt_logvar)
    img_kl_loss = KL_normal(img_mu, img_logvar)
    fusion_kl_loss = KL_normal(mu, logvar)

    # Jensen-Shannon divergence for cross-modal alignment (text vs image)
    js_loss = JS_divergence(txt_mu, txt_logvar, img_mu, img_logvar)

    # Classification losses
    IB_loss = F.cross_entropy(z, tgt)
    fusion_cls_loss = F.cross_entropy(txt_img_logits, tgt)

    # Combined loss with JS divergence for cross-modal alignment
    total_loss = (
        fusion_cls_loss +
        1e-3 * fusion_kl_loss +
        1e-3 * txt_kl_loss +
        1e-3 * img_kl_loss +
        1e-3 * IB_loss +
        1e-2 * js_loss  # JS divergence weight - adjust as needed
    )

    return total_loss

def multimodal_wasserstein(
        txt_img_logits, 
        txt_logits, 
        tgt, 
        img_logits, 
        txt_mu, txt_logvar, 
        img_mu, img_logvar, 
        mu, logvar, 
        z
    ):
    
    # Standard KL losses
    txt_kl_loss = KL_normal(txt_mu, txt_logvar)
    img_kl_loss = KL_normal(img_mu, img_logvar)
    kl_loss = KL_normal(mu, logvar)
    
    # Cross-modal alignment (choose one):
    # Option 1: Wasserstein distance
    cross_modal_loss = wasserstein_distance(txt_mu, txt_logvar, img_mu, img_logvar)
    
    # Option 2: JS divergence (more stable)
    # cross_modal_loss = JS_divergence(txt_mu, txt_logvar, img_mu, img_logvar)
    
    # Option 3: MMD (for sampled latents)
    # txt_z = reparameterise(txt_mu, torch.exp(0.5 * txt_logvar))
    # img_z = reparameterise(img_mu, torch.exp(0.5 * img_logvar))
    # cross_modal_loss = MMD_loss(txt_z, img_z)
    
    # Classification losses
    IB_loss = F.cross_entropy(z, tgt)
    fusion_cls_loss = F.cross_entropy(txt_img_logits, tgt)
    
    # Combined loss with cross-modal alignment
    total_loss = (fusion_cls_loss + 
                  1e-3 * kl_loss + 
                  1e-3 * txt_kl_loss + 
                  1e-3 * img_kl_loss + 
                  1e-3 * IB_loss +
                  1e-2 * cross_modal_loss)  # Adjust weight as needed
    
    return total_loss

def multimodalloss_mmd(
        txt_img_logits, 
        txt_logits, 
        tgt,
        img_logits,
        txt_mu, txt_logvar,
        img_mu, img_logvar,
        mu, logvar, 
        z    
    ):
    """
    Multimodal loss function with MMD for cross-modal alignment.
    
    Uses Maximum Mean Discrepancy (MMD) to align the distributions of 
    text and image latent representations.
    """
    
    # Standard KL divergence losses for regularization
    txt_kl_loss = KL_normal(txt_mu, txt_logvar)
    img_kl_loss = KL_normal(img_mu, img_logvar)
    fusion_kl_loss = KL_normal(mu, logvar)

    # Sample from the latent distributions for MMD computation
    txt_z = reparameterise(txt_mu, torch.exp(0.5 * txt_logvar))
    img_z = reparameterise(img_mu, torch.exp(0.5 * img_logvar))
    
    # MMD loss for cross-modal alignment
    mmd_loss = MMD_loss(txt_z, img_z)

    # Classification losses
    IB_loss = F.cross_entropy(z, tgt)
    fusion_cls_loss = F.cross_entropy(txt_img_logits, tgt)

    # Combined loss with MMD for cross-modal distribution matching
    total_loss = (
        fusion_cls_loss +
        1e-3 * fusion_kl_loss +
        1e-3 * txt_kl_loss +
        1e-3 * img_kl_loss +
        1e-3 * IB_loss +
        1e-2 * mmd_loss  # MMD weight - adjust as needed
    )

    return total_loss


def multimodalloss(
        txt_img_logits, 
        txt_logits, 
        tgt,
        img_logits,
        txt_mu, txt_logvar,
        img_mu, img_logvar,
        mu, logvar, 
        z    
    ):

    txt_kl_loss = KL_normal(txt_mu, txt_logvar)
    img_kl_loss = KL_normal(img_mu, img_logvar)
    fusion_kl_loss = KL_normal(mu, logvar)

    IB_loss = F.cross_entropy(z, tgt)
    fusion_cls_loss = F.cross_entropy(txt_img_logits, tgt)

    total_loss = (
        fusion_cls_loss +
        1e-3 * fusion_kl_loss +
        1e-3 * txt_kl_loss +
        1e-3 * img_kl_loss +
        1e-3 * IB_loss
    )


    return total_loss


def KL_normal(mu, logvar):
    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2  
    return kl_loss.sum(dim=1).mean()

def KL_regular(mu_1, logvar_1, mu_2, logvar_2):
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    KL_loss = logvar_2 - logvar_1 + ((var_1.pow(2) + (mu_1-mu_2).pow(2)) / (2 * var_2.pow(2))) - 0.5
    KL_loss = KL_loss.sum(dim=1).mean()
    return KL_loss

def reparameterise(mu, std):
    """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
    """        
    # get epsilon from standard normal
    eps = torch.randn_like(std) # x != y from random sample
    return mu + std * eps

def con_loss(txt_mu, txt_logvar, img_mu, img_logvar):
    """
        Contrastive loss encourages consistency within each modality by making different
        samples from same distribution similar to each other
        (positive -> positive sim high)
        (positive -> negative sim low, this becomes the case intrinsically)
        
        FIXED: removed while loop
    """
    Conloss = Contrastive_loss(0.5)

    t_z1 = reparameterise(txt_mu, txt_logvar)
    t_z2 = reparameterise(txt_mu, txt_logvar)
    
    i_z1 = reparameterise(img_mu,img_logvar)
    i_z2 = reparameterise(img_mu,img_logvar)



    loss_t = Conloss(t_z1, t_z2)
    loss_i = Conloss(i_z1, i_z2)
    
    return loss_t + loss_i

def cog_uncertainty_sample(mu_l, var_l, mu_v, var_v, sample_times=10):

    l_list = []
    for _ in range(sample_times):
        l_list.append(reparameterise(mu_l, var_l))
    l_sample = torch.stack(l_list, dim=1)

    v_list = []
    for _ in range(sample_times):
        v_list.append(reparameterise(mu_v, var_v))
    v_sample = torch.stack(v_list, dim=1)
    
    return l_sample, v_sample


def cog_uncertainty_normal(unc_dict, normal_type="None"):

    key_list = [k for k, _ in unc_dict.items()]
    comb_list = [t for _, t in unc_dict.items()]
    comb_t = torch.stack(comb_list, dim=1)
    mat = torch.exp(torch.reciprocal(comb_t))
    mat_sum = mat.sum(dim=-1, keepdim=True)
    weight = mat / mat_sum

    if normal_type == "minmax":
        weight = weight / torch.max(weight, dim=1)[0].unsqueeze(-1)  # [bsz, mod_num]
        for i, key in enumerate(key_list):
            unc_dict[key] = weight[:, i]
    else:
        pass
        # raise TypeError("Unsupported Operations at cog_uncertainty_normal!")

    return unc_dict