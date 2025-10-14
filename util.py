import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Contrastive_loss(nn.Module):
    def __init__(self,tau):
        super(Contrastive_loss,self).__init__()
        self.tau=tau

    def sim(self,z1:torch.Tensor,z2:torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1,z2.t())
    
    def semi_loss(self,z1:torch.Tensor,z2:torch.Tensor):
        f=lambda x: torch.exp(x/self.tau)
        refl_sim = f(self.sim(z1,z2))
        between_sim=f(self.sim(z1,z2))

        return -torch.log(between_sim.diag()/(refl_sim.sum(1)+between_sim.sum(1)-refl_sim.diag()))
    
    def forward(self,z1:torch.Tensor,z2:torch.Tensor,mean:bool=True):
        l1=self.semi_loss(z1,z2)
        l2=self.semi_loss(z2,z1)
        ret=(l1+l2)*0.5
        ret=ret.mean() if mean else ret.sum()
        return ret

def totolloss(txt_img_logits, txt_logits,tgt,img_logits,txt_mu,txt_logvar,img_mu,img_logvar,mu,logvar,z):

    txt_kl_loss = KL_normal(txt_mu, txt_logvar)

    img_kl_loss = KL_normal(img_mu, img_logvar)

    kl_loss = KL_normal(mu, logvar)

    IB_loss = F.cross_entropy(z,tgt)

    fusion_cls_loss = F.cross_entropy(txt_img_logits,tgt)

    totol_loss=fusion_cls_loss+1e-3*kl_loss+1e-3*txt_kl_loss+1e-3*img_kl_loss+1e-3*IB_loss
    return totol_loss

def KL_normal(mu, logvar):
    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2  
    return kl_loss.sum(dim=1).mean()

def KL_regular(mu_1,logvar_1,mu_2,logvar_2):
    var_1=torch.exp(logvar_1)
    var_2=torch.exp(logvar_2)
    KL_loss=logvar_2-logvar_1+((var_1.pow(2)+(mu_1-mu_2).pow(2))/(2*var_2.pow(2)))-0.5
    KL_loss=KL_loss.sum(dim=1).mean()
    return KL_loss

def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]        
    """        
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std*eps

def con_loss(txt_mu,txt_logvar,img_mu,img_logvar):
    Conloss=Contrastive_loss(0.5)
    while True:
        t_z1 = reparameterise(txt_mu, txt_logvar)
        t_z2 = reparameterise(txt_mu, txt_logvar)
        
        if not np.array_equal(t_z1, t_z2):
            break 
    while True:
        i_z1=reparameterise(img_mu,img_logvar)
        i_z2=reparameterise(img_mu,img_logvar)
        
        if not np.array_equal(t_z1, t_z2):
            break 


    loss_t=Conloss(t_z1,t_z2)
    loss_i=Conloss(i_z1,i_z2)
    
    return loss_t+loss_i

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