import torch
import torch.nn as nn

def weighted_binary_cross_entropy_loss(pred, target, from_logits=True, weight_list=None):
    if weight_list:
        neg_target = torch.ones([target.shape[0], 1]).cuda() - target
        weight = neg_target*weight_list[0] + target*weight_list[1]
    else:
        weight = None
    if from_logits:
        loss_clf = nn.functional.binary_cross_entropy_with_logits(pred, target, weight=weight) 
    else:
        loss_clf = nn.functional.binary_cross_entropy(pred, target, weight=weight)
    return loss_clf

def euclidean_distance_to_mean(z_list):
    z_stacked = torch.stack(z_list)  # concatenates sequence of tensors along a new dimension
    z_mean = torch.mean(z_stacked, dim=0) # get the mean of z_list along the stacked dimension
    loss_mse_sum = 0
    for i in range(len(z_list)):
        loss_mse_sum += nn.functional.mse_loss(z_list[i], z_mean)
    return loss_mse_sum

def cal_gradient_penalty(netD, real_data, fake_data, device, lambda_gp):
    """
    Calculate gradient penalty
    WGAN-GP: https://arxiv.org/abs/1704.00028
    
    Arguments
    netD: the discriminator network
    real_data: sampled from modality 1
    fake_data: sampled from modality 2
    device: device available
    lambda_gp: 
    """
    batch_size = real_data.size()[0] 
   
    # Calculate interpolates
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)  

    interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolates = Variable(interpolates, requires_grad=True)
    interpolates = interpolates.to(device)

    # Calculate disc_interpolates 
    disc_interpolates = netD(interpolates)

    # Calculate gradients 
    gradients = autograd.grad(disc_interpolates, interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty
