import torch
import numpy as np
import torch.nn.functional as F
mse = torch.nn.MSELoss()

def coral(source, target):

    d = source.size(1)  # dim vector

    source_c, source_mu = compute_covariance(source)
    target_c, target_mu = compute_covariance(target)
    # print('AAAAAAAAAAAAAAAAAAAAA') 
    # print(source_c, target_c)
    # print('BBBBBBBBBBBBBBBBBBBB') 
    # print(source_mu, target_mu)
   

    loss_c = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss_c = loss_c / (4 * d * d)
    # loss_mu = torch.sum(torch.mul((source_mu - target_mu), (source_mu - target_mu)))
    # loss_mu = loss_mu / (2*d)
    loss_mu = mse(source_mu, target_mu)
    #print(loss_c, loss_mu)
    return loss_c #+ 5 * loss_mu #loss_c + 5 * loss_mu # loss_mu  # loss_c #+ loss_mu

def coral_m(source, target):

    d = source.size(1)  # dim vector

    source_c, source_mu = compute_covariance3(source)
    target_c, target_mu = compute_covariance3(target)
    print('AAAAAAAAAAAAAAAAAAAAA') 
    print(source_c, target_c)
    

    loss_c = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss_c = loss_c / (4 * d * d)
    # loss_mu = torch.sum(torch.mul((source_mu - target_mu), (source_mu - target_mu)))
    # loss_mu = loss_mu / (2*d)
    loss_mu = mse(source_mu, target_mu)
    #print('mu', loss_mu, loss_c)
    #print('mumumucococo:', 1000*loss_c, 2000*loss_mu)
    return loss_c #+ 2 * loss_mu # loss_c #+ loss_mu



###
def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
    #c = d_t_d / (n-1)

    return c, mean_column

######

def compute_covariance3(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    id_col = torch.ones(n).resize(n,1).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    #mean_tom = torch.mm(id_col, mean_column)
    #centered_data = torch.add(input_data, (-1 * mean_tom))
    centered_data = input_data - mean_column
    c= torch.mm(centered_data.t(), centered_data) * 1 / (n - 1)
    

    return c, mean_column

######

# def log_var(input_data):
#     n = input_data.size(0)  # batch_size
#     d = input_data.size(1)

#     # Check if using gpu or cpu
#     if input_data.is_cuda:
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
    
#     var = torch.var(input_data, dim=0)
#     # var = var -1 
#     # var = var ** 2
#     # log_var = sum(torch.log(var+1))

#     input_data = F.normalize(input_data, p=2, dim=1)
#     covar = torch.cov(input_data.T)
#     det_covar = torch.linalg.det(covar)
#     log_det_covar = torch.log10(det_covar / d**2 + 1)
#     return log_det_covar



def log_var(feat_A: torch.Tensor):
    # Ensure input tensor is float32, since determinant requires float32 for computation
    feat_A = feat_A.float()

    # Compute covariance matrix (assuming feat_A is of shape [N, D], where N is number of samples and D is dimensionality)
    covar = torch.cov(feat_A.T)

    # Calculate the determinant of the covariance matrix in float32
    det_covar = torch.linalg.det(covar.float())  # Convert to float32 before determinant calculation

    return det_covar



def compute_covariance2(data):
    mean_vector = torch.mean(data, dim=0)
    # Subtract the mean from each data point
    centered_data = data - mean_vector
    # Calculate the covariance matrix
    covariance_matrix = torch.matmul(centered_data.T, centered_data) / (data.shape[0] - 1)
    return covariance_matrix, mean_vector



if __name__ == '__main__':
    a = torch.randn((48, 4))
    b = torch.randn((48, 4))
    loss = coral(a,b)
    c, _ = compute_covariance(a)
    d = log_var(a)
    print('log_var:', c.shape, c)
    print('variance:', d)
    print(loss)


    e, f = compute_covariance2(a)
    g, h = compute_covariance3(a)
    # loss1 = coral(a, b)
    # loss2 = coral2(a, b)
    print('e:', e-g, f-h)
    #print('g:', loss2)

