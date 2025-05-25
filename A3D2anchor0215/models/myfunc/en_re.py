import torch

def entropy_re(A):
    variance_A = torch.var(A, dim=0, unbiased=False)

    log_variance_A = torch.log(variance_A + 1)
    # Calculate the sum of the log-transformed variances
    sum_log_variance_A = torch.sum(log_variance_A)
    #print("EEEEEEEEEEEE:", sum_log_variance_A)
    
    return sum_log_variance_A

if __name__ == '__main__':
    A = torch.randn(4, 6)
    loss = entropy_re(A)
    print('AAAAAAAAAAA:', loss)
