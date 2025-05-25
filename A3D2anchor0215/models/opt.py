import torch
import cvxpy as cp
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def coff(x_tr, y_tr, x_te, nc=4):
    a1 = torch.zeros(nc, x_tr.shape[1])
    for i in range(0, nc):
        index = (y_tr==i).nonzero().squeeze()
        x_sub = x_tr[index]
        x_mean = x_sub.mean(dim=0)
        a1[i,:]=x_mean
    
    a2 = x_te.mean(dim=0)
    return a1, a2


#def pte(x_tr, y_tr, x_te, nc=4):
def pte(a1, a2, nc=4):
    #a1, a2 = coff(x_tr, y_tr, x_te, nc=4)

    x = cp.Variable(nc) #.to(device)
    f = 0
    for i in range(0, nc):
        f += a1[i] * x[i] 
    f = f - a2 
    F = sum(f**2)
    prob = cp.Problem(cp.Minimize(F),
    [x >= 0,
     #x <= 1,
     sum(x) == 1])
    prob.solve()
    return(x.value)
    



def pte_qp(P, nc=4):
    #a1, a2 = coff(x_tr, y_tr, x_te, nc=4)

    x = cp.Variable(nc) #.to(device)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P)),
                    [x >= 0,
                    sum(x) == 1])
    prob.solve()
    return(x.value)





if __name__ == '__main__':
    nc = 4
    x_tr = torch.randn((128, 128))
    x_te = torch.randn((64, 128))
    y_tr = 32*[0] + 32*[2] + 32*[1] + 32 * [3]
    y_tr=torch.tensor(y_tr)

    a1, a2 = coff(x_tr, y_tr, x_te, nc=4)
    print(a1.shape)
    print('AAAAAAAAAAAAAAAAAA')
    print(a2.shape)

    #print(a1, a2)

    sol = pte(a1, a2, 4)
    print(sol)