# Code from Misiakos et. al., ICASSP 2024
#################################################
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

from experiments import utils

def permutation_matrices(T, d):
    I = torch.eye(T - 1)
    P = torch.zeros((T, T))
    Q = torch.zeros((T, T)) 

    R = torch.roll(I, -1, 1)
    P[:-1,:-1] = R
    Q[1:,1:] = R.T
    I = torch.eye(d)
    P = torch.kron(P, I)
    Q = torch.kron(Q, I)
    # plt.figure()
    # plt.imshow(P)
    # plt.savefig("P.pdf")
    # plt.figure()
    # plt.imshow(Q)
    # plt.savefig("Q.pdf")
    return P.to(device), Q.to(device)

class SparseRC(nn.Module):
    def __init__(self, X, lambda1, lambda2, constraint='notears', omega=0.3, fix_sup='False', T=10):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.X = X.clone().detach()
        self.d = self.X.shape[-1]
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.constraint = constraint
        self.omega = omega
        self.T = T
        self.fix_sup = fix_sup
        self.fc = torch.nn.Linear(self.d, self.d, bias=False) # input x : output (A, ) A^Tx + b
        # self.weight = nn.Parameter(torch.empty((1, self.d), **factory_kwargs))
        # self.weight = torch.ones(self.d) 
        # self.weight[:int(0.3 * self.d)] = 0.01 # assuming early/late causes and sparsity 30%
        # self.reset_parameters()
        print(T, int(self.d / T))
        self.P, self.Q = permutation_matrices(T, int(self.d / T))

    def reset_parameters(self) -> None:
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.weight, 1)

    def postprocess_A(self):
        A = self.fc.weight.T
        A_est = torch.where(torch.abs(A) > self.omega, A, 0)
        return A_est.detach().cpu().numpy()
    
    def l1_reg(self):
        A = self.fc.weight
        return torch.sum(torch.abs(A)) # 
    
    def acyclicity(self):
        A = self.fc.weight
        if self.constraint == 'notears':
            assert A.shape[0] == self.d
            return torch.trace(torch.matrix_exp(A * A)) - self.d
        elif self.constraint == 'dag-gnn':
            M = torch.eye(self.d) + A * A / self.d  # (Yu et al. 2019)
            return  torch.trace(torch.linalg.matrix_power(M, self.d)) - self.d
        elif self.constraint == 'frobenius':
            return torch.sum((A * A.T) ** 2)
        
    def constant_topology(self):
        A = self.fc.weight.T
        # return torch.sum((A - self.P @ A @ self.Q) ** 2)
        # print(self.P.shape, self.Q.shape, A.shape)
        return torch.sum(torch.abs(A - self.P @ A @ self.Q))
    
    def upper_block_diagonal(self):
        A = self.fc.weight.T
        d = int(A.shape[0] / self.T)
        I = torch.eye(d)
        J = torch.eye(self.T)
        J_shift = torch.roll(J, 1, 1)
        J_shift[-1:, 0]= 0
        I_block = torch.kron(J_shift, I)
        return torch.sum((A - I_block) ** 2)        
    
    def toeplitz_loss(self):
        A = self.fc.weight.T
        n, m = A.shape
        t_zero = torch.tensor([[0]], device=device)
        A_ = torch.cat([torch.cat([A, torch.zeros((n,1))], dim=1),
                        torch.cat([torch.zeros((1,m), device=device), t_zero], dim=1)], dim=0
                        )
        C = torch.eye(n + 1)
        C = torch.roll(C, 1, 0)
        C[0, -1] = 0

        return torch.sum((A_ - C @ A_ @ C.T) ** 2)  

    def forward(self, X):
        return self.fc(X) # output is XA
        


def sparserc_solver(X, lambda1, lambda2, lambda3=0, epochs=3000, constraint="notears", omega=0.3, T=10):
    '''
        sparserc solver
        params:
        X: data (np.array) of size n x d
        lambda1: coefficient (double) for l1 regularization λ||Α||_1
        lambda2: coefficient (double) for the graph constraint 
        epochs: upper bound for the number of iterations of the optimization solver.
    '''
    X = torch.tensor(X, device=device, dtype=dtype)
    fix_sup = False

    N = X.shape[0]
    print(X.shape)
    model = SparseRC(X, lambda1=lambda1, lambda2=lambda2, constraint=constraint, omega=omega, fix_sup=fix_sup, T=T)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    early_stop = 40
    best_loss = 100000000000000000

    for i in range(epochs):

        def closure():
            # nonlocal so that we can alter their value
            nonlocal best_loss, early_stop

            # zero gradients and compute output = XA
            optimizer.zero_grad()
            output = model(X)

            # compute total optimization loss and back-propagate
            if fix_sup:
                loss = (1 / (2 * N)) * torch.norm((X - output) * model.weight, p=1)   # (1/2n) * |X-XA|_1
            else:
                loss = (1 / (2 * N)) * torch.norm((X - output), p=1)   # (1/2n) * |X-XA|_1
            loss = loss + lambda1 * model.l1_reg() + lambda2 * model.acyclicity() + lambda3 * model.constant_topology() # (1/2n) * |X-XA|_1  + λ1 * |A| + λ2 *  h(A)
            loss.backward()

            # overview of performance
            if i % 10 == 0:
                print("Epoch: {}. Loss={:.3f}, Sparsity={:.3f}, Acyclicity={:.3f}, Toeplitz={:.3f}, Total = {:.3f}"\
                      .format(i, (1 / (2 * N * T)) * torch.norm((X - output), p=1), lambda1 * model.l1_reg(), lambda2 * model.acyclicity(), lambda3 * model.constant_topology(), loss.item()))
            
            # early stopping 
            if loss.item() >= best_loss:
                early_stop -= 1
            else:
                early_stop = 40
                best_loss = loss.item()
                torch.save(model.state_dict(), 'results/best_model.pl')

            return loss
    
        optimizer.step(closure)

        if early_stop == 0:
            break

    # threshold values with absolute values < 0.3
    model = SparseRC(X, lambda1=lambda1, lambda2=lambda2, constraint=constraint, omega=omega, fix_sup=fix_sup, T=T)
    model.load_state_dict(torch.load('results/best_model.pl'))
    A = model.postprocess_A()
    
    return A


# Code from Misiakos et. al., NeurIPS 2023
#################################################
class SparseRC_weightfinder(nn.Module):
    def __init__(self, X, A):
        super().__init__()
        self.X = X.clone().detach()
        self.d = self.X.shape[1]
        self.mask = A.clone().detach()
        self.fc = torch.nn.Linear(self.d, self.d, bias=False) # input x : output (A, ) A^Tx + b

    def postprocess_A(self):
        A = self.fc.weight.T * self.mask
        return A.detach().cpu().numpy()
        
    def forward(self, X):
        return X @ (self.fc.weight.T * self.mask) # output is X (A o M)
        

def sparserc_solver_weight_finder(X, A, epochs=3000):
    '''
        sparserc solver
        params:
        X: data (np.array) of size n x d
        lambda1: coefficient (double) for l1 regularization λ||Α||_1
        lambda2: coefficient (double) for the graph constraint 
        epochs: upper bound for the number of iterations of the optimization solver.
    '''
    X = torch.tensor(X, device=device, dtype=torch.float32)
    A = torch.tensor(A, device=device, dtype=torch.float32)
    N = X.shape[0]

    model = SparseRC_weightfinder(X, A)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    early_stop = 40
    best_loss = 10000000000000000

    for i in range(epochs):

        def closure():
            # nonlocal so that we can alter their value
            nonlocal best_loss, early_stop

            # zero gradients and compute output = XA
            optimizer.zero_grad()
            output = model(X)

            # compute total optimization loss and back-propagate
            loss = ( 1 / (2 * N) ) * torch.norm((X - output), p=1)   # (1/2n) * |X-XA|_1
            loss.backward()

            # overview of performance
            # if i % 10 == 0:
            #     print("Epoch: {}. Loss = {:.3f}".format(i, loss.item()))
            
            # early stopping 
            if loss.item() >= best_loss:
                early_stop -= 1
            else:
                early_stop = 40
                best_loss = loss.item()
                torch.save(model.state_dict(), 'results/best_model.pl')

            return loss
    
        optimizer.step(closure)

        if early_stop == 0:
            break

    # threshold values with absolute values < 0.3
    model = SparseRC_weightfinder(X, A)
    model.load_state_dict(torch.load('results/best_model.pl'))
    A = model.postprocess_A()
    
    return A


if __name__ == "__main__":
    # test for constant signals
    import numpy as np
    import matplotlib.pyplot as plt
    n = 1000
    d = 20
    T = 100
    p = 2
    X = np.ones((n, T * d))          

    W = sparserc_solver(X, lambda1=0.001, lambda2=1., epochs=10000, time_lag=p, omega=0.1, T=T)
    print(W.shape)
    W_est = W[:d, :(p + 1) * d]
    B_est = W_est != 0

    print(B_est)
    plt.figure()
    plt.imshow(B_est)
    plt.show()

