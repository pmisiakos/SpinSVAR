import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

from experiments import utils

class SpinSVAR(nn.Module):
    """
        Sparse root causes model for time-series SVAR data.
    """
    def __init__(self, X, lambda1, lambda2, time_lag=3, constraint='notears', omega=0.3, T=10):
        """
        Args:
            X (n x dT): Time series data tensor.
            lambda1: l1 adjacency sparsity coefficient.
            lambda2: acyclicity coefficient.
            time_lag: data at timestep t depend at most from data at time t - time_lag.
            constraint: acyclicity constraint to choose.
            omega: final thresholding of the weights of the output adjacency matrix.
            T: number of timesteps
        """

        super().__init__()
        self.X = X.clone().detach() # data
        # self.d = int(self.X.shape[1] / T) # extracting the number of nodes of the DAG
        # self.n = X.shape[0] # number of realizations of the time-series.
        self.n, self.T, self.d = torch.tensor(X.shape).to(device)
        self.eye = torch.eye(self.d).to(device)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = time_lag
        self.constraint = constraint
        self.omega = omega
        self.T = T 

        # Fully connected linear layer without bias term represents the p + 1 adjacency matrices that we are looking for. 
        # the linear layer weight matrix has size d x d(p + 1) and the matrices are in reverse order and transposed: [B_p.T, B_{p-1}.T,...,B_1.T, A.T]
        # this order is used to optimize the implementation
        self.fc = torch.nn.Linear(self.d * (self.p + 1), self.d, bias=False)

        # X_past[t] contains X[t - p] X[t - p + 1]  X[t - 1] X[t] in that order, where X[t] (shape = n x d) denotes the data on the DAG at time t 
        # self.X_past = torch.zeros((self.T, X.shape[0], (self.p + 1) * self.d), device=device)
        # for t in range(self.T):
        #     if t < self.p :
        #         self.X_past[t] = torch.cat([torch.zeros((X.shape[0], (self.p - t) * self.d), device=device), X[:, :(t + 1) * self.d]], dim=1)
        #     else:
        #         self.X_past[t] = X[:, (t - self.p) * self.d: (t + 1) * self.d] 

        self.X_past = torch.tensor(utils.X_past(X, self.p, device), device=device, dtype=dtype)

    def postprocess_A(self):
        '''Thresholds and returns the window graph:
            After the algorithm has converged, 
            this method is used to transform the linear layer coeffients into the window graph.

            Output:
            A (np.ndarray): [d, d * (time_lag + 1)] window graph.
            The parameters are : [B_0, B_1,...,B_{p-1}, B_p] in that order.
        '''
        A = self.fc.weight
        A_est = torch.where(torch.abs(A) > self.omega, A, 0) # thresholding
        res = torch.zeros(A_est.shape)
        # reversing the order and transposing
        for i in range(self.p + 1):
            res[:, i * self.d: (i + 1) * self.d] = A_est[:, (self.p - i) * self.d : (self.p + 1 - i) * self.d].T
        return res.detach().cpu().numpy()
    

    def l1_reg(self):
        A = self.fc.weight
        return torch.sum(torch.abs(A)) 

    def logdet(self):
        A = self.fc.weight[:self.d, self.d * self.p:] # this is B_0
        # return torch.log(torch.abs(torch.linalg.det(torch.eye(self.d) - A)))
        return torch.abs(torch.linalg.det(self.eye - A)) 
    
    def no_self_loops(self):
        A = self.fc.weight[:self.d, self.d * self.p:] # this is B_0
        return torch.sum(torch.abs(torch.eye(self.d) * A) ** 2)
    
    def acyclicity(self):
        A = self.fc.weight[:self.d, self.d * self.p:] # this is B_0
        if self.constraint == 'notears':
            assert A.shape[0] == self.d
            return torch.trace(torch.matrix_exp(A * A)) - self.d
        elif self.constraint == 'dag-gnn':
            M = torch.eye(self.d) + A * A / self.d  # (Yu et al. 2019)
            return  torch.trace(torch.linalg.matrix_power(M, self.d)) - self.d
        elif self.constraint == 'frobenius':
            return torch.sum((A * A.T) ** 2)
        
    def forward(self):
        # res = torch.zeros(X.shape, device=device)
        # for t in range(self.T):
            # res[:,t * self.d: (t + 1) * self.d] = self.fc(self.X_past[t])
        res = self.fc(self.X_past) 
        # res.swapaxes(0, 1)
        # return  res.swapaxes(0, 1).reshape(self.n, self.d * self.T) #self.fc(X) # output is XA
        return res
        
def spinsvar_solver(X, lambda1, lambda2, epochs=3000, time_lag=3, constraint="notears", omega=0.3, T=10):
    '''
        SpinSVAR solver
        Input:
            X (n x dT): Time series data tensor.
            lambda1: l1 adjacency sparsity coefficient.
            lambda2: acyclicity coefficient.
            epochs: upper bound for the number of iterations of the optimization solver.
            time_lag: data at timestep t depend at most from data at time t - time_lag.
            constraint: acyclicity constraint to choose.
            omega: final thresholding of the weights of the output adjacency matrix.
            T: number of timesteps
        Output:
            A (np.ndarray): [d, d * (time_lag + 1)] weighted window graph.
            The parameters are : [B_0, B_1,...,B_{p-1}, B_p] in that order.
    '''
    X = torch.tensor(X, device=device, dtype=dtype)

    N = X.shape[0]

    model = SpinSVAR(X, lambda1=lambda1, lambda2=lambda2, time_lag=time_lag, constraint=constraint, omega=omega, T=T)
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
            output = model()

            # compute total optimization loss and back-propagate
            # loss1 = (1 / (2 * N * T)) *  (torch.norm((X - output), p=1) / model.logdet())   # (1/2n) * |X-XA|_1
            # loss1 = model.d * torch.log(torch.norm((X - output), p=1)) - torch.log(model.logdet())
            # loss1 = (1 / (2 * N * T)) * (model.d * torch.log(torch.norm((X - output), p=1)) - torch.log(model.logdet()))
            loss1 = torch.log(torch.norm((X - output), p=1)) - (1 / model.d) * torch.log(model.logdet())
            loss = N *  loss1 + lambda1 * model.l1_reg() + lambda2 * model.acyclicity() # (1 / (2 * N * T)) * loss for gaussian input + 10 * model.no_self_loops()
            loss.backward()

            # overview of performance
            if i % 10 == 0:
                # print("Epoch: {}. Regression Loss={:.3f}, Sparsity={:.3f}, Acyclicity={:.3f}, Total = {:.3f}".format(i, (1 / (2 * N * T)) *  torch.norm((X - output), p=1), lambda1 * model.l1_reg(),lambda2 * model.acyclicity(), loss.item()))
                # print("Epoch: {}. Regression Loss={:.3f}, Det(I-B_0)={:.3f} Sparsity={:.3f}, Acyclicity={:.3f}, Total = {:.3f}".format(i, (1 / (2 * N * T)) *  torch.norm((X - output), p=1), (1 / (2 * N * T)) *  model.logdet(), lambda1 * model.l1_reg(),lambda2 * model.acyclicity(), loss.item()))
                print("Epoch: {}. Regression Loss={:.3f}, LogDet(I-B_0)={:.3f} Sparsity={:.3f}, Acyclicity={:.3f}, Total = {:.3f}".format(i, model.d * torch.log(torch.norm((X - output), p=1)), torch.log(model.logdet()), lambda1 * model.l1_reg(),lambda2 * model.acyclicity(), loss.item()))

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

    model = SpinSVAR(X, lambda1=lambda1, lambda2=lambda2, time_lag=time_lag, constraint=constraint, omega=omega, T=T)
    model.load_state_dict(torch.load('results/best_model.pl'))
    A = model.postprocess_A()
    
    return A

