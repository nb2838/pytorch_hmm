import torch
import torch.distributions as tdist
import pyro.distributions as dist
import pyro 
from torch.distributions.multivariate_normal import MultivariateNormal

class HMM:
    def __init__(self, hidden_dim, n, emission_dim=1, A=None, mu=None, sigma=None, pi=None):
        """
        Arguments: 
        - hidden_dim: dimension of n
        - n: length of the timeseries 
        - A: matrix of transition probabilities
        - mu: means for gaussian emissions (
        - sigma: covariance matrices for gaussian emissions
        """
        self.n = n
        self.hidden_dim = hidden_dim
        self.emission_dim = emission_dim
        self.A = self.init_A(A)
        self.mu = self.init_mu(mu)
        self.sigma = self.init_sigma(sigma)
        self.pi = self.init_pi(pi)
        
        self.emission = MultivariateNormal
        
        assert self.mu.shape == (hidden_dim, emission_dim)
        assert self.sigma.shape == (hidden_dim, emission_dim, emission_dim)
        assert self.A.shape == (hidden_dim, hidden_dim)
        assert self.pi.shape == (1, hidden_dim)

    
    def sample(self): 
        """
        Generates data from the hmm  and returns a tuple with 
        latent and observed variables 
        """
        z = [pyro.sample('z_0', dist.Categorical(self.pi)).item()]
        for i in range(1, self.n):
            z.append(
                pyro.sample(f'z_{i}',dist.Categorical(self.A[z[-1]])).item()
            )

        x = []
        for i in range(len(z)):
            x.append(
                pyro.sample(f'x_{i}', dist.Normal(self.mu[z[i]],self.sigma[z[i]])).item()
            )

        return torch.tensor(z), torch.tensor([x]).T

    def init_A(self, A):
        if A == None:
            dirichlet = tdist.dirichlet.Dirichlet(
                torch.eye(self.hidden_dim) * 0.9 + 0.1 
            )
            A = dirichlet.sample()

        return pyro.param('A', A)

    def init_mu(self, mu):
        if mu == None:
            normal = MultivariateNormal(torch.zeros(self.hidden_dim,self.emission_dim), 10)
            mu = normal.sample()

        return mu

    def init_sigma(self, sigma):
        if sigma == None:
            lkj = tdist.lkj_cholesky.LKJCholesky(self.hidden_dim)
            l = lkj.sample()
            corr = l @ l.T
            diag = torch.diag(torch.diagonal(cov))
            sigma = diag @ corr @ diag 
        return sigma 

    def init_pi(self, pi):
        if pi == None:
            dirichlet = tdist.dirichlet.Dirichlet(
                torch.ones(1, self.hidden_dim)
            )
            pi = dirichlet.sample()
        return pi



    def calculate_local_likelihoods(self,obs):
        """
        Calculates the likelihoods p(x_n, z_n) and stores them in 
        self.likelihoods. 

        self.likelihoods is a list of vectors of shape (hidden_dim,) 
        where self.likelihoods[n][i] = p(x_n| z_n = i) 
        """
        likelihoods = []
        for i in range(len(obs)):
            likelihood = []
            for j in range(self.hidden_dim):
                likelihood.append(
                    torch.exp(self.emission(self.mu[j], self.sigma[j]).log_prob(obs[i]))
                )
                
            likelihoods.append(torch.tensor(likelihood))
            assert likelihoods[-1].shape == (self.hidden_dim,)

        return likelihoods
            

    def E_step(self, obs):
        """
        Implementation of the E step of the EM algorithm to calculate posterior
        p(z_n|X), p(z_n, z_{n-1}|X)

        Implementation follows algorithm described in Pattern Recognition and 
        Machine Learning by Bishop. Assumes that we have called calculate_local_likelihoods
        """
        assert obs.shape == (self.n, self.emission_dim)

        likelihoods = self.calculate_local_likelihoods(obs)
        
        # First we will calculate the alphas
        alphas = []
        c = []
        for i in range(self.n):
            if i == 0: 
                alpha = likelihoods[0] * self.pi[0]
                c.append(torch.sum(alpha))
                alphas.append(alpha/c[-1])
            else:
                alpha = likelihoods[i] * torch.einsum('i,ij->j',alphas[-1], self.A)
                c.append(torch.sum(alpha))
                alphas.append(alpha/c[-1])
            assert alphas[-1].shape == (self.hidden_dim,) 
                
        # Now we calculate the betas
        betas = [torch.ones(self.hidden_dim)]
        for i in range(self.n - 1, -1, -1):
            beta = likelihoods[i] * torch.einsum('i,ij->j', betas[-1], self.A)
            betas.append(beta/c[i])
            assert beta.shape  == (self.hidden_dim,) 

        # Finally we return the suffcient stattistics 
        gammas = [torch.unsqueeze(alphas[i] * betas[self.n - i - 1],0) for i in range(self.n)]
        chis = []
        for i in range(self.n-1):
            result = likelihoods[i+1] * betas[self.n - i - 2]
            result = torch.einsum('i,j-> ij',alphas[i], result)
            chis.append(torch.unsqueeze(result * self.A * c[i+1],0))


        return torch.cat(gammas, 0), torch.cat(chis, 0)

    
    def M_step(self, obs, gammas, chis):
        """
        Does the M_Step of the maximization algorithm
        """
        assert gammas.shape == (self.n, self.hidden_dim)
        assert chis.shape == (self.n-1, self.hidden_dim, self.hidden_dim) 
        
        self.pi = gammas[:1]/torch.sum(gammas[0])
        self.A = torch.sum(chis, axis=0)
        self.A = self.A/torch.unsqueeze(torch.sum(self.A, axis=-1), -1) 


        # obs values weighted by gammas w_obs[n][k] corresponds to value n
        # weighted by gamma[n][k]
        summed_gammas = torch.sum(gammas, 0) 
        w_obs = torch.unsqueeze(obs, 1) * torch.unsqueeze(gammas, -1)
        self.mu = torch.sum(w_obs, 0)/torch.unsqueeze(summed_gammas, -1)
        assert self.mu.shape == (self.hidden_dim, self.emission_dim)


        w_mu = self.mu * torch.unsqueeze(gammas, -1)
        w_obs_sub_mu  = (w_obs - w_mu).permute(1,2,0)
        obs_sub_mu = (torch.unsqueeze(obs, 1) - self.mu).permute(1,0,2)
        self.sigma = w_obs_sub_mu @ obs_sub_mu
        assert self.sigma.shape == (self.hidden_dim, self.emission_dim, self.emission_dim)


        
        
        
        
        
                
    
        


hidden_dim = 5

mu = torch.unsqueeze(torch.arange(hidden_dim) * 10.0,-1)
sigma = torch.ones(hidden_dim,1,1)* 1.0 
hmm = HMM(5, 10, mu=mu, sigma=sigma)
z, x = hmm.sample()


gammas, chis = hmm.E_step(x)
hmm.M_step(x, gammas, chis)


# import matplotlib.pyplot as plt
# from plot import plot_state_diagram
# plot_state_diagram(z,x)

