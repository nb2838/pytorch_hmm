import torch
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet

class HMM:
    def __init__(self, emission, hidden_dim, n, emission_dim=1, A=None,  pi=None):
        """
        Arguments: 
        - hidden_dim: dimension of n
        - n: length of the timeseries 
        - A: matrix of transition probabilities
        - emission: An object representing emision distributions
        """

        # Deal with dimmensions
        self.n = n
        self.hidden_dim = hidden_dim
        self.emission_dim = emission_dim

        # Parameters of the model 
        self.A = self.init_A(A)
        self.pi = self.init_pi(pi)

        # Additional stuff 
        self._log_likelihood = None 
        self.emission = emission
        
        assert self.A.shape == (hidden_dim, hidden_dim)
        assert self.pi.shape == (1, hidden_dim)

    
    def sample(self): 
        """
        Generates data from the hmm  and returns a tuple with 
        latent and observed variables 
        """
        z = [Categorical(self.pi).sample().item()]
        for i in range(1, self.n):
            z.append(Categorical(self.A[z[-1]]).sample().item())

        x = []
        for i in range(len(z)):
            x.append(self.emission.sample(z[i]))
            
        return torch.tensor(z), torch.tensor([x]).T

    def init_A(self, A):
        if A == None:
            dirichlet = Dirichlet(torch.ones(self.hidden_dim, self.hidden_dim)*5)
            A = dirichlet.sample()
        return A

    def init_pi(self, pi):
        if pi == None:
            dirichlet = Dirichlet(torch.ones(1, self.hidden_dim) * 5)    
            pi = dirichlet.sample()
        return pi

    @property
    def log_likelihood(self):
        if not self._log_likelihood:
            raise UnboundLocalError('The E step has to be called at least once')
        else:
            return self._log_likelihood

    def calculate_local_likelihoods(self,obs):
        """
        Calculates the likelihoods p(x_n| z_n) and stores them in 
        self.likelihoods. 

        self.likelihoods is a list of vectors of shape (hidden_dim,) 
        where self.likelihoods[n][i] = p(x_n| z_n = i) 
        """
        likelihoods = []
        for i in range(len(obs)):
            likelihood = []
            for j in range(self.hidden_dim):
                likelihood.append(self.emission.prob(j, obs[i]))
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
        # TODO: Find a nicerway of dealing with loglikelihoods 
        alphas = []
        c = []
        
        self._log_likelihood  = 0 
        for i in range(self.n):
            if i == 0: 
                alpha = likelihoods[0] * self.pi[0]
                c.append(torch.sum(alpha))
                alphas.append(alpha/c[-1])
            else:
                alpha = likelihoods[i] * torch.einsum('i,ij->j',alphas[-1], self.A)
                c.append(torch.sum(alpha))
                alphas.append(alpha/c[-1])

            self._log_likelihood += torch.log(c[-1]) 
            assert alphas[-1].shape == (self.hidden_dim,) 
        assert len(c) == self.n
        # Now we calculate the betas
        betas = [torch.ones(self.hidden_dim)]
        for i in range(self.n - 1, 0, -1):
            beta =  torch.einsum('j,ij->i',likelihoods[i] *betas[-1], self.A)
            betas.append(beta/c[i])
            assert beta.shape  == (self.hidden_dim,) 

        betas.reverse()
        # Finally we return the suffcient stattistics 
        gammas = [torch.unsqueeze(alphas[i] * betas[i],0) for i in range(self.n)]
        chis = []
        for i in range(self.n-1):
            result = likelihoods[i+1] * betas[i+1]
            result = torch.outer(alphas[i], result)
            chis.append(torch.unsqueeze(result * self.A / c[i+1],0))

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
        self.sigma = w_obs_sub_mu @ obs_sub_mu/ torch.unsqueeze(torch.unsqueeze(summed_gammas,-1),-1)
        assert self.sigma.shape == (self.hidden_dim, self.emission_dim, self.emission_dim)

        
    def viterbi(self, obs):
        likelihoods = self.calculate_local_likelihoods(obs)
        omegas = [torch.log(self.pi[0]) + torch.log(likelihoods[0])]
        paths = []

        for i in range(1, self.n):
            x = torch.log(self.A) + torch.unsqueeze(omegas[-1], -1)
            vals, index = torch.max(x, 0)
            omegas.append(vals + torch.log(likelihoods[i]))
            paths.append(index)
            
        max_log_prob = torch.max(omegas[-1])
        likeley_states = [torch.argmax(omegas[-1])]

        paths.reverse()
        for step in paths:
            likeley_states.append(step[likeley_states[-1]])

        likeley_states.reverse()
        likeley_states = [x.numpy() for x in likeley_states]
        return max_log_prob, likeley_states
 
        
    def fit(self, obs, eps):
        """
        Performs MLE estimation on the paramaters of the HMM
        """

        self.emission.init_params(obs)
        
        old_log_likelihood = -float('INF')
        is_improvement = True
        step = 0
        
        while is_improvement:
            gammas, chi = self.E_step(obs)
            self.M_step(obs, gammas, chi)

            is_improvement = self.log_likelihood - old_log_likelihood > eps
            old_log_likelihood = self.log_likelihood
            step += 1
            print(f'Step {step} - log_likelihood {self.log_likelihood.item()}')

    
            
        
        
        
                
    
        
from emissions import NormalEmission

hidden_dim = 2
emission_dim = 1

n = 3000

mu = torch.unsqueeze(torch.arange(hidden_dim) * 10.0,-1)
sigma = torch.ones(hidden_dim,1,1)* 1.0
A = torch.tensor([[0.9, 0.1],[0.1,0.9]])

emission = NormalEmission(hidden_dim=hidden_dim, emission_dim=emission_dim, mu=mu, sigma=sigma)
hmm = HMM(n= n, emission=emission, hidden_dim=hidden_dim, A= A)
pi = hmm.pi 
z, x = hmm.sample()
# hmm.E_step(x)

# print('log_prob', hmm.log_likelihood)


from hmmlearn import hmm as hmms
model = hmms.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, verbose=2,algorithm='viterbi')
model.fit(x)
log_prob, za_pred = model.decode(x, algorithm='viterbi')
print('log_prob', log_prob)

# model.startprob_ = pi[0].numpy()
# model.means_ = hmm.mu.numpy()
# model.covars_ = hmm.sigma.numpy()
# model.transmat_ = hmm.A.numpy()
# print(model._score(x.numpy(), compute_posteriors=True))




hmm = HMM(n= n,hidden_dim=hidden_dim, emission=emission)
print('A', hmm.A)
print('sigma', sigma) 

hmm.fit(x, 0.0001)
log_prob, z_pred = hmm.viterbi(x)
print('log_prob', log_prob)


print(A)
print(hmm.A)

print(hmm.mu)


###############################

import matplotlib.pyplot as plt
from plot import plot_state_diagram

print(x)
plot_state_diagram(z_pred,x.numpy())
plot_state_diagram(za_pred,x.numpy())

