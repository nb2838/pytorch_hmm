import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture

class NormalEmission:
    def __init__(self, hidden_dim=2, emission_dim=1, mu=None, sigma=None):
        """
        Arguments:
        - emission_dim: Dimension of the normal distribution.
        = hidden_dim: Dimension of the latent space
        - mu:  tensor of dimension [hidden_dim, emission_dim] representing means of emissions.
        - sigma: tensor of dimension [hidden_dim, emission_dim, emission_dim] representing 
          covariance matrices for each of the mixtures.
        """
        self.hidden_dim = hidden_dim
        self.emission_dim = emission_dim
        self.mu = mu
        self.sigma = sigma

    def sample(self, i):
        """
        Draws a sample from the ith mixture
        """
        return MultivariateNormal(self.mu[i], self.sigma[i]).sample()

    def prob(self, i, x):
        """
        Calculates the pdf evaluated at x using the ith mixture
        """
        d = MultivariateNormal(self.mu[i], self.sigma[i])
        return torch.exp(d.log_prob(x))

    def init_params(self, obs):
        """
        Initializes the parameters of the emissions using a Gaussian 
        mixture model. 
        """
        gmm = GaussianMixture(n_components=self.hidden_dim)
        gmm.fit(obs.numpy())
        self.mu = torch.tensor(gmm.means_).float()
        self.sigma = torch.tensor(gmm.covariances_).float()


    
