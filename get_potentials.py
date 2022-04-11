from sklearn.covariance import log_likelihood
from utils import load_data, safe_log, plot_trajectory, replace_zeros, get_sampler, show_barycenter, show_barycenters, get_sampler, norm_sq
import torch
import time
import numpy as np
import itertools
import ot
import scipy.stats
import time
import gc

# if torch.cuda.is_available():
#     device=torch.device("cuda")
# Dim=8
# start=(int(Dim/2)-1)*Dim
# finish=(int(Dim/2)+1)*Dim
# """
# Defining the cost function
# """
# Partition = torch.linspace(0, 1, Dim)
# couples = np.array(np.meshgrid(Partition, Partition)).T.reshape(-1, 2)
# x=np.array(list(itertools.product(couples, repeat=2)))
# x = torch.from_numpy(x)
# a = x[:, 0]
# b = x[:, 1]
# C = torch.linalg.norm(a - b, axis=1) ** 2
# C=C.to(device)
#
# A1=torch.cat((torch.ones(1), torch.zeros(Dim**2-1)), 0)
# A1= (A1 / torch.sum(A1)).to(device)
# A2=torch.cat((torch.zeros(Dim**2-Dim), torch.ones(Dim)), 0)
# A2= (A2 / torch.sum(A2)).to(device)
# Archetypes = torch.stack([A1, A2], dim=0).to(device)
#
# NumberOfAtoms= Dim**2
# C=C.reshape(NumberOfAtoms,NumberOfAtoms)
#
# _ , Archetypes= load_data(5, 0, 3, device, noise=None)



def SampleGeneration(samplesize,MeanMatrix,CovMatrix,device):
    Distribution=torch.distributions.multivariate_normal.MultivariateNormal(MeanMatrix,covariance_matrix=CovMatrix)
    data= Distribution.sample((samplesize,)).to(device)
    return data

def Transformation(cost,sample, device):
    if sample.dim()==2:
        lamext=sample.reshape(len(sample),len(sample[0]),1).expand(len(sample),len(sample[0]),len(sample[0])).transpose(2,1)
        lamstar=(cost-lamext).amin(dim=2)
    else:
        lamext=sample.reshape(len(sample),len(sample[0]),len(sample[0,1]),1).expand(len(sample),len(sample[0]),len(sample[0,1]),len(sample[0,1])).transpose(3,2)
        lamstar=(cost-lamext).amin(dim=3)
    del lamext
    if device == 'cuda':
        torch.cuda.empty_cache()
    return (lamstar)

def DSum(mus, nus , lam1, lam2):
    if lam1.dim()==1:
         estimation=torch.sum(mus*lam1) + torch.sum(nus*lam2)
    else:
        estimation=torch.sum((mus*lam1),dim=1) + torch.sum((nus*lam2),dim=1)
    return(estimation)



def algorithm(mean, cov, samplesize, n_steps, Data, cost,  device ,constant):
    # Dim = cost.shape[0]
    Mean=mean
    Cov=cov
    # RealBarycenter=ot.barycenter(replace_zeros(Data.clone()).T.contiguous(), cost, 0.005, numItermax=20000)
    Dataext=torch.broadcast_to(Data,(samplesize,len(Data),len(Data[0])))
    for k in range(1,n_steps):
        Sample=SampleGeneration(samplesize,Mean,Cov,device)
        ShapedSample=Sample.reshape(samplesize,len(Data),len(Data[0]))
        PotDual=Transformation(cost,ShapedSample, device)
        FirstTerm= torch.sum(torch.sum((Dataext*ShapedSample),dim=2),dim=1)
        SecondTerm=-torch.logsumexp(-constant*(torch.sum(PotDual,dim=1)),dim=1)/constant
        loglikelihood=FirstTerm+SecondTerm
        Weights = torch.softmax(80*loglikelihood,dim=0)
        Mean   = Weights @Sample
        Cov=1/k*cov
        ShapedMean=Mean.reshape(len(Data),len(Data[0]))
        MeanDual=Transformation(cost,ShapedMean, device)
        # Barycenter=torch.softmax(-constant*((torch.sum(MeanDual,dim=0))),dim=0)
        if k % 300 == 1:
            print(f"Step {k}/{n_steps-1}")
        #     titles = ['Barycenter (Sinkhorn)', 'Barycenter from potentials']
        #     show_barycenters([RealBarycenter, Barycenter], Dim, 'bary_from_poten', use_softmax=False, iterations=titles, use_default_folder=False)
    return MeanDual


        
        

# Mean =  torch.ones(len(Archetypes)*NumberOfAtoms,dtype=torch.float64).to(device)
# Covariance  = torch.eye(len(Archetypes)*NumberOfAtoms,dtype=torch.float64).to(device)
# print(algorithm(Mean, Covariance, 10000, 5000, Archetypes, C, device,40))



