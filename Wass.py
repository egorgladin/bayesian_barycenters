import torch
import time
import numpy as np
import itertools
import ot
import scipy.stats
import time
import gc
if torch.cuda.is_available():
    device=torch.device("cuda")



def SampleGeneration(samplesize,MeanMatrix,CovMatrix,device):
    Distribution=torch.distributions.multivariate_normal.MultivariateNormal(MeanMatrix,covariance_matrix=CovMatrix)
    data= Distribution.sample((samplesize,)).to(device)
    return data


def Objective(mu, nu, cost, potsample):
    lam = potsample
    n_ = len(potsample[0])
    M = len(lam)

    lamext = lam.reshape(M, n_, 1).expand(M, n_, n_).transpose(2,1)
    lamstar = (cost-lamext).amin(dim=2)
    del lamext
    torch.cuda.empty_cache()

    lamstarext = lamstar.reshape(M, n_, 1).expand(M, n_, n_).transpose(2,1)
    lamstarstar = (cost-lamstarext).amin(dim=2)
    del lamstarext
    torch.cuda.empty_cache()

    mu = torch.broadcast_to(mu, (M, n_))
    nu = torch.broadcast_to(nu, (M, n_))
    estimation = torch.sum((mu*lamstarstar), dim=1) + torch.sum((nu*lamstar), dim=1)

    return estimation


def algorithm(potmean, potcov, potsamplesize, pot_n_steps, first, second, cost, objective, temperature, device):
    PotMean1=potmean
    PotMean2=potmean
    PotCov1=potcov
    PotCov2=potcov
    for k in range(pot_n_steps):    
        PotSample1 = torch.zeros(1*potsamplesize,len(potmean),dtype=torch.double).to(device)
        PotObjectiveValues1 = torch.zeros(1*potsamplesize,dtype=torch.double).to(device)
        for g in range(1):
            PotSample1[g*potsamplesize:(g+1)*potsamplesize,:] = SampleGeneration(potsamplesize, PotMean1, PotCov1,  device)
            PotObjectiveValues1[g*potsamplesize:(g+1)*potsamplesize] = objective(first, second, cost, PotSample1[g*potsamplesize:(g+1)*potsamplesize,:])  
        PotPreWeights1 = torch.exp(temperature*PotObjectiveValues1)
        PotWeights1=  PotPreWeights1/torch.sum(PotPreWeights1)    
        PotMean1   = PotWeights1 @ PotSample1
        PotCov1=0.002*potcov
        BarObjectiveValues1=objective(first,second,cost,PotMean1.expand(1,len(PotMean1)))
        PotMean1ext=PotMean1.reshape(len(PotMean1),1).expand(len(PotMean1),len(PotMean1)).transpose(1,0)
        PotMean2=(cost-PotMean1ext).amin(dim=1)
        PotSample2 = torch.zeros(1*potsamplesize,len(potmean),dtype=torch.double).to(device)
        PotObjectiveValues2 = torch.zeros(1*potsamplesize,dtype=torch.double).to(device)
        for g in range(1):
            PotSample2[g*potsamplesize:(g+1)*potsamplesize,:] = SampleGeneration(potsamplesize, PotMean2, PotCov2,  device)
            PotObjectiveValues2[g*potsamplesize:(g+1)*potsamplesize] = objective(second, first, cost, PotSample2[g*potsamplesize:(g+1)*potsamplesize,:])  
        PotPreWeights2 = torch.exp(temperature*PotObjectiveValues2)
        PotWeights2=  PotPreWeights2/torch.sum(PotPreWeights2)    
        PotMean2   = PotWeights2 @ PotSample2
        PotCov2=0.002*potcov
        BarObjectiveValues2=objective(second,first,cost,PotMean2.expand(1,len(PotMean2)))
        PotMean2ext=PotMean2.reshape(len(PotMean2),1).expand(len(PotMean2),len(PotMean2)).transpose(1,0)
        PotMean1=(cost-PotMean2ext).amin(dim=1)
    a=objective(first,second,cost,PotMean1.expand(1,len(PotMean1)))
    # print(a-ot.emd2(first,second,cost,numItermax=10000))
    return a, PotMean1



def distance(mu,nu, cost, device):   
    PotM =  torch.ones(len(mu),dtype=torch.double).to(device)
    PotC  =  torch.eye(len(mu),dtype=torch.double).to(device)
    return algorithm( PotM, PotC, 10000, 10, mu, nu, cost, Objective, 100, device)
 








