import numpy as np


# linear least-squares fit
def lstsqfit(Y, A, C):
    # we want to solve Y = A @ X, weighted by the covariance
    a = A.T @ np.linalg.solve(C, A)
    b = A.T @ np.linalg.solve(C, Y)

    # a X = b
    theta = np.linalg.solve(a, b)
    cov = np.linalg.inv(a)

    return theta, cov


# compute log likelihood function
def get_loglike(y, mu, sigmas):
    res = y - mu
    chi_sq = np.sum(res**2 / sigmas**2)
    return -0.5 * chi_sq



### PERIODIC SIGNALS ###

# one periodic signal
def func_1sig(t, T, B, A1, A2):
    y = B + A1*np.cos(2*np.pi*t/T)+A2*np.sin(2*np.pi*t/T)
    return y


# two periodic signals, one of which has known period
def func_2sigs(t, T, B, A1, A2, A3, A4, T0=100):
    y = B + A1*np.cos(2*np.pi*t/T0) + A2*np.sin(2*np.pi*t/T0) + A3*np.cos(2*np.pi*t/T) + A4*np.sin(2*np.pi*t/T)
    return y


class Signal:

    def __init__(self, time, signal, sigmas):
        self.time = time
        self.signal = signal
        self.sigmas = sigmas
    
    def fit_1sig(self, T):
        # construct the matrices
        Y = self.signal
        A = np.column_stack((np.ones(len(self.time)), np.cos(2*np.pi*self.time/T), np.sin(2*np.pi*self.time/T)))
        C = np.diag(self.sigmas**2)

        # we want to solve Y = A @ X, weighted by the covariance
        theta, cov = lstsqfit(Y, A, C)

        return theta, cov

    # linear least-squares fit to two periodic signals, one of which has a known period
    def fit_2sigs(self, T, T0=100):
        # construct the matrices
        Y = self.signal
        A = np.column_stack((np.ones(len(self.time)), np.cos(2*np.pi*self.time/T0), np.sin(2*np.pi*self.time/T0), np.cos(2*np.pi*self.time/T), np.sin(2*np.pi*self.time/T)))
        C = np.diag(self.sigmas**2)

        # we want to solve Y = A @ X, weighted by the covariance
        theta, cov = lstsqfit(Y, A, C)

        return theta, cov
    
    def find_second_period(self, T0=100, periods=np.logspace(1, 3, 100)):
        n = len(periods)

        # compute the log likelihood at each period
        loglikes = np.zeros(n)
        for i, T in enumerate(periods):
            theta, cov = self.fit_2sigs(T, T0=T0)
            fit = func_2sigs(self.time, T, *theta, T0=T0)
            loglikes[i] = get_loglike(self.signal, fit, self.sigmas)
        self.second_T_loglikes = loglikes

        # find the period which maximizes the log likelihood
        peak_idx = np.where(loglikes==max(loglikes))
        bestfit_period = periods[peak_idx][0]
        self.second_T = bestfit_period



### MCMC ###

# general function for a M-H MCMC sampler, given a sample distribution, proposal distribution, number of hops and initial state
def MCMC(n, x, sample_dist, prop_dist):
    
    dim = 1 if type(x)==int or type(x)==float else len(x)

    states = np.empty((n, dim))

    current = x

    for i in range(n):
        # proposal PDF
        proposal = prop_dist(current)

        # draw random number from a uniform distribution
        rand_val = np.random.uniform()

        # evaluate the function to be sampled at both current and proposed vals
        p_current = sample_dist(current)
        p_proposal = sample_dist(proposal)

        # do we accept the proposal?
        # avoid division by zero– if p_current is zero, we automatically reject the proposal
        if p_current != 0:
            ratio = p_proposal/p_current
            if ratio > rand_val:
                current = proposal
        states[i] = current
    
    return states


def mh_step(pars, lnpost, proposal):
    # draw from proposal distribution
    new = proposal(pars)
    # ratio between current parameters and proposed parameters in terms of posterior distribution
    lnratio = lnpost(new) - lnpost(pars)
    # random value to be compared
    randval = np.random.uniform()
    # if randval < ratio, move to the new parameters, otherwise keep the current
    if np.log(randval) < lnratio:
        return new
    else:
        return pars


def mh_mcmc(n, init_pars, lnpost, proposal):

    dim = 1 if type(init_pars)==int or type(init_pars)==float else len(init_pars)

    # empty array to hold the states
    states = np.empty((n, dim))

    current = init_pars

    for i in range(n):
        current = mh_step(current, lnpost, proposal)
        states[i] = current
    
    return states