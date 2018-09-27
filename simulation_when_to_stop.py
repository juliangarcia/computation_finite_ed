from numpy.random import geometric
from numpy.random import rand
from numpy import exp
import sys
import numpy as np
from scipy.stats import entropy



def step_fast(pop_size, a, b, c, d, beta, type_a):
    w_a = exp(
        beta * (1.0 / (pop_size - 1)) * (a * (type_a - 1) + b * (pop_size - type_a)))
    w_b = exp(
        beta * (1.0 / (pop_size - 1)) * (c * type_a + d * (pop_size - type_a - 1)))
    mb = (type_a * w_a) / (type_a * w_a + (pop_size - type_a) * w_b)
    md = type_a / pop_size
    nothing_happens = mb * md + (1 - mb) * (1 - md)
    k = geometric(1.0 - nothing_happens)  # how many steps until something happens
    if rand() < w_a / (w_a + w_b): 
        type_a += 1
    else:
        type_a -= 1
    return k, type_a


#################### Fixation probability ####################

def sample_fixation_probability(pop_size, a, b, c, d, beta):
    type_a = 1
    
    while not (type_a == 0 or type_a == pop_size):
        (k, type_a) = step_fast(pop_size, a, b, c, d, beta, type_a)
        
    if type_a == 0:
        fixation = 0
        
    if type_a == pop_size:
        fixation = 1
            
    return fixation



def estimate_fixation_probability(pop_size, beta, a, b, c, d, epsilon, delta, r_0):
    assert 0 < epsilon < 1, "Epsilon must be between 0 and 1"

    sigma_squared = delta * epsilon*epsilon
    number_of_fixation_events = 0
    
    for i in range(r_0):
        number_of_fixation_events += sample_fixation_probability(pop_size, a, b, c, d, beta)
        
    number_of_trials = r_0
    X_k = number_of_fixation_events / number_of_trials
    criterion = X_k * (1 - X_k) / sigma_squared
    
    try:
        while (number_of_trials < criterion or criterion==0):
            number_of_fixation_events += sample_fixation_probability(pop_size, a, b, c, d, beta)
            number_of_trials += 1
            
            X_k = number_of_fixation_events / number_of_trials
            criterion = X_k * (1 - X_k) / sigma_squared
            
            
    except KeyboardInterrupt:
        print("Aborted with poor estimate {} and {} trials".format(X_k, number_of_trials))
    return X_k 

#################### Unconditional fixation time ####################

def sample_unconditional_fixation_time_fast(pop_size, a, b, c, d, beta):
    type_a = 1
    i = 0  # fixation time
    while not (type_a == 0 or type_a == pop_size):
        (k, type_a) = step_fast(pop_size, a, b, c, d, beta, type_a)
        i += k
    return i


def estimate_unconditional_fixation_time(pop_size, beta, a, b, c, d, epsilon, k, r_0):
    assert 0 < epsilon < 1, "Epsilon must be between 0 and 1"

    samples = r_0 * [None]
    for i in range(r_0):
        samples[i] = sample_unconditional_fixation_time_fast(pop_size, a, b, c, d, beta)

    tau_start = np.mean(samples)
    tau_next = 0
    rel_error = np.abs(tau_next - tau_start)
    try:
        while rel_error > epsilon:
            tau_start = tau_next
            for i in range(k):
                samples_i = sample_unconditional_fixation_time_fast(pop_size, a, b, c, d, beta)
                samples.append(samples_i)
            tau_next = np.mean(samples)
            rel_error = np.abs(tau_next - tau_start)/tau_next
    except KeyboardInterrupt:
        print("Aborted with poor estimate {} and {} trials".format(tau_start, len(samples)))
    return tau_next 



#################### Conditional fixation time ####################

def sample_conditional_fixation_time_fast(pop_size, a, b, c, d, beta):
    extinction = True
    steps_called_exctinction = 0
    steps_called_fixation = 0
    while extinction:
        steps_called = 0
        type_a = 1
        i = 0  # fixation time
        while not (type_a == 0 or type_a == pop_size):
            (k, type_a) = step_fast(pop_size, a, b, c, d, beta, type_a)
            steps_called += 1  # step called
            i += k
        if type_a == 0:
            extinction = True
            steps_called_exctinction += steps_called
        if type_a == pop_size:
            extinction = False
            steps_called_fixation += steps_called
    return i



def estimate_conditional_fixation_time(pop_size, beta, a, b, c, d, epsilon, k, r_0):
    assert 0 < epsilon < 1, "Epsilon must be between 0 and 1"

    samples = r_0 * [None]
    for i in range(r_0):
        samples[i] = sample_conditional_fixation_time_fast(pop_size, a, b, c, d, beta)

    tau_start = np.mean(samples)
    tau_next = 0
    rel_error = np.abs(tau_next - tau_start)
    try:
        while rel_error > epsilon:
            tau_start = tau_next
            for i in range(k):
                samples_i = sample_conditional_fixation_time_fast(pop_size, a, b, c, d, beta)
                samples.append(samples_i)
            tau_next = np.mean(samples)
            rel_error = np.abs(tau_next - tau_start)/tau_next
    except KeyboardInterrupt:
        print("Aborted with poor estimate {} and {} trials".format(tau_start, len(samples)))
    return tau_next 


#################### Stationary distribution ####################    
    
def step(pop_size, a, b, c, d, beta, mu, type_a):
    f_a = exp(
        beta * (1.0 / (pop_size - 1)) * (a * (type_a - 1) + b * (pop_size - type_a)))
    f_b = exp(
        beta * (1.0 / (pop_size - 1)) * (c * type_a + d * (pop_size - type_a - 1)))
    a_birth = np.random.rand() < (type_a*f_a)/float(type_a*f_a+(pop_size-type_a)*f_b)
    a_death = np.random.rand() < type_a/float(pop_size)
    mutation = np.random.rand() < mu
    
    if ((a_birth and not a_death and not mutation) or (not a_birth and not a_death and mutation)):
        type_a += 1
    elif ((not a_birth and a_death and not mutation) or (a_birth and a_death and mutation)):
        type_a -= 1    
    return type_a


def run_k_steps(pop_size, a, b, c, d, beta, mu, distribution, type_a, k=None):
    if k is None:
        k=pop_size
    for _ in range(k):
        type_a = step(pop_size, a, b, c, d, beta, mu, type_a)
        distribution[type_a] +=1
    
    return type_a
    


def estimate_stationary(pop_size, beta, a, b, c, d, r_0, epsilon, k, mu):

    "10 arguments required: pop_size beta a b c d r_0 epsilon k mu"

    
    
    distribution = np.zeros(pop_size+1, dtype='float')
    distribution[1] = 1
    type_a = 1
    
    type_a = run_k_steps(pop_size, a, b, c, d, beta, mu, distribution, type_a, r_0-1)
    
    try:
        current_distribution = np.copy(distribution)
        type_a = run_k_steps(pop_size, a, b, c, d, beta, mu, distribution, type_a, k)
        t = 1
        
        kullback = entropy(distribution/np.sum(distribution), current_distribution/np.sum(current_distribution))
        


        while kullback > epsilon:
            current_distribution = np.copy(distribution)
            type_a = run_k_steps(pop_size, a, b, c, d, beta, mu, distribution, type_a, k)
            t += 1
            
            kullback = entropy(distribution/np.sum(distribution), current_distribution/np.sum(current_distribution))
            
            
    except KeyboardInterrupt:
        print("Aborted with divergence {} and {} time steps passed".format(kullback, r_0 + t*k))
        
    time_steps_passed = r_0 + t*k
    
    return distribution/np.sum(distribution)
    

