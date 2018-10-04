from math import exp
import numpy as np
import numba

########## Fixation probability ##########


#@numba.njit
def __gamma_function(j, pop_size, beta, a, b, c, d):
    payoff_mutant = (a * (j - 1) + b * (pop_size - j)
                     ) / (pop_size - 1)
    payoff_resident = (c * j + d * (pop_size - j - 1)
                       ) / (pop_size - 1)
    fitness_mutant = exp(beta * payoff_mutant)
    fitness_resident = exp(beta * payoff_resident)
    return fitness_resident / fitness_mutant


def fixation_probability_direct_method(pop_size, beta, a, b, c, d):
    """
    Numerically computes the fixation probability of player 1 in a population of the other strategy for a Moran process.

    Parameters
    ----------
    pop_size: pop size
    beta: intensity of selection
    a, b, c, d: game

    """
    summation = 0.0
    try:
        gamma = 1.0
        for k in range(1, pop_size):
            gamma *= __gamma_function(k, pop_size,
                                      beta, a, b, c, d)
            summation += gamma
    except OverflowError:
        return 0.0
    return 1.0 / (1.0 + summation)


########## Fixation time ##########

def transition_ratio(pop_size, beta, a, b, c, d, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    pi_A = (i - 1) / (N - 1) * a + (N - i) / (N - 1) * b
    pi_B = (i) / (N - 1) * c + (N - i - 1) / (N - 1) * d
    ratio = exp(beta * (pi_B - pi_A))
    return ratio


def transition_plus(pop_size, beta, a, b, c, d, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    pi_A = (i - 1) / (N - 1) * a + (N - i) / (N - 1) * b
    pi_B = (i) / (N - 1) * c + (N - i - 1) / (N - 1) * d
    fit_A = exp(beta * pi_A)
    fit_B = exp(beta * pi_B)
    prob = (i * fit_A) / (i * fit_A + (N - i) * fit_B) * (N - i) / N
    return prob


def transition_minus(pop_size, beta, a, b, c, d, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    pi_A = (i - 1) / (N - 1) * a + (N - i) / (N - 1) * b
    pi_B = (i) / (N - 1) * c + (N - i - 1) / (N - 1) * d
    fit_A = exp(beta * pi_A)
    fit_B = exp(beta * pi_B)
    prob = ((N - i) * fit_B) / (i * fit_A + (N - i) * fit_B) * i / N
    return prob


def direct_conditional_fixation_time(pop_size, beta, a, b, c, d):
    tau = 0.0
    psi_1 = 1 - fixation_probability_direct_method(pop_size, beta, d, c, b, a) # fixation in A after starting with N-1 A indivs
    phi_1 = fixation_probability_direct_method(pop_size, beta, a, b, c, d)
    psi = psi_1
    R = 1.0
    
    prod_gamma_list = np.zeros(pop_size, dtype='float')
    prod = 1.0
    for h in np.arange(1, pop_size):
        prod_gamma_list[h] = prod #writes 1.0 to index h=0
        gamma = transition_ratio(pop_size, beta, a, b, c, d, h)
        prod = prod * gamma
    
    for k in np.arange(1, pop_size):
        t_plus = transition_plus(pop_size, beta, a, b, c, d, pop_size - k)
        tau += psi / t_plus * R
        
        psi = psi - phi_1 * prod_gamma_list[pop_size - k]
        
        gamma = transition_ratio(pop_size, beta, a, b, c, d, pop_size - k)
        R = 1 + gamma * R

    return tau


def direct_unconditional_fixation_time(pop_size, beta, a, b, c, d):
    tau = 0.0
    phi_1 = fixation_probability_direct_method(pop_size, beta, a, b, c, d)
    R = 1.0
    
    for k in np.arange(1, pop_size):
        t_plus = transition_plus(pop_size, beta, a, b, c, d, pop_size - k)
        tau += R / t_plus
        gamma = transition_ratio(pop_size, beta, a, b, c, d, pop_size - k)
        R = 1 + gamma * R

    return tau * phi_1


########## Stationary distribution ##########


def direct_stationary(pop_size, beta, a, b, c, d, mu):
    
    p_tilde = np.zeros(pop_size+1, dtype='float')
    p = np.zeros(pop_size+1, dtype='float') # not necessary to have two arrays
    
    summing = 1.0
    
    p_tilde[0] = 1.0
    
    for k in range(1, pop_size+1):
        t_plus = transition_plus_with_mutation(pop_size, beta, a, b, c, d, mu, k-1)
        t_minus = transition_minus_with_mutation(pop_size, beta, a, b, c, d, mu, k)
        p_tilde[k] = p_tilde[k-1]*t_plus/t_minus
        summing = summing + p_tilde[k]
        
    for k in range(0, pop_size+1):
        p[k] = p_tilde[k]/summing
    
    
    return p   
        
        
        
def transition_plus_with_mutation(pop_size, beta, a, b, c, d, mu, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    
    pi_A = (i - 1) / (N - 1) * a + (N - i) / (N - 1) * b
    pi_B = (i) / (N - 1) * c + (N - i - 1) / (N - 1) * d
    fit_A = exp(beta * pi_A)
    fit_B = exp(beta * pi_B)
    prob = (i*fit_A) / (i*fit_A+(N-i)*fit_B) * (N-i) / N * (1-mu) + (N-i)*fit_B / (i*fit_A+(N-i)*fit_B) * (N-i)/N * mu
    
    return prob


def transition_minus_with_mutation(pop_size, beta, a, b, c, d, mu, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    
    pi_A = (i - 1) / (N - 1) * a + (N - i) / (N - 1) * b
    pi_B = (i) / (N - 1) * c + (N - i - 1) / (N - 1) * d
    fit_A = exp(beta * pi_A)
    fit_B = exp(beta * pi_B)
    prob = ((N-i)*fit_B) / (i*fit_A+(N-i)*fit_B) * i / N * (1-mu) + i*fit_A / (i*fit_A+(N-i)*fit_B) * i/N * mu
    return prob


if __name__ == "__main__":
    a = 2
    b = 5
    c = 1
    d = 3
    population_size = 10
    intensity_of_selection = 0.2
    print(direct_unconditional_fixation_time(population_size, intensity_of_selection, a, b, c, d))
