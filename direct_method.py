from math import exp
import numpy as np
import numba

########## Fixation probability ##########


@numba.njit
def __gamma_function(j, population_size, intensity_of_selection, a, b, c, d):
    payoff_mutant = (a * (j - 1) + b * (population_size - j)
                     ) / (population_size - 1)
    payoff_resident = (c * j + d * (population_size - j - 1)
                       ) / (population_size - 1)
    fitness_mutant = exp(intensity_of_selection * payoff_mutant)
    fitness_resident = exp(intensity_of_selection * payoff_resident)
    return fitness_resident / fitness_mutant


def fixation_probability_direct_method(population_size, intensity_of_selection, a, b, c, d):
    """
    Numerically computes the fixation probability of player 1 in a population of the other strategy for a Moran process.

    Parameters
    ----------
    population_size: pop size
    intensity_of_selection: intensity of selection
    a, b, c, d: game

    """
    summation = 0.0
    try:
        gamma = 1.0
        for k in range(1, population_size):
            gamma *= __gamma_function(k, population_size,
                                      intensity_of_selection, a, b, c, d)
            summation += gamma
    except OverflowError:
        return 0.0
    return 1.0 / (1.0 + summation)


########## Fixation time ##########

def transition_ratio(pop_size, intensity_of_selection, a, b, c, d, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    pi_A = (i - 1) / float(N - 1) * a + (N - i) / float(N - 1) * b
    pi_B = (i) / float(N - 1) * c + (N - i - 1) / float(N - 1) * d
    ratio = exp(intensity_of_selection * (pi_B - pi_A))
    return ratio


def transition_plus(pop_size, intensity_of_selection, a, b, c, d, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    pi_A = (i - 1) / float(N - 1) * a + (N - i) / float(N - 1) * b
    pi_B = (i) / float(N - 1) * c + (N - i - 1) / float(N - 1) * d
    fit_A = exp(intensity_of_selection * pi_A)
    fit_B = exp(intensity_of_selection * pi_B)
    prob = (i * fit_A) / float(i * fit_A +
                               (N - i) * fit_B) * (N - i) / float(N)
    return prob


def transition_minus(pop_size, intensity_of_selection, a, b, c, d, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    pi_A = (i - 1) / float(N - 1) * a + (N - i) / float(N - 1) * b
    pi_B = (i) / float(N - 1) * c + (N - i - 1) / float(N - 1) * d
    fit_A = exp(intensity_of_selection * pi_A)
    fit_B = exp(intensity_of_selection * pi_B)
    prob = ((N - i) * fit_B) / float(i * fit_A +
                                     (N - i) * fit_B) * i / float(N)
    return prob


def direct_conditional_fixation_time(pop_size, intensity_of_selection, a, b, c, d):
    tau = 0.0
    prod = 1.0
    gamma = transition_ratio(pop_size, intensity_of_selection, a, b, c, d, 1)
    phi_1 = fixation_probability_direct_method(
        pop_size, intensity_of_selection, a, b, c, d)
    phi = phi_1
    Q = 1 / gamma * (1 / phi - 1)
    for k in np.arange(1, pop_size):
        t_plus = transition_plus(
            pop_size, intensity_of_selection, a, b, c, d, k)
        tau += Q * phi / t_plus
        prod = prod * transition_ratio(pop_size,
                                       intensity_of_selection, a, b, c, d, k)
        gamma = transition_ratio(
            pop_size, intensity_of_selection, a, b, c, d, k+1)
        Q = (Q - 1) / gamma
        phi += phi_1 * prod

    return tau


def direct_unconditional_fixation_time(pop_size, intensity_of_selection, a, b, c, d):
    tau = 0.0
    gamma_1 = transition_ratio(pop_size, intensity_of_selection, a, b, c, d, 1)
    phi_1 = fixation_probability_direct_method(
        pop_size, intensity_of_selection, a, b, c, d)
    Q = 1 / gamma_1 * (1 / phi_1 - 1)
    for k in np.arange(1, pop_size):
        t_plus = transition_plus(
            pop_size, intensity_of_selection, a, b, c, d, k)
        tau += Q / t_plus
        gamma = transition_ratio(
            pop_size, intensity_of_selection, a, b, c, d, k+1)
        Q = (Q - 1) / gamma

    return tau * phi_1


########## Stationary distribution ##########

def transition_plus_with_mutation(pop_size, intensity_of_selection, a, b, c, d, mu, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    if i == 0:
        prob = mu
    else:
        pi_A = (i - 1) / float(N - 1) * a + (N - i) / float(N - 1) * b
        pi_B = (i) / float(N - 1) * c + (N - i - 1) / float(N - 1) * d
        fit_A = exp(intensity_of_selection * pi_A)
        fit_B = exp(intensity_of_selection * pi_B)
        prob = (i * fit_A) / float(i * fit_A +
                                   (N - i) * fit_B) * (N - i) / float(N)
    return prob


def transition_minus_with_mutation(pop_size, intensity_of_selection, a, b, c, d, mu, number_of_mutants):
    i = number_of_mutants
    N = pop_size
    if i == N:
        prob = mu
    else:
        pi_A = (i - 1) / float(N - 1) * a + (N - i) / float(N - 1) * b
        pi_B = (i) / float(N - 1) * c + (N - i - 1) / float(N - 1) * d
        fit_A = exp(intensity_of_selection * pi_A)
        fit_B = exp(intensity_of_selection * pi_B)
        prob = ((N - i) * fit_B) / float(i * fit_A +
                                         (N - i) * fit_B) * i / float(N)
    return prob


def direct_stationary(pop_size, intensity_of_selection, a, b, c, d, mu):

    p_tilde = np.zeros(pop_size+1, dtype='float')
    p = np.zeros(pop_size+1, dtype='float')  # not necessary to have two arrays

    summing = 1.0

    p_tilde[0] = 1.0

    for k in range(1, pop_size+1):
        t_plus = transition_plus_with_mutation(
            pop_size, intensity_of_selection, a, b, c, d, mu, k-1)
        t_minus = transition_minus_with_mutation(
            pop_size, intensity_of_selection, a, b, c, d, mu, k)
        p_tilde[k] = p_tilde[k-1]*t_plus/t_minus
        summing = summing + p_tilde[k]

    for k in range(0, pop_size+1):
        p[k] = p_tilde[k]/summing

    return p
