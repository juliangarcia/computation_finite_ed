import numpy as np
from scipy.linalg import solve_banded

########## Fixation probability ##########


def create_transition_matrix(population_size, intensity_of_selection, a, b, c, d):
    """

    Parameters
    ----------
    intensity_of_selection
    population_size
    a, b, c, d: game

    """
    super_diagonal = np.zeros(population_size - 1, dtype='float')
    sub_diagonal = np.zeros(population_size - 1, dtype='float')
    diagonal = np.zeros(population_size - 1, dtype='float')

    for i in range(1, population_size):  # Start at 1 because T_0^+ is zero anyway
        idx = i - 1
        payoff_A = (a * (i - 1) + b * (population_size - i)) / \
            float(population_size - 1)
        payoff_B = (c * i + d * (population_size - i - 1)) / \
            float(population_size - 1)
        t_plus = i * np.exp(intensity_of_selection * payoff_A) / (
            i * np.exp(intensity_of_selection * payoff_A) + (population_size - i) * np.exp(intensity_of_selection * payoff_B)) * (
            population_size - i) / float(population_size)
        t_minus = (population_size - i) * np.exp(intensity_of_selection * payoff_B) / (
            i * np.exp(intensity_of_selection * payoff_A) + (population_size - i) * np.exp(
                intensity_of_selection * payoff_B)) * i / float(population_size)
        super_diagonal[idx] = t_plus
        sub_diagonal[idx] = t_minus
        diagonal[idx] = -(t_plus + t_minus)  # 1 - (t_plus + t_minus)

    # put a zero at first position and drop the last element
    super_diagonal = np.insert(super_diagonal, 0, 0)
    super_diagonal = super_diagonal[0:len(super_diagonal) - 1]
    # drop first element and insert zero at the end
    sub_diagonal = sub_diagonal[1:len(sub_diagonal)]
    sub_diagonal = np.insert(sub_diagonal, len(sub_diagonal), 0)

    return np.matrix([super_diagonal, diagonal, sub_diagonal])


def fixation_probability_matrix_based(population_size, intensity_of_selection, a, b, c, d):
    '''
    Calculates the fixation probability for a given transition matrix.

    Parameters
    ----------
    intensity_of_selection
    population_size
    a, b, c, d: game


    '''
    transition_matrix = create_transition_matrix(
        population_size, intensity_of_selection, a, b, c, d)

    # last element T_{N-1}^{+}  (population_size-1)
    fixating_column = np.zeros(population_size - 1)
    payoff_A = (a * ((population_size - 1) - 1) + b * (population_size - (population_size - 1))) / float(
        population_size - 1)
    payoff_B = (c * (population_size - 1) + d * (population_size - (population_size - 1) - 1)) / float(
        population_size - 1)
    fixating_column[-1] = -1 * ((population_size - 1) * np.exp(intensity_of_selection * payoff_A) / (
        (population_size - 1) * np.exp(intensity_of_selection * payoff_A) + (population_size - (population_size - 1)) * np.exp(
            intensity_of_selection * payoff_B)) * (population_size - (population_size - 1)) / float(population_size))
    # Careful: solve_banded only for well-mixed, linear complexity. Use
    # linalg.solve for other cases
    fixation_probability_all_transient = solve_banded(
        (1, 1), transition_matrix, fixating_column)
    fixation_probability = fixation_probability_all_transient[0]
    return fixation_probability


########## Fixation time ##########


def create_conditional_transition_matrix(pop_size, intensity_of_selection, a, b, c, d,
                                         fixation_probability_all_transient):
    """

    This function creates a conditional transition matrix. Each transition probability is weighted by the
    ratio of the fixation probabilities of the incoming and outgoing state.

    """
    # Insert a zero at the beginning and a 1 at the end
    fixation_probability_array = np.insert(
        fixation_probability_all_transient, 0, 0)
    fixation_probability_array = np.insert(
        fixation_probability_array, len(fixation_probability_array), 1)

    super_diagonal = np.zeros(pop_size - 1, dtype='float')
    sub_diagonal = np.zeros(pop_size - 1, dtype='float')
    diagonal = np.zeros(pop_size - 1, dtype='float')

    for i in range(1, pop_size):  # Start at 1 because T_0^+ is zero anyway
        idx = i - 1
        payoff_A = (a * (i - 1) + b * (pop_size - i)) / float(pop_size - 1)
        payoff_B = (c * i + d * (pop_size - i - 1)) / float(pop_size - 1)
        t_plus = i * np.exp(intensity_of_selection * payoff_A) / (
            i * np.exp(intensity_of_selection * payoff_A) + (pop_size - i) * np.exp(intensity_of_selection * payoff_B)) * (
            pop_size - i) / float(pop_size)
        t_minus = (pop_size - i) * np.exp(intensity_of_selection * payoff_B) / (
            i * np.exp(intensity_of_selection * payoff_A) + (pop_size - i) * np.exp(
                intensity_of_selection * payoff_B)) * i / float(pop_size)

        phi_plus = fixation_probability_array[i + 1]
        phi_current = fixation_probability_array[i]
        phi_minus = fixation_probability_array[i - 1]

        super_diagonal[idx] = t_plus * phi_plus / phi_current
        sub_diagonal[idx] = t_minus * phi_minus / phi_current
        diagonal[idx] = -(t_plus + t_minus)  # 1 - (t_plus + t_minus)

    # put a zero at first position and drop the last element
    super_diagonal = np.insert(super_diagonal, 0, 0)
    super_diagonal = super_diagonal[0:len(super_diagonal) - 1]
    # drop first element and insert zero at the end
    sub_diagonal = sub_diagonal[1:len(sub_diagonal)]
    sub_diagonal = np.insert(sub_diagonal, len(sub_diagonal), 0)

    return np.matrix([super_diagonal, diagonal, sub_diagonal])


def transition_matrix_conditional_fixation_time(pop_size, intensity_of_selection, a, b, c, d):
    '''

    Calculates the conditional fixation time for a given transition matrix.

    '''
    transition_matrix = create_transition_matrix(
        pop_size, intensity_of_selection, a, b, c, d)
    fixating_column = np.zeros(pop_size - 1)
    payoff_A = (a * ((pop_size - 1) - 1) + b * (pop_size - (pop_size - 1))) / float(
        pop_size - 1)
    payoff_B = (c * (pop_size - 1) + d * (pop_size - (pop_size - 1) - 1)) / float(
        pop_size - 1)
    fixating_column[-1] = -1 * ((pop_size - 1) * np.exp(intensity_of_selection * payoff_A) / (
        (pop_size - 1) * np.exp(intensity_of_selection * payoff_A) + (pop_size - (pop_size - 1)) * np.exp(
            intensity_of_selection * payoff_B)) * (pop_size - (pop_size - 1)) / float(pop_size))

    fixation_probability_all_transient = solve_banded(
        (1, 1), transition_matrix, fixating_column)

    conditional_transition_matrix = create_conditional_transition_matrix(pop_size, intensity_of_selection, a, b, c, d,
                                                                         fixation_probability_all_transient)

    solve_column = (-1) * np.ones(pop_size - 1)

    fixation_time_all_transient = solve_banded(
        (1, 1), conditional_transition_matrix, solve_column)
    fixation_time = fixation_time_all_transient[0]
    return fixation_time


def transition_matrix_unconditional_fixation_time(pop_size, intensity_of_selection, a, b, c, d):
    '''

    Calculates the unconditional fixation time for a given transition matrix.

    '''
    transition_matrix = create_transition_matrix(
        pop_size, intensity_of_selection, a, b, c, d)

    solve_column = (-1) * np.ones(pop_size - 1)

    fixation_time_all_transient = solve_banded(
        (1, 1), transition_matrix, solve_column)
    # print fixation_time_all_transient
    fixation_time = fixation_time_all_transient[0]
    return fixation_time


########## Stationary distribution ##########


def create_full_banded_transition_matrix(pop_size, intensity_of_selection, a, b, c, d, mu):
    """

    Creates the full transition matrix (i.e. with mutation) in a banded form.

    """
    super_diagonal = np.zeros(pop_size + 1, dtype='float')
    sub_diagonal = np.zeros(pop_size + 1, dtype='float')
    diagonal = np.zeros(pop_size + 1, dtype='float')

    super_diagonal[0] = mu
    diagonal[0] = -mu

    diagonal[pop_size] = -mu
    sub_diagonal[pop_size] = mu

    for i in range(1, pop_size):
        payoff_A = (a * (i - 1) + b * (pop_size - i)) / float(pop_size - 1)
        payoff_B = (c * i + d * (pop_size - i - 1)) / float(pop_size - 1)
        t_plus = i * np.exp(intensity_of_selection * payoff_A) / (
            i * np.exp(intensity_of_selection * payoff_A) + (pop_size - i) * np.exp(intensity_of_selection * payoff_B)) * (
            pop_size - i) / float(pop_size)
        t_minus = (pop_size - i) * np.exp(intensity_of_selection * payoff_B) / (
            i * np.exp(intensity_of_selection * payoff_A) + (pop_size - i) * np.exp(
                intensity_of_selection * payoff_B)) * i / float(pop_size)
        super_diagonal[i] = t_plus
        sub_diagonal[i] = t_minus
        diagonal[i] = -(t_plus + t_minus)

    matrix_transposed = np.matrix([sub_diagonal, diagonal, super_diagonal])

    return matrix_transposed


def matrix_stationary_remove_equation(pop_size, intensity_of_selection, a, b, c, d, mu):

    full_banded_transition_matrix = create_full_banded_transition_matrix(
        pop_size, intensity_of_selection, a, b, c, d, mu)

    # remove first row and column (=remove first column of band matrix and set
    # first super_diagnoal to zero):
    cropped_matrix = full_banded_transition_matrix[:, 1:pop_size+1]
    cropped_matrix[0, 0] = 0

    solving_column = np.zeros(pop_size, dtype='float')
    solving_column[0] = -mu

    # multiply matrix by factor to avoid underflow which makes matrix singular
    if pop_size < 50:
        factor = 10**2
    elif pop_size >= 50 and pop_size < 70:
        factor = 10**3
    else:
        factor = 10**10

    cropped_distribution = solve_banded(
        (1, 1), cropped_matrix*factor, solving_column*factor)

    stationary_distribution = np.concatenate(
        (np.array([1.0]), cropped_distribution), axis=0)

    sum_stationary = np.sum(stationary_distribution)
    stationary_distribution = stationary_distribution/sum_stationary

    return stationary_distribution
