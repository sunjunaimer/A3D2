import cvxpy as cp
import numpy as np
import time


def ots(c, n):
    tic = time.time()

    # Define the optimization variable x (dimension n x n)
    x = cp.Variable((n, n), nonneg=True)

    # Define the all-one vector
    ones = np.ones(n)

    # Define the constraints
    constraints = [
        x @ ones == ones/n,    # x * (all-one vector) = (all-one vector)
        ones @ x == ones/n,    # (all-one vector) * x = (all-one vector)
    ]

    # Define the objective function
    objective = cp.Minimize(cp.sum(cp.multiply(c, x)))

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Print the results
    obj = problem.value
    # print("Optimal value:", obj)
    # print("Optimal x:", x)
    # print(x.value)
    xx = x.value
    toc = time.time()
    runtime = toc - tic 
    print('time of OT solver:', runtime)
    return xx, obj


if __name__ == '__main__':
    # Problem dimensions
    n = 256  # Example dimension, you can set this to your desired size

    # Fixed input matrix c (dimension n x n)
    c = np.random.rand(n, n)  # Replace this with your specific matrix

    x, obj = ots(c, n)
    print("Optimal value:", obj)
    print("Optimal x:", x)
