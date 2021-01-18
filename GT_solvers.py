import numpy as np
import gurobipy as g
import operator
from cvxopt import matrix, solvers

def solve_NFG_gurobi(matrix, fp = None, fp_thresh = 0.01):
    """
    Solve normal-form game using gurobi

    Parameters
    ----------
    matrix : np.array
        Game matrix
    fp : np.array
        list of false-positive rates of all classifiers
    fp_thresh : float
        threshold for the false-positive constraint
    """
    grb_obj = g.GRB.MINIMIZE
    grb_op = operator.le

    model = g.Model()
    model.setParam('OutputFlag', False)

    # Create variables x, y.
    U = model.addVar(vtype=g.GRB.CONTINUOUS, lb=float('-inf'), name="U")
    s = []
    for i, u in enumerate(matrix[0]):
        s.append(model.addVar(vtype=g.GRB.CONTINUOUS, lb=0, name="s" + str(i)))

    # Integrate new variables into model.
    model.update()

    model.setObjective(U, grb_obj)

    for ui in matrix:
        const = 0
        for i, u in enumerate(ui):
            const += s[i] * u
        model.addConstr(grb_op(const, U))

    rm = 1
    if not (fp is None):
        rm = 2
        constraint = 0
        for i, u in enumerate(matrix[0]):
            constraint += s[i]*fp[i]
        model.addConstr(constraint <= fp_thresh)

    constraint = 0
    for i, u in enumerate(matrix[0]):
        constraint += s[i]
    model.addConstr(constraint == 1)


    # Solve the model.
    model.optimize()
    model.write('out.lp')

    sol1 = np.zeros(len(s))
    for i, var in enumerate(s):
        sol1[i] = var.x

    constr=model.getConstrs()
    sol2 = np.zeros(len(constr)-rm)
    for i, const in enumerate(constr[:-rm]):
        sol2[i] = -1*const.getAttr("Pi")

    return sol1, sol2, model.objVal

def solve_NFG_cvx(Q, fp = None, fp_thresh = 0.01):
    """
    Solve normal-form game using CVXOPT

    Parameters
    ----------
    matrix : np.array
        Game matrix
    fp : np.array
        list of false-positive rates of all classifiers
    fp_thresh : float
        threshold for the false-positive constraint
    """
    solvers.options['show_progress'] = False
    A = np.hstack((Q, -1*np.ones((Q.shape[0],1))))

    r = np.ones((1,Q.shape[1]+1))
    r[:,-1] = 0
    A = np.vstack((A, r))
    r = -1 * r
    r[:, -1] = 0
    A = np.vstack((A, r))
    v = np.hstack((np.diag(-1*np.ones(Q.shape[1])),np.zeros((Q.shape[1],1))))
    A = np.vstack((A,v))

    c = np.zeros(Q.shape[1]+1)
    c[-1] = 1

    b = np.zeros(np.sum(Q.shape) + 2)
    b[Q.shape[0]] = 1
    b[Q.shape[0]+1] = -1

    if not fp is None:
        r = np.hstack((np.array(fp), 0))
        A = np.vstack((A, r))
        b = np.hstack((b, fp_thresh))

    A1 = matrix(A)
    b = matrix(b)
    c = matrix(c)

    sol = solvers.lp(c, A1, b)
    solution1 = np.reshape(sol['x'][:-1,:], (-1))
    solution2 = np.reshape(sol['z'][:Q.shape[0], :], (-1))
    return solution1, solution2, sol['primal objective']