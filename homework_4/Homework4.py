import numpy as np
from gradEb_hessEb_Shell import getTheta, gradEb_hessEb_Shell
from shell_functions import getEdgesAndHinges


def mesh_3x3_plate():
    elements = []
    nx, ny = 3, 3
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i            # lower-left
            n1 = j * nx + (i + 1)      # lower-right
            n2 = (j + 1) * nx + i      # upper-left
            n3 = (j + 1) * nx + (i + 1)  # upper-right
            
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    return np.array(elements, dtype=int)


def Eb(q, thetaBar, kb):
    # bending energy Eb = 0.5 * kb * (theta - thetaBar)^2 for one hinge
    return 0.5 * kb * (getTheta(q) - thetaBar) ** 2


def main():
    
    x0 = np.array([0.0, 0.0, 0.0])
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.5, 1.0, 0.0])
    x3 = np.array([0.5, -1.0, 1.0])
    q = np.concatenate([x0, x1, x2, x3])

    thetaBar = 0.0
    kb = 1.0

    theta0 = getTheta(q)
    Eb0 = Eb(q, thetaBar, kb)
    print(f"theta = {theta0:.6f} rad, Eb = {Eb0:.6f} J\n")

   
    eps = 0.01
    Eb_perturbed = np.zeros(12)
    gradFD = np.zeros(12)

    for i in range(12):
        dq = np.zeros(12)
        dq[i] = eps
        Eb_perturbed[i] = Eb(q + dq, thetaBar, kb)
        gradFD[i] = (Eb_perturbed[i] - Eb0) / eps

    for node_id in range(4):
        vals = Eb_perturbed[3 * node_id : 3 * node_id + 3]
        print(f"Q{3 + node_id} (perturb x{node_id}): {vals[0]:.5f}, {vals[1]:.5f}, {vals[2]:.5f} J")

  

    dF, dJ = gradEb_hessEb_Shell(q, None, None, None, thetaBar, kb)
   


if __name__ == "__main__":
    main()