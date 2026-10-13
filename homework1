import numpy as np

def gradEs(xa, xb, k):
    return k * (xb - xa) * np.array([-1.0, 1.0]) #Es = 0.5(k)(x)^2 --> grad = kx

def hessEs(k):
    return k * np.array([[1.0, -1.0], [-1.0, 1.0]]) #def of hessian

def dampingForce(xa_new, xb_new, xa_old, xb_old, b, dt):
    ua = (xa_new - xa_old) / dt
    ub = (xb_new - xb_old) / dt
    return b * (ub - ua) * np.array([-1.0, 1.0]) #damping force = b*v

def dampingJacobian(b, dt):
    return (b / dt) * np.array([[1.0, -1.0], [-1.0, 1.0]]) 

def getFJ(x_new, x_old, u_old, params):
    m = params['m']
    k = params['k']
    b = params['b']
    g = params['g']
    dt = params['dt']
    N = len(m) - 1 #[cite: 5]
    
    f = np.zeros(N + 1)
    J = np.zeros((N + 1, N + 1))
    
    # Inertial and external force (gravity) terms for nodes
    for j in range(1, N + 1): #diagonals
        f[j] = (m[j] / dt) * (((x_new[j] - x_old[j]) / dt) - u_old[j]) - m[j] * g #equation 7 in pdf
        J[j, j] += m[j] / (dt**2) #equation 8 in pdf
        
    # element assembly loop over springs and dampers
    for i in range(1, N + 1):
        idx = [i - 1, i]
        f[idx] += gradEs(x_new[i-1], x_new[i], k[i]) #spring
        J[np.ix_(idx, idx)] += hessEs(k[i]) #spring
        f[idx] += dampingForce(x_new[i-1], x_new[i], x_old[i-1], x_old[i], b[i], dt) #damper
        J[np.ix_(idx, idx)] += dampingJacobian(b[i], dt) #damper
        
    return f, J

def implicitEulerStep(x_old, u_old, params):
    N = len(params['m']) - 1
    dt = params['dt']
    eps = 1e-4 #acceptable error!!
    
    x_new = x_old.copy() 
    free = list(range(1, N + 1)) 
    
    err = np.inf
    while err >= eps:
        f, J = getFJ(x_new, x_old, u_old, params) #given
        
        f_free = f[free]
        J_free = J[np.ix_(free, free)]
        
        delta_x_free = np.linalg.solve(J_free, f_free) #given in pseudocode
        x_new[free] -= delta_x_free #given in pseudocode
        
        f_check, _ = getFJ(x_new, x_old, u_old, params)
        err = np.linalg.norm(f_check[free]) #checking convergence
        
    u_new = (x_new - x_old) / dt 
    return x_new, u_new

def explicitEulerStep(x_old, u_old, params):
   
    m = params['m']
    k = params['k']
    b = params['b']
    g = params['g']
    dt = params['dt']
    N = len(m) - 1
    
    f_int = np.zeros(N + 1)
    
    for i in range(1, N + 1):
        idx = [i - 1, i] #index
        
        f_int[idx] += gradEs(x_old[i-1], x_old[i], k[i]) #spring interbal force eqn 3
        
        # damper internal force eqn 4
        u_a = u_old[i-1]
        u_b = u_old[i]
        g_i = b[i] * (u_b - u_a) * np.array([-1.0, 1.0]) 
        f_int[idx] += g_i

    x_new = np.zeros(N + 1)
    u_new = np.zeros(N + 1)
    
    for j in range(1, N + 1):
        a_j = g - (1.0 / m[j]) * f_int[j] #[cite: 7]
        
        x_new[j] = x_old[j] + dt * u_old[j] #[cite: 7]
        u_new[j] = u_old[j] + dt * a_j #[cite: 7]
        
    return x_new, u_new


#test cases implemented

if __name__ == "__main__":
    # swapping these to test each case
    dt = 0.1 # time step from prompt
    total_time = 10.0 # final simulation time
    
    params = {
        'm': np.array([0, 1]), # N=1, mass = 1
        'k': np.array([0, 1]), # spring stiffness = 1
        'b': np.array([0, 0]), # no damping (b=0)
        'g': 9.8,
        'dt': dt
    }
    
    # initial conditions
    N = len(params['m']) - 1
    x = np.zeros(N + 1) 
    u = np.zeros(N + 1)
    
    # time loop
    num_steps = int(total_time / dt)
    for step in range(num_steps):
        
        x, u = explicitEulerStep(x, u, params) 
        
    print(f"Final x at t={total_time}s:", x)
    print(f"Final u at t={total_time}s:", u)
