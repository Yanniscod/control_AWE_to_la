from acados_template import AcadosOcp, AcadosOcpSolver, ocp_get_default_cmake_builder
from tether_model import export_drone_tether_fo_model
import numpy as np
import scipy.linalg
from utils import plot_drone_tet_fo_eval
from casadi import SX, vertcat, Function, sqrt, fmax, sin, cos, pi
import time

def set_init_values(mpc_model='attitude-fo'):
    x0, u0 = None, None
    if mpc_model=='attitude-fo':
            x0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            u0 = np.array([0.0, 0.0, 0.0, 0.5])
    return x0, u0

def set_constraints_model(mpc_model='attitude-fo'):
    lbu, ubu = None, None
    lbx, ubx = None, None
    if mpc_model == 'attitude-fo':
        lbu = np.array([-pi/4, -pi/4, -pi, 0.0])
        ubu = np.array([pi/4, pi/4, pi, 1.0])
        lbx = np.array([-200.0, -200.0, 0.0, -pi/4, -pi/4,
            -pi, -10.0, -10.0, -10.0])
        ubx = np.array([+200.0, +200.0, +200.0, +pi/4, +pi/4,
            +pi, +10.0, +10.0, +10.0])
    else:
        print('Wrong model name')

    return lbu, ubu, lbx, ubx

def set_cost_model(Q, R, mpc_model='attitude-fo'):
    if mpc_model == 'attitude-fo':
        Q[0,0] = 10.0     # x
        Q[1,1] = 10.0     # y
        Q[2,2] = 10.0        # z
        Q[3,3] = 1.0     # phi
        Q[4,4] = 1.0     # theta
        Q[5,5] = 1.0     # psi
        Q[6,6] = 1.0       # vwx
        Q[7,7] = 1.0        # vwy
        Q[8,8] = 1.0        # vwz

        R[0,0] = 1.0    # phi_cmd
        R[1,1] = 1.0    # theta_cmd
        R[2,2] = 1.0    # psi_cmd
        R[3,3] = 10.0    # thrust
    else:
        print('Wrong model name')
    
    return Q, R
    
def solve_ocp(mpc_model='attitude-fo'):
    ocp = AcadosOcp()

    # set model
    model = None
    if mpc_model=='attitude-fo':
        model = export_drone_tether_fo_model()
    ocp.model = model

    Tf = 1.0
    N = 40 # horizon
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    nsim = 10 # number of simulations
    ocp.solver_options.N_horizon = N

    # Set costs
    Q = np.eye(nx)
    R = np.eye(nu)
    Q, R = set_cost_model(Q, R, mpc_model)

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = 50*scipy.linalg.block_diag(Q)

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    init_pose_ref = np.array([0.0, 0.0, 1.0]) # x_w, y_w, z_w

    yref = np.zeros((ny, ))
    yref[0] = init_pose_ref[0] # x_w
    yref[1] = init_pose_ref[1] # y_w
    yref[2] = init_pose_ref[2] # z_w
    ocp.cost.yref = yref
    yref_e = np.zeros((ny_e, ))
    yref_e[0] = init_pose_ref[0] # x_w
    yref_e[1] = init_pose_ref[1] # y_w
    yref_e[2] = init_pose_ref[2] # z_w
    ocp.cost.yref_e = yref_e

    ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_e = vertcat(ocp.model.x)

    # set constraints
    lbu, ubu, lbx, ubx = set_constraints_model(mpc_model)
    print('1', np.size(lbx))

    lbx2 = np.array([-200.0, -200.0, 0.0, -pi/4, -pi/4,
            -pi, -10.0, -10.0, -10.0])
    print('2', np.size(lbx2))
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = np.arange(nx)

    # set options, not differentiating between models atm
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
    ocp.solver_options.tf = Tf
    ocp.solver_options.tol = 1e-4

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    x0, u0 = set_init_values(mpc_model)
    ocp.constraints.x0 = x0

    x_init = x0 # would be from sensor x_init = get_state_from_px4()
    u_init = u0 # just initial guess for input
    for i in range(N):
        ocp_solver.set(i, "x", x_init)
        ocp_solver.set(i, "u", u_init)
    ocp_solver.set(N, "x", x_init)
    print("x0:", x_init)
    print("u0:", u_init)
    theta_ref = 0.0
    for nsim in range(nsim):
        # constraints state
        # x_init = get_state_from_px4()
        print(lbx)
        print(x_init)
        ocp_solver.constraints_set(0, 'lbx', x_init) # == data from sensor
        ocp_solver.constraints_set(0, 'ubx', x_init)

        # yref = get_target_from_px4()
        # set circular ref
        yref[0] = 1.0*cos(theta_ref)
        yref[1] = 1.0*sin(theta_ref)
        yref[2] = 1.0
        for i in range(N):
            ocp_solver.set(i, "yref", yref)
        ocp_solver.set(N, "yref", yref[:nx])

        t0 = time.time()
        status = ocp_solver.solve()
        t1 = time.time()
        print(f"solve time: {(t1-t0)*1000} ms")

        if status != 0:
            ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
            raise Exception(f'acados returned status {status}.')

        # get solution
        for i in range(N):
            simX[i,:] = ocp_solver.get(i, "x")
            simU[i,:] = ocp_solver.get(i, "u")
        simX[N,:] = ocp_solver.get(N, "x")

        # shift, reuse prev solution as initial guess
        for i in range(N-1):
            ocp_solver.set(i, "x", simX[i+1])
            ocp_solver.set(i, "u", simU[i+1])
        ocp_solver.set(N-1, "u", simU[-1])        # last state guess = last predicted
        ocp_solver.set(N, "x", simX[-1])        # last state guess = last predicted

        x_init = ocp_solver.get(1, "x") # uses simulated state as sensor data
        u_init = ocp_solver.get(0, "u")
        print("new x0:", x_init)
        print("new u0:", u_init)
        theta_ref += 0.01

    plot_drone_tet_fo_eval(np.linspace(0, Tf, N+1), pi/4, simU, simX, latexify=True)

if __name__ == "__main__":
    solve_ocp()