from acados_template import AcadosOcp, AcadosOcpSolver, ocp_get_default_cmake_builder
from tether_model import export_drone_tether_fo_model
import numpy as np
import scipy.linalg
from utils import plot_drone_tet_fo_eval
from casadi import SX, vertcat, Function, sqrt, fmax, sin, cos

def solve_ocp():
    ocp = AcadosOcp()

    # set model
    model = export_drone_tether_fo_model()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    N = 50 # horizon

    # set dimensions
    ocp.solver_options.N_horizon = N

    # State cost
    Q = np.eye(nx)
    Q[0,0] = 1.0     # x
    Q[1,1] = 1.0     # y
    Q[2,2] = 10.0        # z
    Q[3,3] = 1.0     # phi
    Q[4,4] = 1.0     # theta
    Q[5,5] = 1.0     # psi
    Q[6,6] = 0.0       # vwx
    Q[7,7] = 1.0        # vwy
    Q[8,8] = 1.0        # vwz

    # Input cost
    R = np.eye(nu)
    R[0,0] = 0.0    # phi_cmd
    R[1,1] = 0.0    # theta_cmd
    R[2,2] = 0.0    # psi_cmd
    R[3,3] = 10.0    # thrust

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = 50*scipy.linalg.block_diag(Q)

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    pose_ref = np.array([0.0, 0.0, 1.0]) # x_w, y_w, z_w

    yref = np.zeros((ny, ))
    yref[0] = pose_ref[0] # x_w
    yref[1] = pose_ref[1] # y_w
    yref[2] = pose_ref[2] # z_w
    ocp.cost.yref = yref
    yref_e = np.zeros((ny_e, ))
    yref_e[0] = pose_ref[0] # x_w
    yref_e[1] = pose_ref[1] # y_w
    yref_e[2] = pose_ref[2] # z_w
    ocp.cost.yref_e = yref_e

    ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_e = vertcat(ocp.model.x)

    # set constraints
    ocp.constraints.lbu = np.array([-np.pi/4, -np.pi/4, -np.pi, 0.0])
    ocp.constraints.ubu = np.array([np.pi/4, np.pi/4, np.pi, 1.0])
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x0 = ocp.constraints.x0
    u0 = np.array([0.0, 0.0, 0.0, 0.5])
    ocp.constraints.lbx = np.array([-200.0, -200.0, 0.0, -np.pi/4, -np.pi/4,
        -np.pi, -10.0, -10.0, -10.0])
    ocp.constraints.ubx = np.array([+200.0, +200.0, +200.0, +np.pi/6, +np.pi/6,
        +np.pi, +10.0, +10.0, +10.0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    f_expl_fun = Function("f_expl_fun", [model.x, model.u], [model.f_expl_expr])
    print("f_expl at x0:", f_expl_fun(x0, u0))

    cost_y_fun = Function("cost_y_fun", [model.x, model.u], [model.cost_y_expr])
    print("cost_y_expr at x0:", cost_y_fun(x0, u0))

    # set options
    ocp.solver_options.print_level = 3
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    x_init = x0 # would be from sensor x_init = get_state_from_px4()
    u_init = u0 # just initial guess for input
    for i in range(N):
        ocp_solver.set(i, "x", x_init)
        ocp_solver.set(i, "u", u_init)
    ocp_solver.set(N, "x", x_init)
    nsim = 10
    print("x0:", x_init)
    print("u0:", u_init)
    theta_ref = 0.0
    for nsim in range(nsim):
        # constraints state
        ocp_solver.constraints_set(0, 'lbx', x_init) # == data from sensor
        ocp_solver.constraints_set(0, 'ubx', x_init)

        status = ocp_solver.solve()

        if status != 0:
            ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
            raise Exception(f'acados returned status {status}.')

        yref[0] = 1.0*sin(theta_ref) # x_w
        yref[1] = 1.0*cos(theta_ref) # y_w
        yref[2] = 1.0
        # get solution
        for i in range(N):
            simX[i,:] = ocp_solver.get(i, "x")
            simU[i,:] = ocp_solver.get(i, "u")
            # ocp_solver.set(i, "yref",yref)
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

        # ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    # plot_drone_tet_fo_eval(np.linspace(0, Tf, N+1), np.pi/4, simU, simX, latexify=True)

if __name__ == "__main__":
    solve_ocp()