from acados_template import AcadosOcp, AcadosOcpSolver
from tether_model import export_drone_tether_ode_model
import numpy as np
import scipy.linalg
from utils import plot_drone_tet_gpt_eval
from casadi import vertcat, sin, cos, pi, sqrt
import time

def solve_ocp():
    ocp = AcadosOcp()

    moving_ref = False
    init_pose_ref = np.array([1.622028, -2.523693, 5.000000]) # x_w, y_w, z_w
    x0 = np.array([1.123196, -2.017403, 4.428124, -0.033080, 0.029961, 1.907401, 0.760853, 0.409851, -0.136514, -0.017031, 0.070304, -0.042765, 5.032940])
    u0 = np.array([0.000000, 0.000000, 0.000000, 0.473300, 0.032940])

    freq = 10.0 # Hz, loop frequency
    N = 20 # horizon length

    model = export_drone_tether_ode_model()
    ocp.model = model

    Tf = N/freq
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    nsim = 1 # number of simulations
    ocp.solver_options.N_horizon = N

    # Set costs
    Q = np.eye(nx)
    R = np.eye(nu)
    Q[0,0] = 5.0     # x
    Q[1,1] = 5.0     # y
    Q[2,2] = 5.0     # z
    Q[3,3] = 0.0      # phi
    Q[4,4] = 0.0      # theta
    Q[5,5] = 1.0      # psi
    Q[6,6] = 10.0      # vwx
    Q[7,7] = 10.0      # vwy
    Q[8,8] = 10.0      # vwz
    Q[9,9] = 5.0      # p
    Q[10,10] = 5.0    # q
    Q[11,11] = 5.0    # r
    Q[12,12] = 1.0    # l_tet

    R[0,0] = 1.0    # tau_phi_cmd
    R[1,1] = 1.0    # tau_theta_cmd
    R[2,2] = 1.0    # tau_psi_cmd
    R[3,3] = 8.0   # thrust
    R[4,4] = 1.0    # l_tet_cmd

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = scipy.linalg.block_diag(Q) # caution: acados already increases the cost for the terminal state

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

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

    tet_sag_eps = 0.1 # margin to have some slack and not a full taught tether
    dist_gs_drone = sqrt(model.x[0]**2 + model.x[1]**2 + model.x[2]**2)
    tet_constraint = model.x[12] - dist_gs_drone - tet_sag_eps
    ocp.model.con_h_expr = vertcat(tet_constraint)
    ocp.constraints.lh = np.array([0.0])
    ocp.constraints.uh = np.array([50.0])

    ocp.model.con_h_expr_e = vertcat(tet_constraint)
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([10.0])

    POS_W_MAX = 100.0
    VEL_W_MAX = 15.0
    ANG_RATE_MAX = 10.0
    TET_LEN_MIN = 0.1
    TET_LEN_MAX = 50.0
    TAU_MAX = 10.0

    # set constraints
    lbu = np.array([-TAU_MAX, -TAU_MAX, -TAU_MAX, 0.0, 0.1])
    ubu = np.array([TAU_MAX, TAU_MAX, TAU_MAX, 1.0, 50.0])
    lbx = np.array([-POS_W_MAX, -POS_W_MAX, -2.0, -pi/4, -pi/4,
        -pi, -VEL_W_MAX, -VEL_W_MAX, -VEL_W_MAX, -ANG_RATE_MAX, -ANG_RATE_MAX, -ANG_RATE_MAX, TET_LEN_MIN])
    ubx = np.array([POS_W_MAX, POS_W_MAX, POS_W_MAX, +pi/4, +pi/4,
        +pi, VEL_W_MAX, VEL_W_MAX, VEL_W_MAX, ANG_RATE_MAX, ANG_RATE_MAX, ANG_RATE_MAX, TET_LEN_MAX])
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = np.arange(nx)
    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    x_init = x0.copy() # would be from sensor x_init = get_state_from_px4()
    u_init = u0.copy() # just initial guess for input
    for i in range(N):
        ocp_solver.set(i, "x", x0)
        ocp_solver.set(i, "u", u0)
    ocp_solver.set(N, "x", x0)
    theta_ref = 0.0
    for nsim in range(nsim):
        # constraints state
        ocp_solver.constraints_set(0, 'lbx', x0) # == data from sensor
        ocp_solver.constraints_set(0, 'ubx', x0)

        # set circular ref
        if(moving_ref):
            yref[0] = 1.0*cos(theta_ref)
            yref[1] = 1.0*sin(theta_ref)
            yref[2] = 1.0
            for i in range(N):
                ocp_solver.set(i, "yref", yref)
            ocp_solver.set(N, "yref", yref[:nx+1])

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
        u_init = ocp_solver.get(1, "u")
        print("new x0:", x_init)
        print("new u0:", u_init)
        theta_ref += 0.01

    plot_drone_tet_gpt_eval(np.linspace(0, Tf, N+1), 10.0, 0.1, simU, simX, latexify=True)

if __name__ == "__main__":
    solve_ocp()