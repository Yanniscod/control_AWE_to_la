from acados_template import AcadosOcp, AcadosOcpSolver
from tether_model import export_drone_tether_ode_model_gpt
import numpy as np
import scipy.linalg
from utils import plot_drone_tet_gpt_eval
from casadi import vertcat, sin, cos, pi, sqrt, fmax, log, exp
import time

def solve_ocp():
    ocp = AcadosOcp()

    model = None
    moving_ref = False
    init_pose_ref = np.array([0.0, 0.0, 1.0, 0.0, 0.2, 1.0]) # x_w, y_w, z_w

    model = export_drone_tether_ode_model_gpt()
    ocp.model = model

    Tf = 1.0
    N = 50 # horizon
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    nsim = 1 # number of simulations
    ocp.solver_options.N_horizon = N

    # Set costs
    Q = np.eye(nx)
    R = np.eye(nu)
    Q[0,0] = 10.0     # x
    Q[1,1] = 10.0     # y
    Q[2,2] = 10.0     # z
    Q[3,3] = 10.0      # phi
    Q[4,4] = 10.0      # theta
    Q[5,5] = 4.0      # psi
    Q[6,6] = 1.0      # vwx
    Q[7,7] = 1.0      # vwy
    Q[8,8] = 1.0      # vwz
    Q[9,9] = 1.0      # p
    Q[10,10] = 1.0    # q
    Q[11,11] = 1.0    # r
    Q[12,12] = 1.0    # l_tet

    R[0,0] = 1.0    # tau_phi_cmd
    R[1,1] = 1.0    # tau_theta_cmd
    R[2,2] = 1.0    # tau_psi_cmd
    R[3,3] = 10.0   # thrust
    R[4,4] = 1.0    # l_tet_cmd

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = 40*scipy.linalg.block_diag(Q)

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    yref = np.zeros((ny, ))
    yref[0] = init_pose_ref[0] # x_w
    yref[1] = init_pose_ref[1] # y_w
    yref[2] = init_pose_ref[2] # z_w
    yref[3] = init_pose_ref[3] # phi
    yref[4] = init_pose_ref[4] # theta
    yref[5] = init_pose_ref[5] # psi
    ocp.cost.yref = yref
    yref_e = np.zeros((ny_e, ))
    yref_e[0] = init_pose_ref[0] # x_w
    yref_e[1] = init_pose_ref[1] # y_w
    yref_e[2] = init_pose_ref[2] # z_w
    yref_e[3] = init_pose_ref[3] # phi
    yref_e[4] = init_pose_ref[4] # theta
    yref_e[5] = init_pose_ref[5] # psi
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_e = vertcat(ocp.model.x)

    tet_sag_eps = 0.05 # margin to have some slack and not a full taught tether
    dist_gs_drone = sqrt(model.x[0]**2 + model.x[1]**2 + model.x[2])
    tet_constraint = model.x[12] - dist_gs_drone - tet_sag_eps
    ocp.model.con_h_expr = vertcat(tet_constraint)
    ocp.constraints.lh = np.array([0.0])
    ocp.constraints.uh = np.array([50.0])

    ocp.model.con_h_expr_e = vertcat(tet_constraint)
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([10.0])

    # set constraints
    lbu = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    ubu = np.array([5.0, 5.0, 5.0, 1.0, 50.0])
    lbx = np.array([-200.0, -200.0, 0.0, -pi/4, -pi/4,
        -pi, -10.0, -10.0, -10.0,-10.0, -10.0, -10.0, 0.1])
    ubx = np.array([+200.0, +200.0, +200.0, +pi/4, +pi/4,
        +pi, +10.0, +10.0, +10.0, +10.0, +10.0, +10.0, +50.0])
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx
    ocp.constraints.idxbx = np.arange(nx)
    
    x0 = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.7])
    u0 = np.array([0.0, 0.0, 0.0, 0.4, 0.7])
    ocp.constraints.x0 = x0

    # set options, not differentiating between models atm
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.tf = Tf
    # ocp.solver_options.tol = 1e-4

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
        # x_init = get_state_from_px4()
        ocp_solver.constraints_set(0, 'lbx', x0) # == data from sensor
        ocp_solver.constraints_set(0, 'ubx', x0)

        # yref = get_target_from_px4()
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
        u_init = ocp_solver.get(0, "u")
        print("new x0:", x_init)
        print("new u0:", u_init)
        theta_ref += 0.01

    plot_drone_tet_gpt_eval(np.linspace(0, Tf, N+1), 10.0, 0.1, simU, simX, latexify=True)

if __name__ == "__main__":
    solve_ocp()