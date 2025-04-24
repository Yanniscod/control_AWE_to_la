from acados_template import AcadosOcp, AcadosOcpSolver, ocp_get_default_cmake_builder
from tether_model import export_drone_tether_ode_model
import numpy as np
import scipy.linalg
from utils import plot_drone_tet_eval
from casadi import SX, vertcat

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_drone_tether_ode_model()
ocp.model = model

Tf = 1.0
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu
ny_e = 6
N = 20

# set dimensions
ocp.solver_options.N_horizon = N

# set cost
Q_p = 2*np.diag([1e3, 1e3, 1e-2]) # position
Q_ori = 2*np.diag([1e3, 1e3, 1e-2]) # rpy
Q_v = 2*np.diag([1e3, 1e3, 1e-2]) # linear vel
Q_w = 2*np.diag([1e-2, 1e-2, 1e-2]) # angular vel
q_l = 2*1e-2 # cost on tether length
Q_u = 2*np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

ocp.cost.W_e = scipy.linalg.block_diag(Q_p, Q_ori)
ocp.cost.W = scipy.linalg.block_diag(Q_p, Q_ori, Q_v, Q_w, q_l, Q_u)

ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.cost.cost_type_e = 'NONLINEAR_LS'

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx)

Vu = np.zeros((ny, nu))
Vu[4,0] = 1.0
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.eye(nx)

x_w = model.x[0]
y_w = model.x[1]
z_w = model.x[2]
dist_gs_drone = np.sqrt(x_w**2 + y_w**2 + z_w**2)
epsilon = 0.3
l_tet_ref = 0.0 #dist_gs_drone - epsilon

yref = np.zeros((ny, ))
yref[12] = l_tet_ref # l_tet
ocp.cost.yref = yref
ocp.cost.yref_e = np.zeros((ny_e, ))

ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
ocp.model.cost_y_expr_e = vertcat(ocp.model.x[:6])

# set constraints
tau_max = 10 # [N*m]
l_tet_max = 100 # [m]
l_tet_min = 0.1 # [m]
ocp.constraints.lbu = np.array([-tau_max, -tau_max, -tau_max, 0.0, l_tet_min])
ocp.constraints.ubu = np.array([tau_max, tau_max, tau_max, 1.0, l_tet_max])
ocp.constraints.idxbu = np.arange(nu)

l_tet_init = 2.5
x_pos_init = (0.3, 0.3, 0.2) # init world pose [m]
x_orient_init = (0.0, 0.0, 0.0) # rpy [deg]
x_speed_init = (0.0, 0.0, 0.0) # init world speed [m/s]
# should init from last sensor measurements <-------------------------------------------------------------------------------------------------
ocp.constraints.x0 = np.array([x_pos_init[0], x_pos_init[1], x_pos_init[2], x_orient_init[0], x_orient_init[1], x_orient_init[2], x_speed_init[0], x_speed_init[1], x_speed_init[2], 0.0, 0.0, 0.0, l_tet_init])

ocp.parameter_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ocp.constraints.lbx = np.array([-200.0, -200.0, 0.0, -np.pi/6, -np.pi/6, 0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, l_tet_min])
ocp.constraints.ubx = np.array([200.0, 200.0, 200.0, np.pi/6, np.pi/6, np.pi, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, l_tet_max])
ocp.constraints.idxbx = np.arange(nx)

# set options
ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.print_level = 1
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

# set prediction horizon
ocp.solver_options.tf = Tf

# use the CMake build pipeline
cmake_builder = ocp_get_default_cmake_builder()

ocp_solver = AcadosOcpSolver(ocp)

simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))

vx = 0.0
vy = 0.0
vz = 0.0
p = 0.0
q = 0.0
r = 0.0
# theta_s = 0.0
# phi_s = 0.0

for stage in range(N):
    # set yref for tether_length
    ocp_solver.set(stage, "p", np.array([vx, vy, vz, p, q, r]))

status = ocp_solver.solve()

if status != 0:
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
    raise Exception(f'acados returned status {status}.')

# get solution
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

# to implem
plot_drone_tet_eval(np.linspace(0, Tf, N+1), tau_max, l_tet_min, simU, simX, latexify=True)
