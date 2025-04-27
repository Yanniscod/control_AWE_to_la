from acados_template import AcadosOcp, AcadosOcpSolver, ocp_get_default_cmake_builder
from tether_model import export_drone_tether_fo_model
import numpy as np
import scipy.linalg
from utils import plot_drone_tet_fo_eval
from casadi import SX, vertcat, Function

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_drone_tether_fo_model()
ocp.model = model

Tf = 1.0
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu
ny_e = nx
N = 50

# set dimensions
ocp.solver_options.N_horizon = N

# set cost
Q = np.eye(nx)
Q[0,0] = 1.0     # x
Q[1,1] = 1.0     # y
Q[2,2] = 1.0        # z
Q[3,3] = 1.0     # phi
Q[4,4] = 1.0     # theta
Q[5,5] = 1.0     # psi
Q[6,6] = 0.0       # vwx
Q[7,7] = 1.0        # vwy
Q[8,8] = 1.0        # vwz
Q[9,9] = 1.0     # l_tet

R = np.eye(nu)
R[0,0] = 0.0    # phi_cmd
R[1,1] = 0.0    # theta_cmd
R[2,2] = 0.0    # psi_cmd
R[3,3] = 1.0    # thrust
R[4,4] = 1.0    # l_tet_cmd

ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = 50*Q

ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.cost.cost_type_e = 'NONLINEAR_LS'

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx)

Vu = np.zeros((ny, nu))
Vu[nx:, :] = np.eye(nu)
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.eye(nx)

yref = np.zeros((ny, ))
yref[2] = 1.0 # z_w
ocp.cost.yref = yref
yref_e = np.zeros((ny_e, ))
yref_e[2] = 1.0 # z_w
ocp.cost.yref_e = yref_e

ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
ocp.model.cost_y_expr_e = vertcat(ocp.model.x)

# set constraints
l_tet_max = 50 # [m]
l_tet_min = 0.1 # [m]
ocp.constraints.lbu = np.array([-np.pi/4, -np.pi/4, -np.pi, 0.0, l_tet_min])
ocp.constraints.ubu = np.array([np.pi/4, np.pi/4, np.pi, 1.0, l_tet_max])
ocp.constraints.idxbu = np.arange(nu)

l_tet_init = 1.5
ocp.constraints.x0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, l_tet_init])
x0 = ocp.constraints.x0
u0 = np.array([0.0, 0.0, 0.0, 0.5, 1.5])
ocp.constraints.lbx = np.array([-200.0, -200.0, 0.0, -np.pi/4, -np.pi/4,
    -np.pi, -10.0, -10.0, -10.0, l_tet_min])
ocp.constraints.ubx = np.array([+200.0, +200.0, +200.0, +np.pi/6, +np.pi/6,
    +np.pi, +10.0, +10.0, +10.0, l_tet_max])
ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

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

# set prediction horizon
ocp.solver_options.tf = Tf

# use the CMake build pipeline
cmake_builder = ocp_get_default_cmake_builder()

ocp_solver = AcadosOcpSolver(ocp)

simX = np.zeros((N+1, nx))
simU = np.zeros((N, nu))

for i in range(N):
    ocp_solver.set(i, "x", x0)
    ocp_solver.set(i, "u", u0)
ocp_solver.set(N, "x", x0)

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
plot_drone_tet_fo_eval(np.linspace(0, Tf, N+1), np.pi/4, l_tet_min, simU, simX, latexify=True)
