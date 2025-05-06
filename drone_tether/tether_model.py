from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, tan, atan2, sqrt, fmax, if_else, fabs
import math

def export_drone_tether_ode_model() -> AcadosModel:

    model_name = 'drone_tether_ode'

    # constants
    g = 9.81 # gravity constant [m/s^2]
    # drone (based on tarot680 sim model)
    m_ce = 0.5 # mass of the center of the drone [kg]
    r_ce = 0.2 # approximated radius of the center of the drone  [m]
    m_ro = 0.005 # mass of one rotor [kg]
    n_ro = 6 # number of rotors
    m_dr = 1.616 # mass of the drone [kg] SIMULATION
    l_dr = 0.43 # length of arm of the drone [m]
    k_t = 41.93 # 1.616*9.81/0.378, pwm to thrust conversion factor, SIMULATION
    # tether
    rho_te = 970 # density of the tether [kg/m^3]
    A_te = 0.0000000078 # cross-sectional area of the tether [m^2]
    tau_l = 0.4 # time constant of the tether length, RANDOM [s]
    # Compute moments of inertia
    j_sphere = (1/2)*m_ce*r_ce**2
    j_motors = n_ro * m_ro*(l_dr**2)
    j_xx = j_sphere + j_motors/2
    j_yy = j_sphere + j_motors/2
    j_zz = j_sphere + j_motors

    # Inertia matrix
    J = SX.zeros(3, 3)
    J[0, 0] = j_xx
    J[1, 1] = j_yy
    J[2, 2] = j_zz

    # set up states & controls
    # state x
    x_w         = SX.sym('x_w')
    y_w         = SX.sym('y_w')
    z_w         = SX.sym('z_w')
    phi         = SX.sym('phi')
    theta       = SX.sym('theta')
    psi         = SX.sym('psi')
    vx_w        = SX.sym('vx_w')
    vy_w        = SX.sym('vy_w')
    vz_w        = SX.sym('vz_w')
    p           = SX.sym('p')
    q           = SX.sym('q')
    r           = SX.sym('r')
    l_tet       = SX.sym('l_tet')

    x = vertcat(x_w, y_w, z_w, phi, theta, psi, vx_w, vy_w, vz_w, p, q, r, l_tet)

    # xdot
    x_dot       = SX.sym('x_dot')
    y_dot       = SX.sym('y_dot')
    z_dot       = SX.sym('z_dot')
    phi_dot     = SX.sym('phi_dot')
    theta_dot   = SX.sym('theta_dot')
    psi_dot     = SX.sym('psi_dot')
    vx_dot      = SX.sym('vx_dot')
    vy_dot      = SX.sym('vy_dot')
    vz_dot      = SX.sym('vz_dot')
    p_dot       = SX.sym('p_dot')
    q_dot       = SX.sym('q_dot')
    r_dot       = SX.sym('r_dot')
    l_tet_dot   = SX.sym('l_tet_dot')

    xdot = vertcat(x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, vx_dot, vy_dot, vz_dot, p_dot, q_dot, r_dot, l_tet_dot)

    # input u
    tau_x = SX.sym('tau_x') # Normalized thrust command
    tau_y = SX.sym('tau_y') # roll command [rad]
    tau_z = SX.sym('tau_z') # pitch command [rad]
    thrust = SX.sym('thrust') # thrust command in PWM
    l_tet_cmd = SX.sym('l_tet_cmd') # length of the tether [m]

    u = vertcat(tau_x, tau_y, tau_z, thrust, l_tet_cmd)

    # spherical angles
    eps = 1e-8
    theta_s = atan2(if_else(sqrt(x_w**2 + y_w**2) < eps, eps, sqrt(x_w**2 + y_w**2)),
                     if_else(z_w**2 < eps, eps, z_w**2)) # world frame polar angle of the tether [rad]
    phi_s = atan2(y_w, if_else(fabs(x_w) < eps, eps, x_w)) # world frame azimutal angle of the tether [rad]

    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)
    t_theta = tan(theta)
    s_theta_s = sin(theta_s)
    c_theta_s = cos(theta_s)
    s_phi_s = sin(phi_s)
    c_phi_s = cos(phi_s)

    # Rotation matrix body to world
    R_bw = SX.zeros(3, 3)
    R_bw[0, 0] = c_psi*c_theta
    R_bw[0, 1] = c_psi*s_theta*s_phi - s_psi*c_phi
    R_bw[0, 2] = c_psi*s_theta*c_phi + s_psi*s_phi
    R_bw[1, 0] = s_psi*c_theta
    R_bw[1, 1] = s_psi*s_theta*s_phi + c_psi*c_phi
    R_bw[1, 2] = s_psi*s_theta*c_phi - c_psi*s_phi
    R_bw[2, 0] = -s_theta
    R_bw[2, 1] = c_theta*s_phi
    R_bw[2, 2] = c_theta*c_phi

    # Inverse rotation matrix for angular velocity ZYX
    Rw_bw = SX.zeros(3, 3)
    Rw_bw[0, 0] = 1
    Rw_bw[0, 1] = s_phi*t_theta
    Rw_bw[0, 2] = c_phi*t_theta
    Rw_bw[1, 0] = 0
    Rw_bw[1, 1] = c_phi
    Rw_bw[1, 2] = -s_phi
    Rw_bw[2, 0] = 0
    Rw_bw[2, 1] = s_phi/c_theta
    Rw_bw[2, 2] = c_phi/c_theta

    # Force propellers
    e_prop = vertcat(0, 0, 1) # body frame unit vector in the direction of the thrust
    F_prop = k_t*thrust*R_bw @ e_prop # thrust force in world frame

    # Force tether
    e_tet = vertcat(s_theta_s*c_phi_s, s_theta_s*s_phi_s, c_theta_s) # cartesian unit vector in the direction of the tether reaction (-) force
    f_winch = 1.0 # NOT CONSIDERING WINCH FORCE ATM
    F_grav_tet = (f_winch + g*rho_te*A_te*l_tet)*e_tet # tether force in world frame
    F_tet = F_grav_tet
    # gravity
    gravity = vertcat(0, 0, -g) # gravity force in world frame

    # dynamics
    f_expl = vertcat(vx_w, vy_w, vz_w,
                     Rw_bw @ vertcat(p, q, r),
                     gravity + 1/m_dr*(F_prop - F_tet),
                     1/J[0, 0]*(tau_x - (J[2, 2] - J[1, 1])*q*r),
                     1/J[1, 1]*(tau_y - (J[0, 0] - J[2, 2])*p*r),
                     1/J[2, 2]*(tau_z - (J[1, 1] - J[0, 0])*p*q),
                     1/tau_l*(l_tet_cmd - l_tet)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]', '$z$ [m]', r'$\phi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]',
                      '$\dot{x}$ [m/s]', '$\dot{y}$ [m/s]', '$\dot{z}$ [m/s]', r'$\dot{\phi}$ [rad/s]', r'$\dot{\theta}$ [rad/s]',
                      r'$\dot{\psi}$ [rad/s]', '$l_{tet}$ [m]']
    model.u_labels = ['$\tau_x$ [N]', r'$\tau_y$ [rad]', r'$\tau_z$ [rad]', '$t$ [s]', '$l_{tet}^{cmd}$ [m]']
    model.t_label = '$t$ [s]'

    return model

def export_drone_fo_model() -> AcadosModel:

    model_name = 'drone_tether_fo_model'

    # constants
    g = 9.81 # gravity constant [m/s^2]
    # drone
    m_dr = 1.616 # mass of the drone [kg]
    k_t = 41.93 #0.378 # pwm to thrust conversion factor
    # dynamics
    tau_phi = 0.2
    tau_theta = 0.3
    tau_psi = 0.22

    # state x
    x_w         = SX.sym('x_w')
    y_w         = SX.sym('y_w')
    z_w         = SX.sym('z_w')
    phi         = SX.sym('phi')
    theta       = SX.sym('theta')
    psi         = SX.sym('psi')
    vx_w        = SX.sym('vx_w')
    vy_w        = SX.sym('vy_w')
    vz_w        = SX.sym('vz_w')

    x = vertcat(x_w, y_w, z_w, phi, theta, psi, vx_w, vy_w, vz_w)

    # xdot
    x_dot       = SX.sym('x_dot')
    y_dot       = SX.sym('y_dot')
    z_dot       = SX.sym('z_dot')
    phi_dot     = SX.sym('phi_dot')
    theta_dot   = SX.sym('theta_dot')
    psi_dot     = SX.sym('psi_dot')
    vx_dot      = SX.sym('vx_dot')
    vy_dot      = SX.sym('vy_dot')
    vz_dot      = SX.sym('vz_dot')

    xdot = vertcat(x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, vx_dot, vy_dot, vz_dot)

    # input u
    phi_cmd = SX.sym('phi_cmd') # Normalized thrust command
    theta_cmd = SX.sym('theta_cmd') # roll command [rad]
    psi_cmd = SX.sym('psi_cmd') # pitch command [rad]
    thrust = SX.sym('thrust') # thrust command in PWM

    u = vertcat(phi_cmd, theta_cmd, psi_cmd, thrust)

    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)

    # Rotation matrix body to world
    R_bw = SX.zeros(3, 3)
    R_bw[0, 0] = c_psi*c_theta
    R_bw[0, 1] = c_psi*s_theta*s_phi - s_psi*c_phi
    R_bw[0, 2] = c_psi*s_theta*c_phi + s_psi*s_phi
    R_bw[1, 0] = s_psi*c_theta
    R_bw[1, 1] = s_psi*s_theta*s_phi + c_psi*c_phi
    R_bw[1, 2] = s_psi*s_theta*c_phi - c_psi*s_phi
    R_bw[2, 0] = -s_theta
    R_bw[2, 1] = c_theta*s_phi
    R_bw[2, 2] = c_theta*c_phi

    # Force propellers
    e_prop = vertcat(0, 0, 1) # body frame unit vector in the direction of the thrust
    F_prop = k_t*thrust*R_bw @ e_prop # thrust force in world frame

    # gravity
    gravity = vertcat(0, 0, -g) # gravity force in world frame

    # dynamics
    f_expl = vertcat(vx_w, vy_w, vz_w,
                     1/tau_phi*(phi - phi_cmd),
                     1/tau_theta*(theta - theta_cmd),
                     1/tau_psi*(psi -psi_cmd),
                     gravity[0] + 1/m_dr*F_prop[0],
                     gravity[1] + 1/m_dr*F_prop[1],
                     gravity[2] + 1/m_dr*F_prop[2],
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]', '$z$ [m]', r'$\phi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]',
                      '$\dot{x}$ [m/s]', '$\dot{y}$ [m/s]', '$\dot{z}$ [m/s]', r'$\dot{\phi}$ [rad/s]', r'$\dot{\theta}$ [rad/s]',
                      r'$\dot{\psi}$ [rad/s]', '$l_{tet}$ [m]']
    model.u_labels = ['$\tau_x$ [N]', r'$\tau_y$ [rad]', r'$\tau_z$ [rad]', '$t$ [s]', '$l_{tet}^{cmd}$ [m]']
    model.t_label = '$t$ [s]'

    return model

