#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_drone_ode_model() -> AcadosModel:

    model_name = 'drone_ode'

    # constants
    g = 9.81 # gravity constant [m/s^2]
    # drone
    m_ce = 0.5 # mass of the center of the drone [kg]
    r_ce = 0.1 # approximated radius of the center of the drone  [m]
    m_ro = 0.5 # mass of one rotor [kg]
    n_ro = 6 # number of rotors
    m_dr = 1.5 # mass of the drone [kg]
    l_dr = 0.8 # length of arm of the drone [m]
    # tether
    rho_te = 0.1 # density of the tether [kg/m^3]
    A_te = 0.01 # cross-sectional area of the tether [m^2]


    # set up states & controls
    # state x
    x           = SX.sym('x')
    y           = SX.sym('y')
    z           = SX.sym('z')
    phi         = SX.sym('phi')
    theta       = SX.sym('theta')
    psi         = SX.sym('psi')
    vx          = SX.sym('vx')
    vy          = SX.sym('vy')
    vz          = SX.sym('vz')
    dphi        = SX.sym('dphi')
    dtheta      = SX.sym('dtheta')
    dpsi        = SX.sym('dpsi')

    x = vertcat(x, y, z, phi, theta, psi, vx, vy, vz, dphi, dtheta, dpsi)

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
    dphi_dot    = SX.sym('dphi_dot')
    dtheta_dot  = SX.sym('dtheta_dot')
    dpsi_dot    = SX.sym('dpsi_dot')

    xdot = vertcat(x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, vx_dot, vy_dot, vz_dot, dphi_dot, dtheta_dot, dpsi_dot)

    # input u
    thrust_dr = SX.sym('thrust_dr') # Normalized thrust command
    phi_cmd = SX.sym('phi_cmd') # roll command [rad]
    theta_cmd = SX.sym('theta_cmd') # pitch command [rad]
    psi_cmd = SX.sym('psi_cmd') # yaw command [rad]
    l_tether = SX.sym('l_tether') # length of the tether [m]

    u = vertcat(phi_cmd, theta_cmd, psi_cmd, thrust_dr, l_tether)

    # udot
    phi_dot      = SX.sym('phi_dot')
    theta_dot   = SX.sym('theta_dot')
    psi_dot      = SX.sym('psi_dot')
    thrust_dot  = SX.sym('thrust_dot')
    l_tether_dot  = SX.sym('l_tether_dot')

    udot = vertcat(phi_dot, theta_dot, psi_dot, thrust_dot, l_tether_dot)

    # disturbances
    f_win = SX.sym('f_win') # winch reaction force [N]
    tau_x = SX.sym('tau_x') # roll torque [Nm]
    tau_y = SX.sym('tau_y') # pitch torque [Nm]
    tau_z = SX.sym('tau_z') # yaw torque [Nm]

    d = vertcat(f_win, tau_x, tau_y, tau_z)
    
    # Augmented dynamics
    x_aug = vertcat(x, u)
    x_aug_dot = vertcat(xdot, udot)

    # dynamics
    f_expl = vertcat(vx,
                     vy,
                     vz,
                     dphi,
                     dtheta,
                     dpsi,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
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
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

