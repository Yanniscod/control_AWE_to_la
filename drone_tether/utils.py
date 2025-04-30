import matplotlib.pyplot as plt
from acados_template import latexify_plot

def plot_drone_tet_gpt_eval(time, tau_max, l_tet_min,  simU, simX, latexify=False):
    if latexify:
        latexify_plot()
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)

    #################### INPUTS ####################
    # Plot control inputs (tau)
    axs[0,0].plot(time[:-1], simU[:, 0], label=r'$\tau_{\phi}$')
    axs[0,0].plot(time[:-1], simU[:, 1], label=r'$\tau_{\theta}$')
    axs[0,0].plot(time[:-1], simU[:, 2], label=r'$\tau_{\psi}$')
    axs[0,0].axhline(tau_max, color='gray', linestyle='--', label='bounds')
    axs[0,0].axhline(-tau_max, color='gray', linestyle='--')
    axs[0,0].set_ylabel("tau input [Nm]")
    axs[0,0].legend()
    axs[0,0].grid(True)

    # Plot thrust input
    axs[1,0].plot(time[:-1], simU[:, 3], label=r'$thrust$')
    axs[1,0].axhline(0.0, color='gray', linestyle='--', label='')
    axs[1,0].axhline(1.0, color='gray', linestyle='--', label='')
    axs[1,0].set_ylabel("thrust input")
    axs[1,0].set_xlabel("Time [s]")
    axs[1,0].legend()
    axs[1,0].grid(True)

    # Plot tether length command
    axs[2,0].plot(time[:-1], simU[:, 4], label=r'$l_{\mathrm{tether_{cmd}}}$')
    axs[2,0].axhline(l_tet_min, color='gray', linestyle='--', label='')
    axs[2,0].set_ylabel("Tether Length [m]")
    axs[2,0].legend()
    axs[2,0].grid(True)
    fig.suptitle('Input commands', fontsize=14)

    #################### STATES ####################
    # Plot drone position
    axs[0,1].plot(time, simX[:, 0], label='x')
    axs[0,1].plot(time, simX[:, 1], label='y')
    axs[0,1].plot(time, simX[:, 2], label='z')
    axs[0,1].set_ylabel("Position [m]")
    axs[0,1].legend()
    axs[0,1].grid(True)

    # Plot drone orientation
    axs[1,1].plot(time, simX[:, 3], label='phi')
    axs[1,1].plot(time, simX[:, 4], label='theta')
    axs[1,1].plot(time, simX[:, 5], label='psi')
    axs[1,1].set_ylabel("Orientation [deg]")
    axs[1,1].legend()
    axs[1,1].grid(True)

    # Plot l_tet
    axs[2,1].plot(time, simX[:, 12], label='l_tet')
    axs[2,1].set_ylabel("tether length [m]")
    axs[2,1].set_xlabel("Time [s]")
    axs[2,1].legend()
    axs[2,1].grid(True)
    fig.suptitle('States', fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_drone_tet_fo_eval(time, rpy_max,  simU, simX, latexify=False):
    if latexify:
        latexify_plot()
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # Plot control inputs (tau)
    axs[0].plot(time[:-1], simU[:, 0], label=r'$\phi_{cmd}$')
    axs[0].plot(time[:-1], simU[:, 1], label=r'$\theta_{cmd}$')
    axs[0].plot(time[:-1], simU[:, 2], label=r'$\psi_{cmd}$')
    axs[0].axhline(rpy_max, color='gray', linestyle='--', label='RPY bounds')
    axs[0].axhline(-rpy_max, color='gray', linestyle='--')
    axs[0].set_ylabel("RPY [rad]")
    axs[0].legend()
    axs[0].grid(True)

    # Plot thrust command
    axs[1].plot(time[:-1], simU[:, 3], label=r'$\mathrm{thrust_{cmd}}$')
    axs[1].axhline(0.0, color='gray', linestyle='--', label='')
    axs[1].axhline(1.0, color='gray', linestyle='--', label='Thrust bound')
    axs[1].set_ylabel("Thrust cmd")
    axs[1].legend()
    axs[1].grid(True)

    # Plot drone position
    axs[2].plot(time, simX[:, 0], label='x')
    axs[2].plot(time, simX[:, 1], label='y')
    axs[2].plot(time, simX[:, 2], label='z')
    axs[2].set_ylabel("Position [m]")
    axs[2].legend()
    axs[2].grid(True)

    # Plot drone orientation
    axs[3].plot(time, simX[:, 3], label='phi')
    axs[3].plot(time, simX[:, 4], label='theta')
    axs[3].plot(time, simX[:, 5], label='psi')
    axs[3].set_ylabel("RPY [rad]")
    axs[3].set_xlabel("Time [s]")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
