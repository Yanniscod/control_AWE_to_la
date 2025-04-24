import matplotlib.pyplot as plt
from acados_template import latexify_plot

def plot_drone_tet_eval(time, tau_max, l_tet_min,  simU, simX, latexify=False):
    if latexify:
        latexify_plot()
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

    # Plot control inputs (tau)
    axs[0].plot(time[:-1], simU[:, 0], label=r'$\tau_1$')
    axs[0].plot(time[:-1], simU[:, 1], label=r'$\tau_2$')
    axs[0].plot(time[:-1], simU[:, 2], label=r'$\tau_3$')
    axs[0].axhline(tau_max, color='gray', linestyle='--', label='Torque bounds')
    axs[0].axhline(-tau_max, color='gray', linestyle='--')
    axs[0].set_ylabel("Torque [Nm]")
    axs[0].legend()
    axs[0].grid(True)

    # Plot tether length command
    axs[2].plot(time[:-1], simU[:, 3], label=r'$l_{\mathrm{tether_{cmd}}}$')
    axs[2].axhline(l_tet_min, color='gray', linestyle='--', label='Tether cmd bound')
    axs[2].set_ylabel("Tether Length [m]")
    axs[2].legend()
    axs[2].grid(True)

    # Plot tether length
    axs[1].plot(time, simX[:, -1], label=r'$l_{\mathrm{tether}}$')
    axs[1].set_ylabel("Tether Length [m]")
    axs[1].legend()
    axs[1].grid(True)

    # Plot drone position
    axs[3].plot(time, simX[:, 0], label='x')
    axs[3].plot(time, simX[:, 1], label='y')
    axs[3].plot(time, simX[:, 2], label='z')
    axs[3].set_ylabel("Position [m]")
    axs[3].set_xlabel("Time [s]")
    axs[3].legend()
    axs[3].grid(True)

    # Plot drone orientation
    axs[4].plot(time, simX[:, 3], label='phi')
    axs[4].plot(time, simX[:, 4], label='theta')
    axs[4].plot(time, simX[:, 5], label='psi')
    axs[4].set_ylabel("Orientation [deg]")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()
