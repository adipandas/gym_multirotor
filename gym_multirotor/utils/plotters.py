import numpy as np
import matplotlib.pyplot as plt
from gym_multirotor.utils.rotation_transformations import quat2euler


def plot_xyz(time,
             state,
             super_title='X-Y-Z plots',
             ylabels=('$x$', '$y$', '$z$'),
             xlabel='time',
             save_as='XYZ.pdf',
             lineWidth=0.5
             ):
    fig, axs = plt.subplots(3, sharex='all')

    for i in range(3):
        axs[i].plot(time, state[:, i], lineWidth=lineWidth)
        axs[i].grid(True)
        axs[i].set(xlabel=xlabel, ylabel=ylabels[i])
        # axs[i].set_ylim(-5, 5)
        # axs[i].set_xticks(time[::200])
        axs[i].set_xlim(time[0], time[-1])
        axs[i].minorticks_on()

    for ax in axs.flat:
        ax.label_outer()

    if super_title is not None:
        fig.suptitle(super_title)

    if save_as is not None:
        plt.savefig(save_as, dpi=150)


def plot_rpy(time, state, super_title="Roll-Pitch-Yaw", save_as='roll_pitch_yaw.pdf', lineWidth=2):
    quats = state[:, 3:7]

    euler = []
    for q in quats:
        euler.append(quat2euler(q))
    euler = np.array(euler)

    fig, axs = plt.subplots(3, sharex='all')

    ylabels = ['$\phi (^{\circ})$', '$\\theta (^{\circ})$', '$\psi (^{\circ})$']
    for i in range(3):
        axs[i].plot(time, euler[:, i] * 180 / np.pi, lineWidth=lineWidth)
        axs[i].grid(True)
        axs[i].set(xlabel='time', ylabel=ylabels[i])

        # axs[i].set_yticks(np.arange(-15, 15))
        # axs[i].minorticks_on()
        # axs[i].tick_params(axis='y', labelsize=5)
        # axs[i].tick_params(axis='x', labelsize=5)

        axs[i].set_xlim(time[0], time[-1])

    for ax in axs.flat:
        ax.label_outer()

    if super_title is not None:
        fig.suptitle(super_title)

    if save_as is not None:
        plt.savefig(save_as, dpi=150)


def plot_action_tiltrotor(time, action, set_super_title=True, save_as=('thrust.pdf', 'tilt.pdf'), lineWidth=0.5,
                          super_title=('Config1-Thrust', 'Config1-Tilt')):
    fig, axs = plt.subplots(4, sharex='all')

    ylabels = ['$rotor_{front}$', '$rotor_{left}$', '$rotor_{back}$', '$rotor_{right}$',
               '$\\theta^{tilt}_{front} (^{\circ})$', '$\\theta^{tilt}_{left} (^{\circ})$',
               '$\\theta^{tilt}_{back} (^{\circ})$', '$\\theta^{tilt}_{right} (^{\circ})$']

    for i in range(4):
        axs[i].plot(time, action[:, i], lineWidth=lineWidth)
        axs[i].grid(True)
        axs[i].set(xlabel='time', ylabel=ylabels[i])
        axs[i].set_ylim(0, 1.2)
        # axs[i].set_xticks(time[::200])
        axs[i].set_xlim(time[0], time[-1])
        axs[i].minorticks_on()

    for ax in axs.flat:
        ax.label_outer()

    if super_title is not None:
        fig.suptitle(super_title[0])

    # if set_super_title:
    #     fig.suptitle('Thrust')

    if save_as is not None:
        plt.savefig(save_as[0], dpi=150)

    fig, axs = plt.subplots(4, sharex='all')
    for i in range(4):
        axs[i].plot(time, action[:, i + 4] * 1.04 * 180. / np.pi, lineWidth=lineWidth)
        axs[i].grid(True)
        axs[i].set(xlabel='time', ylabel=ylabels[i + 4])
        axs[i].set_ylim(-61, 62)
        axs[i].set_xlim(time[0], time[-1])
        # axs[i].set_xticks(time[::200])
        axs[i].minorticks_on()

    for ax in axs.flat:
        ax.label_outer()

    # if set_super_title:
    #     fig.suptitle('Tilt')

    if super_title is not None:
        fig.suptitle(super_title[1])

    if save_as is not None:
        plt.savefig(save_as[1], dpi=150)

