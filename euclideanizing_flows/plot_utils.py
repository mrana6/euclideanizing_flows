from __future__ import print_function
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def sim_euler(dynamics, state_0, t_final, dt, t_init=None):
    '''
    Simulate dynamics via euler method
    TODO: convert this pytorch
    '''
    if t_init is not None:
        t = np.array([t_init])
        T = t_final - t_init
    else:
        t = np.array([0.])
        T = t_final

    state = state_0

    if state_0.ndim == 1:
        y = state_0.reshape(-1, 1)
        for i in np.arange(dt, T + dt, dt):
            state = state + dt * dynamics(t[-1], state)
            y = np.concatenate((y, state.reshape(-1, 1)), axis=1)
            t = np.concatenate((t, np.array([i])))
    else:
        y = state_0
        for i in np.arange(dt, T + dt, dt):
            state = state + dt * dynamics(t[-1], state)
            y = np.concatenate((y, state), axis=1)
            t = np.concatenate((t, np.array([i])))

    return t, y


def generate_trajectories(
        model, x_init, time_dependent=False, order=2,
        return_label=True, filename=None,
        t_init=0., t_final=10., t_step=0.01, method='ivp'):
    '''
    generate roll-out trajectories of the given dynamical model
    :param model (torch.nn.Module): dynamical model
    :param x_init (torch.Tensor): initial condition
    :param order (int): 1 if first order system, 2 if second order system
    :param return_label (bool): set True to return the velocity (order=1) / acceleration (order=2) along the trajectory
    :param filename (string): if not None, save the generated trajectories to the file
    :param t_init (float): initial time for numerical integration
    :param t_final (float): final time for numerical integration
    :param t_step (float): step for t_eval for numerical integration
    :param method (string): method of integration (ivp or euler)
    :return: state (torch.Tensor) if return_label is False
    :return: state (torch.Tensor), control (torch.Tensor) if return label is True
    '''

    # make sure that the initial condition has dim 1
    x_init = x_init.reshape(-1)

    # dynamics for numerical integration

    # dynamics for the first order system
    def dynamics(t, state):
        # for first order systems, the state is the position

        # compute the velocity under the given model
        state = torch.from_numpy(state).float()
        if time_dependent:
            y_pred = model(t, state)
        else:
            y_pred = model(state)

        # the time-derivative of the state is the velocity
        x_dot = y_pred.detach().squeeze().numpy()
        return x_dot

    # the times at which the trajectory is computed
    t_eval = np.arange(t_init, t_final, t_step)

    # integrate the trajectory numerically

    if method == 'ivp':
        sol = solve_ivp(dynamics, [t_init, t_final], x_init, t_eval=t_eval)
        x_data = torch.from_numpy(sol.y.T).float()
    elif method == 'euler':
        t, y = sim_euler(dynamics, x_init, t_final, t_step, t_init=t_init)
        x_data = torch.from_numpy(y.T).float()
    else:
        assert ValueError('Unknown integration method!')

    # if the control inputs along the trajectory are also needed,
    # compute the control inputs (useful for generating datasets)
    if return_label:
        y_data = model(x_data)
        data = (x_data, y_data)
    else:
        data = x_data

    if filename is not None:
        torch.save(data, filename)

    return data


def visualize_field(x1_coords, x2_coords, z_coords,
                    cmap='viridis', type='stream', color=None):
    '''
    plot the streamplot or quiver plot for 2-dim vector fields
    see matplotlib.pyplot.streamplot or matplotlib.pyplot.quiver for more information
    :param x1_coords (np.ndarray): x1 coordinates, created by torch.meshgrid
    :param x2_coords (np.ndarray): x2_coordinates, created by torch.meshgrid
    :param z_tensor (np.ndarray): the vector field to be visualized
    :param cmap (colormap): colormap of the plot
    :param type ('stream or 'quiver'): type of plot, streamplot or quiverplot
    :return:
    '''

    # reshape the tensor so that it has the same shape as x1_coords and x2_coords
    n_rows, n_cols = x1_coords.shape
    z_coords = z_coords.reshape(n_rows, n_cols, -1)
    n_dims = z_coords.shape[-1]

    # make sure that z has at least 2 dims to visualize
    assert n_dims >= 2

    # visualize the first 2 dimensions of z
    if type == 'stream':
        # create streamplot
        if color is None:
            color = np.linalg.norm(z_coords, axis=2)
        if cmap is not None:
            plt.streamplot(
                x1_coords, x2_coords,
                z_coords[:, :, 0], z_coords[:, :, 1],
                color=color, cmap=cmap
            )
        else:
            if color is None:
                color = '0.4'
            plt.streamplot(
                x1_coords, x2_coords,
                z_coords[:, :, 0], z_coords[:, :, 1],
                color=color, cmap=cmap
            )
    elif type == 'quiver':
        # create quiverplot
        if color is None:
            color = np.linalg.norm(z_coords, axis=2)
        plt.quiver(
            x1_coords, x2_coords,
            z_coords[:, :, 0], z_coords[:, :, 1],
            color, units='width', cmap=cmap
        )


def visualize_vel(model, x_lim=[[0, 1], [0, 1]], delta=0.05, cmap=None, color=None):
    '''
    visualize the velocity model (first order)
    similar to visualize_accel
    :param model (torch.nn.Module): the velocity model to be visualized
    :param x_lim (array-like): the range of the state-space (positions) to be sampled over
    :param delta (float): the step size for the sampled positions
    :return: None
    '''

    # generate a meshgrid of coordinates to test
    # note that we do x2, x1 due to a special requirement of the streamplot function
    # see matplotlib.pyplot.streamplot for more information
    x2_coords, x1_coords = torch.meshgrid(
        [torch.arange(x_lim[1][0], x_lim[1][1], delta),
         torch.arange(x_lim[0][0], x_lim[0][1], delta)])

    # generate a flat version of the coordinates for forward pass
    x_test = torch.zeros(x1_coords.nelement(), 2)
    x_test[:, 0] = x1_coords.reshape(-1)
    x_test[:, 1] = x2_coords.reshape(-1)

    # forward pass
    y_pred = model(x_test).detach()

    # visualize the velocity as a vector field (equivalent to the warped potential)
    visualize_field(
        x1_coords.numpy(),
        x2_coords.numpy(),
        y_pred.numpy(),
        cmap=cmap,
        color=color
    )
