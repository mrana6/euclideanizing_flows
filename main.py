'''
Main script for training stable dynamics using Euclideanizing flows on LASA handwriting dataset

Ref: M. Asif Rana et al, Euclideanizing Flows: Diffeomorphic Reduction for Learning Stable Dynamical Systems, L4DC 2020
(https://arxiv.org/pdf/2005.13143.pdf)
'''

from __future__ import print_function
import torch.optim as optim
from torch.utils.data import TensorDataset
from euclideanizing_flows.flows import *
from euclideanizing_flows.train_utils import *
from euclideanizing_flows.plot_utils import *
from euclideanizing_flows.data_utils import *
import argparse


parser = argparse.ArgumentParser(description='Euclideanizing flows for learning stable dynamical systems')

parser.add_argument(
    '--data-name',
    type=str,
    default='Leaf_2',
    help='name of the letter in LASA dataset')

args = parser.parse_args()

# ---------------
# params
data_name = args.data_name
test_learner_model = True               # to plot the rollouts and vector fields
load_learner_model = False              # to load a saved model
coupling_network_type = 'rffn'          # rffn/fcnn (specify random fourier features or neural network for coupling layer)
plot_resolution = 0.01                  # plotting resolution (only use for testing)

# -----------------------------------------------------------------------
# learner params (for normalizing flows)
if coupling_network_type == 'fcnn':         # neural network parameterization
    num_blocks = 7                          # number of coupling layers
    num_hidden = 100                        # hidden layer dimensions (there are two of hidden layers)
    # only for fcnn!
    t_act = 'elu'                           # activation fcn in each network (must be continuously differentiable!)
    s_act = 'elu'

    minibatch_mode = True                   # True uses the batch_size arg below
    batch_size = 64                         # size of minibatch
    learning_rate = 0.0005
    sigma = None                            # not required for fcnn
    print('WARNING: FCNN params are not tuned!! ')

elif coupling_network_type == 'rffn':       # random fourier features parameterization
    num_blocks = 10                         # number of coupling layers
    num_hidden = 200                        # number of random fourier features per block
    sigma = 0.45                            # length scale for random fourier features

    minibatch_mode = False
    batch_size = 64
    s_act = None                            # not required for rffn
    t_act = None                            # not required for rffn
    learning_rate = 0.0001                  # low learning rate helps!

else:
    raise TypeError('Coupling layer network not defined!')

# ------------------------------------------------------------------
# Training params

eps = 1e-12
no_cuda = True          # TODO: cuda compatibility not tested fully!
seed = None
weight_regularizer = 1e-10
epochs = 5000
loss_clip = 1e3
clip_gradient = True
clip_value_grad = 0.1
log_freq = 10
plot_freq = 200
stopping_thresh = 250

cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# ---------------------------------------------------------------
print('Loading dataset...')

dataset = LASA(data_name=data_name)
# dataset.plot_data()

goal = dataset.goal
idx = dataset.idx

x_train = dataset.x
xd_train = dataset.xd

scaling = torch.from_numpy(dataset.scaling).float()
translation = torch.from_numpy(dataset.translation).float()

normalize_ = lambda x: x * scaling + translation
denormalize_ = lambda x: (x - translation) / scaling

n_dims = dataset.n_dims
n_pts = dataset.n_pts

dt = dataset.dt

dataset_list = []
time_list = []
expert_traj_list = []
s0_list = []
t_final_list = []
for n in range(len(idx) - 1):
    x_traj_tensor = torch.from_numpy(x_train[idx[n]:idx[n + 1]])
    xd_traj_tensor = torch.from_numpy(xd_train[idx[n]:idx[n + 1]])
    s0_list.append(x_traj_tensor[0].numpy())
    traj_dataset = torch.utils.data.TensorDataset(x_traj_tensor, xd_traj_tensor)
    expert_traj_list.append(x_traj_tensor)
    dataset_list.append(traj_dataset)
    t_final = dt * (x_traj_tensor.shape[0] - 1)
    t_final_list.append(t_final)
    t_eval = np.arange(0., t_final + dt, dt)
    time_list.append(t_eval)

n_experts = len(dataset_list)
x_train = dataset.x
x_train_tensor = torch.from_numpy(x_train)

xd_train = dataset.xd
xd_train_tensor = torch.from_numpy(xd_train)

if not minibatch_mode:
    batch_size = xd_train.shape[0]

#  ------------------------------------------
# finding the data range

xmin = np.min(x_train[:, 0])
xmax = np.max(x_train[:, 0])
ymin = np.min(x_train[:, 1])
ymax = np.max(x_train[:, 1])

x_lim = [[xmin - 0.1, xmax + 0.1], [ymin - 0.1, ymax + 0.1]]


# --------------------------------------------------------------------------------
# Learner setup

# bijection network
taskmap_net = BijectionNet(num_dims=n_dims, num_blocks=num_blocks, num_hidden=num_hidden, s_act=s_act, t_act=t_act,
                           sigma=sigma,
                           coupling_network_type=coupling_network_type)


y_pot_grad_fcn = lambda y: F.normalize(y)   # potential fcn gradient (can use quadratic potential instead)

# pulled back dynamics (natural gradient descent system)
euclideanization_net = NaturalGradientDescentVelNet(taskmap_fcn=taskmap_net,
                                                    grad_potential_fcn=y_pot_grad_fcn,
                                                    origin=torch.from_numpy(goal).float(),
                                                    scale_vel=True,
                                                    is_diffeomorphism=True,
                                                    n_dim_x=n_dims,
                                                    n_dim_y=n_dims,
                                                    eps=eps,
                                                    device=device)
learner_model = euclideanization_net

if not load_learner_model:
    print('Training model ...')
    # Training learner
    optimizer = optim.Adam(learner_model.parameters(), lr=learning_rate, weight_decay=weight_regularizer)
    criterion = nn.SmoothL1Loss()
    loss_fn = criterion

    dataset = TensorDataset(x_train_tensor, xd_train_tensor)
    learner_model.train()
    best_model, train_loss = \
        train(learner_model, loss_fn, optimizer, dataset, epochs, batch_size=batch_size, stop_threshold=stopping_thresh)

    print(
        'Training loss: {:.4f}'.
            format(train_loss))

    try:
        os.makedirs('models')
    except OSError:
        pass

    learner_model = best_model
    torch.save(learner_model.state_dict(), os.path.join('models', '{}.pt'.format(data_name)))

else:
    print('Loading model ...')
    # Loading learner
    learner_model.load_state_dict(torch.load(os.path.join('models', '{}.pt'.format(data_name))))

# ---------------------------------------------------------
# Plotting best results

if test_learner_model:
    print('Plotting rollouts and vector fields. This may take a few moments ...')
    learner_model.eval()
    learner_traj_list = []

    # rollout trajectories
    for n in range(n_experts):
        s0 = s0_list[n]
        t_final = t_final_list[n]
        learner_traj = generate_trajectories(learner_model, s0, order=1, return_label=False, t_step=dt, t_final=t_final,
                                             method='euler')

        learner_traj_list.append(learner_traj)

    # visualize vector field and potentials
    taskmap_net = learner_model.taskmap_fcn
    potential_fcn = lambda x: torch.norm(taskmap_net(x)[0] - taskmap_net(torch.from_numpy(goal).float())[0], dim=1)

    x1_test = np.arange(x_lim[0][0], x_lim[0][1], plot_resolution)
    x2_test = np.arange(x_lim[1][0], x_lim[1][1], plot_resolution)
    X1, X2 = np.meshgrid(x1_test, x2_test)
    x_test = np.concatenate((X1.flatten().reshape(-1, 1), X2.flatten().reshape(-1, 1)), 1)

    x_test_tensor = torch.from_numpy(x_test).float()
    z_test_tensor = potential_fcn(x_test_tensor)
    z_test = z_test_tensor.detach().cpu().numpy()

    max_z = np.max(z_test)
    min_z = np.min(z_test)
    z_test = (z_test - min_z) / (max_z - min_z)
    Z = z_test.reshape(X1.shape[0], X1.shape[1])

    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.set_xlim(x_lim[0])
    ax1.set_ylim(x_lim[1])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    ax1.imshow(Z, extent=[x_lim[0][0], x_lim[0][1], x_lim[1][0], x_lim[1][1]], origin='lower', cmap='viridis')
    # ax1.axis(aspect='image')
    visualize_vel(learner_model, x_lim=x_lim, delta=plot_resolution, cmap=None, color='#f2e68f')

    fig2 = plt.figure()
    ax2 = plt.gca()
    ax2.set_xlim(x_lim[0])
    ax2.set_ylim(x_lim[1])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    ax2.imshow(Z, extent=[x_lim[0][0], x_lim[0][1], x_lim[1][0], x_lim[1][1]], origin='lower', cmap='viridis')
    # ax2.axis(aspect='image')
    contours = plt.contour(X1, X2, Z, 25, cmap=None, colors='#f2e68f')

    expert_traj_list = [traj.numpy() for traj in expert_traj_list]
    learner_traj_list = [traj.numpy() for traj in learner_traj_list]

    for n in range(n_experts):
        expert_traj = expert_traj_list[n]
        learner_traj = learner_traj_list[n]

        ax1.plot(expert_traj[:, 0], expert_traj[:, 1], 'w', linewidth=4, linestyle=':')
        ax1.plot(learner_traj[:, 0], learner_traj[:, 1], 'r', linewidth=3)
        ax1.plot(expert_traj[-1, 0], expert_traj[-1, 1], 'xg', linewidth=10, markersize=12, markeredgecolor='black')

        ax2.plot(expert_traj[:, 0], expert_traj[:, 1], 'w', linewidth=4, linestyle=':')
        ax2.plot(learner_traj[:, 0], learner_traj[:, 1], 'r', linewidth=3)
        ax2.plot(expert_traj[-1, 0], expert_traj[-1, 1], 'xg', linewidth=10, markersize=12, markeredgecolor='black')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    fig1.savefig(os.path.join('plots', '{}_vector_field.pdf'.format(data_name)), dpi=300)
    fig2.savefig(os.path.join('plots', '{}_contour_plot.pdf'.format(data_name)), dpi=300)

    plt.show()


