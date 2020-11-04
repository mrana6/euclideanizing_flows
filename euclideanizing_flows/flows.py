import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import numpy as np


class NaturalGradientDescentVelNet(nn.Module):
	"""
	taskmap_fcn: map to a latent space
	grad_taskmap_fcn: jacobian of the map
	grad_potential_fcn: gradient of a potential fcn defined on the mapped space
	n_dim_x: observed (input) space dimensions
	n_dim_y: latent (output) space dimentions
	origin (optional): shifted origin of the input space (this is the goal usually)
	scale_vel (optional): if set to true, learns a scalar velocity multiplier
	is_diffeomorphism (optional): if set to True, use the inverse of the jacobian itself rather than pseudo-inverse
	"""
	def __init__(self, taskmap_fcn, grad_potential_fcn, n_dim_x, n_dim_y,
				 scale_vel=True, is_diffeomorphism=True,
				 origin=None, eps=1e-12, device='cpu'):

		super(NaturalGradientDescentVelNet, self).__init__()
		self.taskmap_fcn = taskmap_fcn
		self.grad_potential_fcn = grad_potential_fcn
		self.n_dim_x = n_dim_x
		self.n_dim_y = n_dim_y
		self.eps = eps
		self.device = device
		self.I = torch.eye(self.n_dim_x, self.n_dim_x, device=device).unsqueeze(0)
		self.is_diffeomorphism = is_diffeomorphism
		self.scale_vel = scale_vel

		# scaling network (only used when scale_vel param is True!)
		self.log_vel_scalar = FCNN(n_dim_x, 1, 100, act='leaky_relu')					 # a 2-hidden layer network
		self.vel_scalar = lambda x: torch.exp(self.log_vel_scalar(x)) + self.eps 		 # always positive scalar!

		if origin is None:
			self.origin = torch.zeros(1, self.n_dim_x, device=self.device)
		else:
			self.origin = origin.to(device)

		if self.is_diffeomorphism:
			assert (n_dim_x == n_dim_y), 'Input and Output dims need to be same for diffeomorphism!'

	def forward(self, x):
		if x.ndimension() == 1:
			flatten_output = True  # output the same dimensions as input
			x = x.view(1, -1)
		else:
			flatten_output = False

		origin_, _ = self.taskmap_fcn(self.origin)
		y_hat, J_hat = self.taskmap_fcn(x)
		y_hat = y_hat - origin_  			# Shifting the origin
		yd_hat = -self.grad_potential_fcn(y_hat)  		# negative gradient of potential

		if self.is_diffeomorphism:
			J_hat_inv = torch.inverse(J_hat)
		else:
			I = self.I.repeat(J_hat.shape[0], 1, 1)
			J_hat_T = J_hat.permute(0, 2, 1)
			J_hat_inv = torch.matmul(torch.inverse(torch.matmul(J_hat_T, J_hat) + 1e-12 * I), J_hat_T)

		xd_hat = torch.bmm(J_hat_inv, yd_hat.unsqueeze(2)).squeeze() 	# natural gradient descent

		if self.scale_vel:
			xd = self.vel_scalar(x) * xd_hat  							# mutiplying with a learned velocity scalar
		else:
			xd = xd_hat

		if flatten_output:
			xd = xd.squeeze()

		return xd


class BijectionNet(nn.Sequential):
	"""
	A sequential container of flows based on coupling layers.
	"""
	def __init__(self, num_dims, num_blocks, num_hidden, s_act=None, t_act=None, sigma=None,
				 coupling_network_type='fcnn'):
		self.num_dims = num_dims
		modules = []
		print('Using the {} for coupling layer'.format(coupling_network_type))
		mask = torch.arange(0, num_dims) % 2  # alternating inputs
		mask = mask.float()
		# mask = mask.to(device).float()
		for _ in range(num_blocks):
			modules += [
				CouplingLayer(
					num_inputs=num_dims, num_hidden=num_hidden, mask=mask,
					s_act=s_act, t_act=t_act, sigma=sigma, base_network=coupling_network_type),
			]
			mask = 1 - mask  # flipping mask
		super(BijectionNet, self).__init__(*modules)

	def jacobian(self, inputs, mode='direct'):
		'''
		Finds the product of all jacobians
		'''
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		return J

	def forward(self, inputs, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		return inputs, J


class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, num_hidden, mask,
				 base_network='rffn', s_act='elu', t_act='elu', sigma=0.45):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.mask = mask

		if base_network == 'fcnn':
			self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
			self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)
			print('Using neural network initialized with identity map!')

			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.translate_net.network[-1].bias.data)

			nn.init.zeros_(self.scale_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].bias.data)

		elif base_network == 'rffn':
			print('Using random fouier feature with bandwidth = {}.'.format(sigma))
			self.scale_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)
			self.translate_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)

			print('Initializing coupling layers as identity!')
			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].weight.data)
		else:
			raise TypeError('The network type has not been defined')

	def forward(self, inputs, mode='direct'):
		mask = self.mask
		masked_inputs = inputs * mask
		# masked_inputs.requires_grad_(True)

		log_s = self.scale_net(masked_inputs) * (1 - mask)
		t = self.translate_net(masked_inputs) * (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return inputs * s + t
		else:
			s = torch.exp(-log_s)
			return (inputs - t) * s

	def jacobian(self, inputs):
		return get_jacobian(self, inputs, inputs.size(-1))


class RFFN(nn.Module):
	"""
	Random Fourier features network.
	"""

	def __init__(self, in_dim, out_dim, nfeat, sigma=10.):
		super(RFFN, self).__init__()
		self.sigma = np.ones(in_dim) * sigma
		self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.network = nn.Sequential(
			LinearClamped(in_dim, nfeat, self.coeff, self.offset),
			Cos(),
			nn.Linear(nfeat, out_dim, bias=False)
		)

	def forward(self, x):
		return self.network(x)


class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

	def forward(self, x):
		return self.network(x)


class LinearClamped(nn.Module):
	'''
	Linear layer with user-specified parameters (not to be learrned!)
	'''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values, bias=True):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight, self.bias)
		return F.linear(input, self.weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class Cos(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return torch.cos(inputs)


def get_jacobian(net, x, output_dims, reshape_flag=True):
	"""

	"""
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	x_m.requires_grad_(True)
	y_m = net(x_m)
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J

