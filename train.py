import os
import glob
import time
import argparse
import matplotlib.pyplot as plt
import csv

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets

from itertools import accumulate
from functools import reduce

from parse_stats import *
from utils import get_palet, get_model_list


# Generic pretrained model loading
# We solve the dimensionality mismatch between
# final layers in the constructed vs pretrained
# modules at the data level.
def diff_states(dict_canonical, dict_subset):
	# Sanity check that param names overlap
	# Note that params are not necessarily in the same order
	# for every pretrained model
	names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
	not_in_1 = [n for n in names1 if n not in names2]
	not_in_2 = [n for n in names2 if n not in names1]
	assert len(not_in_1) == 0
	assert len(not_in_2) == 0

	for name, v1 in dict_canonical.items():
		v2 = dict_subset[name]
		assert hasattr(v2, 'size')
		if v1.size() != v2.size():
			yield (name, v1)


def load_defined_model(name, num_classes):
	model = models.__dict__[name](num_classes=num_classes)

	# Densenets don't (yet) pass on num_classes, hack it in for 169
	if name == 'densenet169':
		model = torchvision.models.DenseNet(num_init_features=64,
											growth_rate=32,
											block_config=(6, 12, 32, 32),
											num_classes=num_classes)

	pretrained_state = model_zoo.load_url(list_of_models[name]['url'])

	# Diff
	diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
	print("Replacing the following state from initialized ", name, ":", [d[0] for d in diff])

	for name, value in diff:
		pretrained_state[name] = value

	assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0

	# Merge
	model.load_state_dict(pretrained_state)
	return model, diff


def get_model(name, num_classes, mode='retrain_deep'):
	assert mode in ['scratch', 'retrain_shallow', 'retrain_deep']

	model = None
	params = None
	is_retrained = False
	is_retrained_shallow = False

	if mode == 'scratch':
		model = models.__dict__[name](num_classes=num_classes)
	else:
		model, diff_ = load_defined_model(name, num_classes)

		if mode == 'retrain_deep':
			is_retrained = True
		elif mode == 'retrain_shallow':
			params = [d[0] for d in diff_]
			is_retrained = True
			is_retrained_shallow = True

	if use_gpu:
		print("Transferring model to GPU(s)...")
		model = torch.nn.DataParallel(model).cuda()
	return model, params, is_retrained, is_retrained_shallow


def filtered_params(net, param_list=None):
	def in_param_list(s):
		for p in param_list:
			if s.endswith(p):
				return True
		return False
	# Caution: DataParallel prefixes '.module' to every parameter name
	params = net.named_parameters() if param_list is None \
		else (p for p in net.named_parameters() if in_param_list(p[0]))

	return params


def load_data(data_path, resize):

	data_transforms = {
		'train': transforms.Compose([
			transforms.Resize(resize),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			# Higher scale-up for inception
			transforms.Resize(resize),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}

	dsets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
				for x in ['train', 'val']}

	dsets_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True)
					 for x in ['train', 'val']}

	return dsets_loaders['train'], dsets_loaders['val']


def train(net, trainloader, valloader, network_name, epochs, param_list=None, plot_loss=False):

	def in_param_list(s):
		for p in param_list:
			if s.endswith(p):
				return True
		return False

	criterion = nn.CrossEntropyLoss()
	if use_gpu:
		criterion.cuda()

	params = (p for p in filtered_params(net, param_list))

	# if finetuning model, turn off grad for other params
	if param_list:
		for p_fixed in (p for p in net.named_parameters() if not in_param_list(p[0])):
			p_fixed[1].requires_grad = False

	# Optimizer (from paper)
	optimizer = optim.SGD((p[1] for p in params), lr=0.001, momentum=0.9)

	losses = []
	accs = []
	if plot_loss:
		plt.ion()

	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			x, y = data
			if use_gpu:
				x, y = Variable(x.cuda()), Variable(y.cuda(non_blocking=True))
			else:
				x, y = Variable(x), Variable(y)

			# Zero the parameter gradients
			optimizer.zero_grad()

			# Forward + Backward + Optimize
			outputs = net(x)

			# For nets that have multiple outputs (like inception)
			if isinstance(outputs, tuple):
				loss = sum((criterion(o, y) for o in outputs))
			else:
				loss = criterion(outputs, y)

			loss.backward()
			optimizer.step()

			# Print statistics
			# running_loss += loss.data[0]
			running_loss += loss.item()

			if i % 29 == 0:
				losses.append(loss.item())
				print('[%s, %d, %5d] loss: %.3f' % (network_name, epoch + 1, i + 1, loss.item()))
				running_loss = 0.0
				if plot_loss:
					plt.plot(epoch * len(trainloader) + i + 1, loss.item(), plot_colors[name])
					plt.pause(0.05)
		acc_epo = evaluate_test(net, valloader)
		accs.append(acc_epo)
		print("Epc = %d - Acc = %f" % (epoch+1, acc_epo))
	if not os.path.exists(my_path + '/../models'):
		os.mkdir(my_path + '/../models')

	# save_checkpoint(net, my_path + "/../models/" + start_time + "_" + network_name)
	save_checkpoint(net, args.stats + "/" + start_time + "_" + network_name)
	print("Finished training!")
	return losses, accs


# Get stats for training and evaluation in a structured way
# if param_list is None, all relevant parameters are tuned,
# otherwise, only parameters that have been constructed for custom
# num_classes
def train_stats(net, trainloader, valloader, network_name, epochs, param_list=None):
	stats = {}
	params = filtered_params(net, param_list)
	counts = 0, 0
	for counts in enumerate(accumulate((reduce(lambda d1, d2: d1 * d2, p[1].size()) for p in params))):
		pass
	stats['variables_optimized'] = counts[0] + 1
	stats['params_optimized'] = counts[1]

	tic = time.time()
	losses, accs = train(net, trainloader, valloader, network_name, epochs=epochs, param_list=param_list)
	stats['training_time'] = time.time() - tic
	stats['training_loss'] = losses[-1] if len(losses) else float('nan')
	stats['training_losses'] = losses
	stats['training_accs'] = accs

	return stats


def train_eval(net, trainloader, valloader, network_name, epochs=30, param_list=None):
	print("Training..." if not param_list else "Retraining...")
	stats_train = train_stats(net, trainloader, valloader, network_name, epochs=epochs, param_list=param_list)

	print("Evaluating %s" % network_name)
	net = net.eval()
	eval_stats = evaluate_stats(net, valloader)

	return {**stats_train, **eval_stats}


def save_checkpoint(state, filename='model.pth.tar'):
	torch.save(state, filename)


def evaluate_test(net, valloader):
	correct = 0
	total = 0

	for i, data in enumerate(valloader, 0):
		x, y = data

		if use_gpu:
			x, y = (x.cuda()), (y.cuda(non_blocking=True))

		outputs = net(Variable(x))
		_, predicted = torch.max(outputs.data, 1)
		total += y.size(0)
		correct += (predicted == y).sum()
	accuracy = correct / total
	return accuracy


def evaluate_stats(net, testloader):
	global accfinal
	stats = {}
	correct = 0
	total = 0

	tic = time.time()
	for i, data in enumerate(testloader, 0):
		x, y = data

		if use_gpu:
			x, y = (x.cuda()), (y.cuda(non_blocking=True))

		outputs = net(Variable(x))
		_, predicted = torch.max(outputs.data, 1)
		total += y.size(0)
		correct += (predicted == y).sum()
		print(correct)
	accuracy = correct / total
	stats['accuracy'] = accuracy
	stats['eval_time'] = time.time() - tic
	accfinal = accuracy
	print('Accuracy on test images: %f' % accuracy)
	return stats


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', help='path to data folder', type=str, required=True)
	parser.add_argument('-m', '--mode', help='mode (scratch, retrain_shallow, retrain_deep)', type=str, default='retrain_deep', required=False)
	parser.add_argument('-e', '--epoch', help='number of epochs', type=int, required=True)
	parser.add_argument('-b', '--batch', help='batch size', type=int, required=True)
	parser.add_argument('-s', '--save', help='Path to save models and stats', default=None, required=True)
	args = parser.parse_args()

	data_path = args.data
	save_path = args.save
	train_mode = args.mode
	num_epochs = args.epoch
	batch_size = args.batch

	start_time        = str(int(time.time()))
	list_of_models    = get_model_list()
	models_to_test    = ['alexnet']

	use_gpu           = torch.cuda.is_available()
	plot_colors       = get_palet(len(models_to_test))
	accfinal          = 0

	stats_file = save_path + '/' + start_time + '_' + os.path.basename(data_path) + '_' + args.mode + '_stats.csv'

	number_classes = len(glob.glob(data_path+'/train/*'))

	stats = []

	print('Data:', data_path)
	print('Mode:', train_mode)
	print('Saving models to:', save_path)
	print('Saving stats to:', stats_file)

	for name in models_to_test:
		print('\nTargetting {} with {} classes'.format(name, number_classes))
		print('-------------------------------------------------------')

		model, params, is_retrained, is_shallow = get_model(name, number_classes, train_mode)

		resize = list_of_models[name]['size']

		trainloader, valloader = load_data(data_path, resize)

		model_stats = train_eval(model, trainloader, valloader, name, epochs=num_epochs, param_list=params)

		stats.append(model_stats)

		with open(stats_file, 'w') as csvfile:
			fieldnames = stats[0].keys()
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			writer.writeheader()
			for s in stats:
				writer.writerow(s)

	parse_stats(stats_path)

