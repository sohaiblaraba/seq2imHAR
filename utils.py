from matplotlib import colors as mcolors
import smtplib


def get_palet(number=5):
	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	my_colors = list(colors.keys())
	my_colors.remove('w')
	my_colors.remove('aliceblue')
	my_colors.remove('antiquewhite')
	my_colors = my_colors[:number]
	plot_colors = {}
	for i in range(number):
		plot_colors[number] = my_colors[i]
	
	return plot_colors


def get_model_urls():
	return {
		'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
		'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
		#'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
		'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
		'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
		'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
		#  truncated _google to match module name
		'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
		'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
		'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
		'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
		'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
		'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
		'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
		'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
		'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
		'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
		'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
		'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
		'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
		'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
		'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
		'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
	}


def get_model_sizes():
	return {
		'alexnet': (224, 224),
		'densenet': (224, 224),
		'resnet': (224, 224),
		'inception': (299, 299),
		'squeezenet': (224, 224),  # not 255,255 acc. to https://github.com/pytorch/pytorch/issues/1120
		'vgg': (224, 224)
	}