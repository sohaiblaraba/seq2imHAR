from matplotlib import colors as mcolors

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


def get_model_list():
	return {
		'alexnet': {'url': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
					'size': (224, 224)},
		'densenet121': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth',
			 'size': (224, 224)},
		'densenet169': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth',
			 'size': (224, 224)},
		'densenet201': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth',
			 'size': (224, 224)},
		'densenet161': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth',
			 'size': (224, 224)},
		'inception_v3': {'url': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
			 'size': (299, 299)},
		'resnet18': {'url': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
			 'size': (224, 224)},
		'resnet34': {'url': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
			 'size': (224, 224)},
		'resnet50': {'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
			 'size': (224, 224)},
		'resnet101': {'url': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
			 'size': (224, 224)},
		'resnet152': {'url': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
			 'size': (224, 224)},
		'squeezenet1_0': {'url': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
			 'size': (224, 224)},
		'squeezenet1_1': {'url': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
			 'size': (224, 224)},
		'vgg11': {'url': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
			 'size': (224, 224)},
		'vgg13': {'url': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
			 'size': (224, 224)},
		'vgg16': {'url': 'https://download.pytorch.org/models/vgg16-397923af.pth',
			 'size': (224, 224)},
		'vgg19': {'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
			 'size': (224, 224)},
		'vgg11_bn': {'url': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
			 'size': (224, 224)},
		'vgg13_bn': {'url': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
			 'size': (224, 224)},
		'vgg16_bn': {'url': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
			 'size': (224, 224)},
		'vgg19_bn': {'url': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
			 'size': (224, 224)}
	}