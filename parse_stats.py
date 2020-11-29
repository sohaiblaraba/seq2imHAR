import os
import csv
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def parse_stats(stats_path):
	reader = csv.DictReader(open(stats_path, 'r'))
	dict_list = []
	
	for line in reader:
		dict_list.append(line)
		
	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	my_colors = list(colors.keys())
	my_colors.remove('w')
	my_colors.remove('aliceblue')
	my_colors.remove('antiquewhite')
	my_colors = my_colors[:len(dict_list)]
	
	plt.ion()
	fig, ax = plt.subplots(3)
	
	c = 0
	for d in dict_list:
		ax[0].plot(eval(d['training_losses']), color=my_colors[c], label=d['name'])
		ax[1].plot(eval(d['training_accs']), color=my_colors[c], label=d['name'])
		c += 1
	
	x = range(len(dict_list))
	accs = []
	names = []
	for d in dict_list:
		accs.append(float(d['accuracy']))
		names.append(d['name'])
	
	ax[2].bar(x, accs, color=my_colors)
	plt.xticks(x, names, rotation='vertical')
	for i, v in enumerate(accs):
		ax[2].text(i, v, "{:.3f}".format(v), color='black')
	
	plt.show(block=True)


if __name__ == "__main__":
	stats_path = "1561543166_NTU120_OS_retrain_deep_stats.csv"
	parse_stats(stats_path)
