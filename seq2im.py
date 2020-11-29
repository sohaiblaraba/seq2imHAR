import numpy as np
import os
from PIL import Image
import cv2
from copy import deepcopy
import sys
import glob

import json
import shutil

import matplotlib.pyplot as plt


class Seq2Im:
	def __init__(self):
		self.folder     = None
		self.file       = None
		self.data       = None
		self.nofjoint   = 21
		self.image      = None
		self.norm       = None
		self.rearrange  = None
		self.size       = (256, 256)
		self.is_mocap     = False

	def normalize(self):
		# nof*noj*3
		for i in range(self.data.shape[1]):
			self.data[:, i, 0] -= self.data[:, self.norm, 0]
			self.data[:, i, 1] -= self.data[:, self.norm, 1]
			self.data[:, i, 2] -= self.data[:, self.norm, 2]
		# remove the id of the noramlized joint
		np.delete(self.data, self.norm, 1)

	def generate_channels(self):
		min_x = np.min(self.data[:, :, 0])
		min_y = np.min(self.data[:, :, 1])
		min_z = np.min(self.data[:, :, 2])

		max_x = np.max(self.data[:, :, 0])
		max_y = np.max(self.data[:, :, 1])
		max_z = np.max(self.data[:, :, 2])

		self.data[:, :, 0] = (self.data[:, :, 0] - min_x) / (max_x - min_x)
		self.data[:, :, 1] = (self.data[:, :, 1] - min_y) / (max_y - min_y)
		self.data[:, :, 2] = (self.data[:, :, 2] - min_z) / (max_z - min_z)


	def seq2im(self):

		if self.norm is not None and type(self.norm) is int:
			self.normalize()

		self.generate_channels()

		rgb = np.zeros(self.data.shape, 'uint8')
		rgb[..., 0] = self.data[:, :, 0]*255
		rgb[..., 1] = self.data[:, :, 1]*255
		rgb[..., 2] = self.data[:, :, 2]*255

		self.image = Image.fromarray(rgb)
		self.image = self.image.resize(self.size)
		self.image = self.image.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)


	def parse(self):
		name, extension = os.path.splitext(self.file)
		self.is_mocap = True

		if extension == ".txt":
			data = np.loadtxt(self.file, usecols=range(self.nofjoint*3))
			if data.shape[0] > 0 and data.shape[1] > 0:
				x = data[:, ::3]
				y = data[:, 1::3]
				z = data[:, 2::3]
				self.data = np.dstack((x, y, z))
			else:
				self.data = np.array((0, 0, 0))

		elif extension == ".v3d":
			framerate = 179   # to be changed into a global varialble (to insert as inputs if needed)
			with open(self.file, encoding='iso-8859-1') as f:
				data = np.loadtxt(f, skiprows=5)
			timestamp = data[:, 0]/framerate
			data = data[:, 1:]
			x = data[:, ::6]
			y = data[:, 1::6]
			z = data[:, 2::6]
			self.data = np.dstack((x, y, z))

		elif extension == ".skeleton":  # NTU RGB+D
			with open(self.file) as f:
				content = f.readlines()

			beg = 1
			end = 28
			data = np.zeros((25, 1))
			i = 0
			while end <= len(content):
				frame_meta = content[beg:beg+2]
				if int(frame_meta[0]) == 0:
					beg = beg + 1
					end = end + 1
				else:
					frame = np.array([np.array(content[i].split()).astype(np.float)[0:3]*1000 for i in range(beg+3, beg+28)])
					data = np.hstack((data, frame))
					beg = beg + 27 * int(frame_meta[0]) + 1
					end = end + 27 * int(frame_meta[0]) + 1

			data = np.delete(data, 0, 1)

			x = np.transpose(data[:, ::3])
			y = np.transpose(data[:, 1::3])
			z = np.transpose(data[:, 2::3])
			self.data = np.dstack((x, y, z))

		elif extension == ".pku":
			data = np.loadtxt(self.file)
			data = data[:, :75]
			x = data[:, ::3]
			y = data[:, 1::3]
			z = data[:, 2::3]
			self.data = np.dstack((x, y, z))

		else:
			self.is_mocap = False

	def rearrange(self):
		data_out = np.zeros([self.data.shape[0], len(self.rearrange), self.data.shape[2]])
		for i in range(len(self.rearrange)):
			data_out[:, i, :] = self.data[:, self.rearrange[i], :]
		return data_out

	def run(self, file=None, folder=None, output=None, nojoints=21, norm=None, rearrange=None, size=(256, 256)):
		self.file = file
		self.folder = folder
		self.size = size
		self.norm = norm
		self.rearrange = rearrange

		if file is not None:
			self.parse()
			if self.is_mocap:
				if self.rearrange:
					self.data = self.rearrange()
				self.seq2im()

				if output is not None:
					plt.imsave(output, np.array(self.image))

		elif self.folder is not None:
			for file in glob.glob(self.folder + '/*'):
				self.file = file
				self.parse()
				if self.is_mocap:
					if self.rearrange:
						self.data = self.rearrange()
					self.seq2im()

					if output is not None:
						if not os.path.exists(output):
							os.mkdir(output)
						name, extension = os.path.splitext(file)
						file_out = name+'.jpg'
						plt.imsave(file_out, np.array(self.image))

if __name__ == "__main__":
	seq2im = Seq2Im()

	# File
	file = "data/examples/S001C001P001R001A001.txt"
	output = "data/examples/S001C001P001R001A001.jpg"
	seq2im.run(file=file, norm=4, output=output)
	plt.imshow(seq2im.image)
	plt.show()

	# Folder
	# folder = "data"
	# output = "data"	
	# seq2im.run(folder=folder, norm=4, output=output)

	

