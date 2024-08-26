import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torchvision.transforms as transforms
import numpy as np
import sys
from PIL import Image

import matplotlib
matplotlib.use('Agg')


def normalize_tensor(data, target_min=0, target_max=1):
	data_reshape = data.view(data.shape[0],-1)
	data_min     = torch.min(data_reshape, dim=1).values.unsqueeze(1)
	data_max     = torch.max(data_reshape, dim=1).values.unsqueeze(1)
	data_reshape = (data_reshape - data_min) / (data_max - data_min)
	data_reshape = (target_max - target_min) * data_reshape + target_min 
	return data_reshape.view(data.shape)

'''
def normalize_tensor(data, target_min=0, target_max=1):
	data_reshape = data.flatten()
	data_min     = torch.min(data_reshape)
	data_max     = torch.max(data_reshape)
	data_reshape = (data_reshape - data_min) / (data_max - data_min)
	data_reshape = (target_max - target_min) * data_reshape + target_min 
	return data_reshape.view(data.shape)
'''


def normalize_tensor_2d(data, target_min=0, target_max=1):
	print('warning: normalize_tensor_2d() was integraed to normalize_tensor().')
	return normalize_tensor(data, target_min, target_max)


#
# Example:
#
# python3 visualize.py
#
def plot_curve_errorbar(X, Y, errs, labels, colors=['red','blue','green','y','magenta'], x_interval=1,xlabel=None, ylabel=None, ylim=None, yscale='linear', legend=True, alpha=0.2, grid=False, normalize_each=False, filename=None):
	plt.cla()
	fig,ax = plt.subplots()

	# When plot data are given as list OR numpy array with more than 2 data
	if isinstance(Y,list) or (isinstance(Y, np.ndarray) and len(Y.shape)>1): 
		for i in range(len(Y)):
			y     = Y[i]
			if normalize_each: 
				miny = y.min()
				maxy = y.max()
				y = (y - miny) / (maxy - miny)
			if X is None: x = x_interval * np.arange(len(y))
			else:         x = X[i]
			label = labels[i]
			if colors is not None: 
				color = colors[i]
				ax.plot(x, y, label=label, color=color)
			else:
				ax.plot(x, y, label=label)
				

			if errs is not None: 
				err = errs[i]
				ax.fill_between(x, y-err, y+err, alpha=alpha, color=color)
	else: 
			y     = Y
			if normalize_each: 
				miny = y.min()
				maxy = y.max()
				y = (y - miny) / (maxy - miny)
			if X is None: x = x_interval * np.arange(len(y))
			else:         x = X
			label = labels
			if colors is not None: 
				color = colors[0]
				ax.plot(x, y, label=label, color=color)
			else:
				ax.plot(x, y, label=label)

			if errs is not None: 
				err = errs
				ax.fill_between(x, y-err, y+err, alpha=alpha, color=color)

	if xlabel is not None:   ax.set_xlabel( xlabel )
	if ylabel is not None:   ax.set_ylabel( ylabel )
	if ylim is not None:     ax.set_ylim( ylim[0],ylim[1] )
	ax.set_yscale( yscale )
	if grid:                 ax.grid()
	if legend:               ax.legend()
	if filename is not None: plt.savefig(filename, bbox_inches='tight')
	plt.gcf().clear() 
	plt.close()


def plot_curve_errorbar_twinaxis(
values,  values2,
errs,    errs2,
labels,  labels2, 
colors=['blue','green','magenta'], colors2=['red','y','black'], x_interval=1,xlabel=None, ylabel=None, ylim=None, yscale='linear', legend=True, alpha=0.2, grid=False, normalize_each=False, filename=None):
	plt.cla()
	fig,ax = plt.subplots()

	# When plot data are given as list OR numpy array with more than 2 data
	if isinstance(values,list) or (isinstance(values, np.ndarray) and len(values.shape)>1): 
		for i in range(len(values)):
			y     = values[i]
			if normalize_each: 
				miny = y.min()
				maxy = y.max()
				y = (y - miny) / (maxy - miny)
			x     = x_interval * np.arange(len(y))
			label = labels[i]
			color = colors[i]
			ax.plot(x, y, label=label, color=color)

			if errs is not None: 
				err = errs[i]
				ax.fill_between(x, y-err, y+err, alpha=alpha, color=color)
	else: 
			y     = values
			if normalize_each: 
				miny = y.min()
				maxy = y.max()
				y = (y - miny) / (maxy - miny)
			x     = x_interval * np.arange(len(y))
			label = labels
			color = colors[0]
			ax.plot(x, y, label=label, color=color)

			if errs is not None: 
				err = errs
				ax.fill_between(x, y-err, y+err, alpha=alpha, color=color)

	ax2 = ax.twinx()
	# When plot data are given as list OR numpy array with more than 2 data
	if isinstance(values2,list) or (isinstance(values2, np.ndarray) and len(values2.shape)>1): 
		for i in range(len(values2)):
			y     = values2[i]
			if normalize_each: 
				miny = y.min()
				maxy = y.max()
				y = (y - miny) / (maxy - miny)
			x     = x_interval * np.arange(len(y))
			label = labels2[i]
			color = colors2[i]
			ax2.plot(x, y, label=label, color=color)

			if errs is not None: 
				err = errs[i]
				ax2.fill_between(x, y-err, y+err, alpha=alpha, color=color)
	else: 
			y     = values2
			if normalize_each: 
				miny = y.min()
				maxy = y.max()
				y = (y - miny) / (maxy - miny)
			x     = x_interval * np.arange(len(y))
			label = labels2
			color = colors2[0]
			ax2.plot(x, y, label=label, color=color)

			if errs is not None: 
				err = errs
				ax2.fill_between(x, y-err, y+err, alpha=alpha, color=color)	

	if xlabel is not None:   ax.set_xlabel( xlabel )
	if ylabel is not None:   ax.set_ylabel( ylabel )
	if ylim is not None:     ax.set_ylim( ylim[0],ylim[1] )
	ax.set_yscale( yscale )
	if grid:                 ax.grid()
	if legend:               ax.legend(loc='upper left')
	if legend:               ax2.legend(loc='upper right')
	if filename is not None: plt.savefig(filename, bbox_inches='tight')
	plt.gcf().clear() 
	plt.close()



def plot_image_grid_now(handle_figure, image_tensor, filename=None, normalize=True):

	# f = plt.figure()
	plt.figure(handle_figure.number)

	num_image = image_tensor.shape[0]
	nrows	  = np.sqrt(num_image)
	nrows	  = int(np.ceil(nrows))

	# image_data = torch.clamp(image_tensor, 0, 255) 
	# image_data = torch.view(image_tensor.size(0),-1)
	
	if normalize:  image_data = normalize_tensor(image_tensor, 0, 1)
	
	image_data = image_data.detach().cpu()
	image_grid = make_grid(image_data, nrow=nrows)
	image_grid = image_grid.permute(1, 2, 0)

	# plt.tight_layout()
	plt.imshow(image_grid, vmin=0, vmax=1)

	'''
	if image_tensor.shape[1] == 1:

		mappable = plt.cm.ScalarMappable(cmap='gray')
	
	elif image_tensor.shape[1] == 3:

		mappable = plt.cm.ScalarMappable(cmap='hsv')

	else:

		pass

	mappable.set_array([0,1])   
	plt.colorbar(mappable)
	''' 
	
	# plt.show()
	plt.pause(0.001)

	if filename is not None:
		handle_figure.savefig(filename)
		pass



def resize_examples(examples, size=64):
	transform = transforms.Compose([
	    transforms.ToPILImage(),
	    transforms.Resize(size=size,interpolation=transforms.InterpolationMode.NEAREST),
	    transforms.ToTensor()
	])
	return [transform(x) for x in examples]


def draw_image_grid(image_data, filename=None, nrows=None, upsize=True, normalize=None, scale_each=None, pad_value=None):  # pad_value: color of grid lines, 0 black, 1 white.

	image_data = image_data.detach().cpu()

	num_image, channel, height, width  = image_data.shape

	if normalize is None:
		if channel == 1: normalize = True
		else:            normalize = True   # color images

	if scale_each is None:
		if channel == 1: scale_each = False
		else:            scale_each = True   # color images

	if pad_value is None:
		if channel == 1: pad_value = 1  # white grid line
		else:            pad_value = 0  # black grid line

	if nrows is None or nrows * nrows > num_image:
		nrows = int(np.ceil( np.sqrt(num_image) ))  

	if normalize:  image_data = normalize_tensor(image_data, 0, 1)

	#print(image_data.shape)

	if upsize and width <= 64:
		image_data = resize_examples(image_data, size=64)
	

	image_grid = make_grid(image_data, nrow=nrows, pad_value=pad_value, scale_each=scale_each)
	# https://pytorch.org/vision/stable/_modules/torchvision/utils.html#save_image
	ndarr = image_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
	im = Image.fromarray(ndarr)
	im.save(filename)




if __name__=='__main__':
	
	#
	# Demonstration of plot_curve_errobar()
	#
	x  = np.arange(10)
	y1 = x**2
	e1 = np.random.randn(*y1.shape)
	y2 = 2*x + 1
	e2 = np.random.randn(*y2.shape)

	plot_curve_errorbar(None, y1, e1, labels='real', ylim=[0,100], filename='test_plot_curve_errorbar1.png')
	plot_curve_errorbar(X=[None,None], Y=[y1,y2], errs=[e1,e2], labels=['real','fake'], ylim=[0,100], filename='test_plot_curve_errorbar2.png')


	#
	# Demonstration of plot_batch()
	#
	batch = torch.rand((64,3,32,32)) # number of images, channel, height, and width of image

	# we plot 4 x 4 images
	#plot_batch(batch=batch, filename='./test_plot_batch.png')
	draw_image_grid(image_data=batch, filename='./test_plot_batch.png')


