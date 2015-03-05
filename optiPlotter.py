import configparser as cfgp
import csv
import numpy as np
import matplotlib.pyplot as mplot
import mpl_toolkits.mplot3d.axes3d as p3
import maplotlib.animation as animation

def readData(cfg):
	dataSet = np.empty(3, cfg['DEFAULT']['particles'], cfg['DEFAULT']['iterations'])
	with open('particle_trace.csv', newline='') as csvfile:
		tracereader = csv.reader(csvfile)
		p = 0
		for particle in tracereader:
			i = 0
			for iteration in particle.split(';'):
				vals = iteration.split(',')
				dataSet[0][i][p] = vals[0]
				dataSet[1][i][p] = vals[1]
				dataSet[2][i][p] = vals[2]
				p += 1
				i += 1
	return dataSet

def animate(figure, cfg, dataSet, updateFunc):
	fig = mplot.figure()
	axes = p3.Axes3D(fig)
	plot = [axes.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1]) for data in dataSet]
	
	axes.set_xlim3d([0.0, cfg['DEFAULT']['lx']])
	axes.set_xlabel('X')

	axes.set_ylim3d([0.0, cfg['DEFAULT']['ly']])
	axes.set_ylabel('Y')

	axes.set_zlim3d([0.0, cfg['DEFAULT']['lz']])
	axes.set_zlabel('Z')

	axes.set_title('optiPic')

	picAnimation = animation.FuncAnimation(figure, updateFunc, frames, fargs=(dataSet, plot), interval=25, blit=False)
	mplot.show()
	
def update(num, dataSet, plot):
	for line, data in zip(plot, dataSet):
		line.set_data(data[0:2, num-20:num])
		line.set_3d_properties(data[2, num-20:num])
	return plot

def main():	
	cfg = cfgp.ConfigParser()
	cfg.read('settings.cfg')
	dataSet = readData(cfg)
	animate(fig, cfg, dataSet, update)
main()
