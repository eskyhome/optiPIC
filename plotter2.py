#Based on http://matplotlib.org/examples/animation/simple_3danim.html

import numpy
import matplotlib.pyplot as mplot
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import xml.etree.ElementTree as ET

fig = mplot.figure()
axes = p3.Axes3D(fig)

plotType = 'line'
#plotType = 'scatter'


def readData():
	#read data
	xml_data = ET.parse('particles.xml')
	root = xml_data.getroot()
	particle_interval = int(root.get('particle_interval'))
	n_iterations = int(int(root.get('n_iterations'))/(particle_interval)) + 1
	n_particles = int(root.get('n_particles'))
	its = root.findall('iteration')
	
	if plotType == 'line':
		data = [numpy.empty((3, n_iterations)) for index in range(n_particles)]
	elif plotType == 'scatter':
		data = [numpy.empty((3, n_particles)) for index in range(n_iterations)]
	
	if plotType == 'line':
		for p in range(n_particles):
			for i in range(n_iterations):
				data[p][0, i] = float(its[i][p].find('position').get('x'))
				data[p][1, i] = float(its[i][p].find('position').get('y'))
				data[p][2, i] = float(its[i][p].find('position').get('z'))
	elif plotType == 'scatter':
		for i in range(n_iterations):
			for p in range(n_particles):
				data[i][0, p] = float(its[i][p].find('position').get('x'))
				data[i][1, p] = float(its[i][p].find('position').get('y'))
				data[i][2, p] = float(its[i][p].find('position').get('z'))
	return data, n_iterations
	
def update(num, dataSet, plot):
	for line, data in zip(plot, dataSet):
		line.set_data(data[0:2, num-20:num])
		line.set_3d_properties(data[2, num-20:num])
	return plot
	
def updateScatter(num, dataSet, plot):
	plot.set_3d_properties = dataSet[num][2, :]
	return plot
	
def main():
	data, frames = readData()
	#print(numpy.dtype(data))
	#if plotType=='line':
	plot = [axes.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
	#elif plotType=='scatter':
	#	d = data[0]
	#	plot = axes.scatter(d[0, :], d[1, :], d[2, :])
		
	axes.set_xlim3d([0.0, 0.2])
	axes.set_xlabel('X')

	axes.set_ylim3d([0.0, 0.2])
	axes.set_ylabel('Y')

	axes.set_zlim3d([0.0, 0.2])
	axes.set_zlabel('Z')

	axes.set_title('PIC plot demo')

	pic_animation = animation.FuncAnimation(fig, update, frames, fargs=(data, plot), interval=25, blit=False)
	mplot.show()
	
main()