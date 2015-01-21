#Based on http://matplotlib.org/examples/animation/simple_3danim.html

import numpy
import matplotlib.pyplot as mplot
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import xml.etree.ElementTree as ET

def readData():
	#read data
	xml_data = ET.parse('particles.xml')
	root = xml_data.getroot()
	n_iterations = root.get('n_iterations')
	n_particles = root.get('n_particles')
	plt_data = [[0 for j in range(n_particles)] for i in range(n_iterations)]
	
	for i_plot, iteration in zip(plt_data, root.findall('iteration')):
		for p_plot, particle in zip(i_plot, iteration.findall('particle')):
			axes.plot(
				particle.find('position').get('x'),
				particle.find('position').get('y'),
				particle.find('position').get('z')
			)
	
	return plt_data, n_iterations
	
def update(num, data) :
    return data[num]
	
def main():
	fig = mplot.figure()
	axes = p3.Axes3D(fig)
	data, iterations = readData(axes)
	pic_plot = [axes.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]?

	axes.set_xlim3d([0.0, 0.2])
	axes.set_xlabel('X')

	axes.set_ylim3d([0.0, 0.2])
	axes.set_ylabel('Y')

	axes.set_zlim3d([0.0, 0.2])
	axes.set_zlabel('Z')

	axes.set_title('PIC plot demo')

	pic_animation = animation.FuncAnimation(fig, update, iterations, fargs=(data), interval=50, blit=False)
	plt.show()
	
main()