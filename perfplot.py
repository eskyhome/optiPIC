import numpy as np
import matplotlib.pyplot as mlt


def readData(file_name):
	data = np.loadtxt('results\timing_'+file_name+'.xml', delimiter=',')
	return data;

def pplot(testname, measure, param1, param2):
	#testname = 'iterations1'
	#measure = 'Runtime'
	#param1 = 'N_iterations'
	#param2 = 'N_x'
	data = readData(testname)
	
	fig = plt.figure()

	axes = fig.plot(data)

	axes.set_title(testname)
	axes.set_xlabel(param1)
	axes.set_ylabel(measure)
	#axes.set_zlabel(param2)

	plt.savefig(testname + '.png', bbox_inches='tight')