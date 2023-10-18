import numpy as np
from matplotlib.pyplot import *
from fuzzylab import *



def fuzzy_dynamic(inputdynamic, inputstd):
	fis = sugfis()
	fis.addInput([0, 3], Name = 'dynamic_range')
	fis.addMF('dynamic_range', 'trapmf',[0, 0, 1.7, 1.8], Name = 'Too_small')
	fis.addMF('dynamic_range', 'trapmf', [2, 2.1, 3, 3], Name = 'Good')
	fis.addMF('dynamic_range', 'gaussmf', [0.06, 1.9], Name = 'Little_small')
	#plotmf(fis,'input',0)

	fis.addInput([0, 100], Name = 'standard')
	fis.addMF('standard', 'trapmf',[0, 0, 25, 35], Name = 'Good')
	fis.addMF('standard', 'trapmf', [30, 40, 100, 100], Name = 'Too_big')

	fis.addOutput([-50, 100], Name = 'output1')
	fis.addMF('output1', 'trimf',[-10, 0, 10], Name = 'Zero')
	fis.addMF('output1', 'trimf',[20, 30, 40], Name = 'Add_little')
	fis.addMF('output1', 'trimf',[50, 60, 70], Name = 'Add_some')
	fis.addMF('output1', 'trimf',[-40, -30, -20], Name = 'Minus_little')
	#plotmf(fis, 'output', 0)
	ruleList = [[1, -1, 1, 1,1], [2 ,-1 ,3, 1,1], [0,1,3, 1,1], [-1 ,1, 3, 1,1], [0,0, 1, 1,1]]
	fis.addRule(ruleList)

	#std = inputstd

	#plotcontrol = evalfis(fis, [dL,dF,dR])
	pecontrol = evalfis(fis, [inputdynamic, inputstd])
	#print('output = ', plotcontrol)

	return pecontrol

def fuzzy_canny(inputline):
	fis = sugfis()
	fis.addInput([0, 35000], Name = 'Line_number')
	fis.addMF('Line_number', 'trapmf',[0, 0, 19000, 22000], Name = 'Too_small')
	fis.addMF('Line_number', 'gaussmf', [500, 24000], Name = 'Good')
	fis.addMF('Line_number', 'trapmf',[26000, 29000, 58000, 58000], Name = 'Too_big') #40000 40000
	fis.addMF('Line_number', 'gaussmf', [500, 22000], Name = 'A_little_small')
	fis.addMF('Line_number', 'gaussmf', [500, 26000], Name = 'A_little_big')
	#plotmf(fis,'input',0)

	fis.addOutput([-6, 6], Name = 'output1')
	fis.addMF('output1', 'trimf',[-2.5, -2, -1.5], Name = 'Minus_more')
	fis.addMF('output1', 'trimf',[-1.5, -1, -0.5], Name = 'Minus_some')
	fis.addMF('output1', 'trimf',[-1, -0.5, 0], Name = 'Minus_little')
	fis.addMF('output1', 'trimf',[-0.5, 0, 0.5], Name = 'Zero')
	fis.addMF('output1', 'trimf',[0, 0.5, 1], Name = 'Add_little')
	fis.addMF('output1', 'trimf',[0.5, 1, 1.5], Name = 'Add_some')
	fis.addMF('output1', 'trimf',[3.5, 4, 4.5], Name = 'Add_more') #1.5 2 2.5
	#plotmf(fis,'output',0)
	ruleList = [[0,1,1,1], [1,3,1,1], [2,6,1,1], [3,4,1,1], [4,2,1,1]]
	fis.addRule(ruleList)


	#plotcontrol = evalfis(fis, [dL,dF,dR])
	edgecontrol = evalfis(fis, [inputline])
	#print('output = ', plotcontrol)

	return edgecontrol

def fuzzy_linenumber(inputstd):
	fis = sugfis()
	fis.addInput([0, 2], Name = 'Line_std')
	#fis.addMF('Line_std', 'trapmf',[0, 0, 0.2, 0.3], Name = 'Good')
	fis.addMF('Line_std', 'trapmf',[0, 0, 0.5, 0.7], Name = 'Good')
	#fis.addMF('Line_std', 'trapmf', [0.5, 0.6, 2, 2], Name = 'Too_big')
	fis.addMF('Line_std', 'trapmf', [1.5, 1.6, 2, 2], Name = 'Too_big')
	#fis.addMF('Line_std', 'gaussmf', [0.05, 0.4], Name = 'A_little_big')
	fis.addMF('Line_std', 'gaussmf', [0.1, 0.9], Name = 'A_little_big')
	#plotmf(fis,'input',0)

	fis.addOutput([-2, 2], Name = 'output1')
	fis.addMF('output1', 'trimf',[-1.8, -1.5, -1.2], Name = 'Minus_some')
	fis.addMF('output1', 'trimf',[-1, -0.85, -0.7], Name = 'Minus_little')
	fis.addMF('output1', 'trimf',[-0.3, 0, 0.3], Name = 'Zero')
	fis.addMF('output1', 'trimf',[0.7, 0.85, 1], Name = 'Add_little')
	fis.addMF('output1', 'trimf',[1.2, 1.5, 1.8], Name = 'Add_some')
	#fis.addMF('output1', 'constant',-1.5, Name = 'Minus_some')
	#fis.addMF('output1', 'constant',-0.85, Name = 'Minus_little')
	#fis.addMF('output1', 'constant',0, Name = 'Zero')
	#fis.addMF('output1', 'constant',0.85, Name = 'Add_little')
	#fis.addMF('output1', 'constant',1.5, Name = 'Add_some')
	#plotmf(fis, 'output', 0)
	ruleList = [[0,3,1,1], [1,0,1,1], [2,1,1,1]]
	fis.addRule(ruleList)

	std = inputstd

	#plotcontrol = evalfis(fis, [dL,dF,dR])
	plotcontrol = evalfis(fis, [std])
	#print('output = ', plotcontrol)

	return plotcontrol

#print(fuzzy_dynamic(1,100))
#a = fuzzy_sobel(25000)
#print(a)