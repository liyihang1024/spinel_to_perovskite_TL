#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: Yihang LI
# @Email:  liyihang@shu.edu.cn
# @Create_Date:        2019-09-27 10:04:29
# @Last Modified by:   Yihang LI
# @Last Modified time: 2023-03-28 13:05:33


import os
import re
import sys
import pprint
import numpy as np
import pandas as pd
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.visualize import view
from ase.spacegroup import Spacegroup

print(__doc__)

def get_unique_sites(struc):

	conCell = read(struc)    # Read a cif structure
	sg = Spacegroup(227)     # Set the number of space groups to 227 [Fd-3m]
	coordinates  = conCell.get_scaled_positions()  # Fractional coordinates of atoms in structures
	cell_lengths = conCell.get_cell()[0][0]        # lattice constant
	# unique_sites = sg.unique_sites(coordinates)                # Ineffective sites: fractional coordinates
	unique_sites = sg.unique_sites(coordinates)*cell_lengths   # Ineffective sites: absolute coordinates
	# print(unique_sites)
	unique_sites = [np.around(i,8) for i in unique_sites]      # Retain 8 decimal places for each value of non equivalent coordinates
	unique_sites = [i.tolist() for i in unique_sites]          # Convert np.array to list
	unique_symbol_and_index = [{atom.symbol:atom.index} for atom in conCell if np.around(atom.position, 8).tolist() in unique_sites]   # Save non equivalent bit element symbols and their indexes to a dictionary
	# print(unique_symbol_and_index,cell_lengths)
	# print(type(unique_symbol_and_index),type(cell_lengths),type(struc))
	return unique_symbol_and_index, cell_lengths

def getCifInfo(struc):

	cif = read(struc)  # Read a cif structure

	unique_sites, cell_lengths = get_unique_sites(struc)  # Obtain the nonequivalent site {element symbol: atomic index} and lattice constant of the structure
	# print(unique_sites)
	ele = re.compile(r'([A-Z]([a-z])?)([A-Z]([a-z])?)2(\w)4')     # Match each element
	formula = os.path.splitext(os.path.split(struc)[1])[0]        # Chemical formula of structure
	elements = ele.search(formula)
	A = elements.group(1)  # A site elements in structures
	B = elements.group(3)  # B site elements in structures
	# print(formula.split(".")[0] + '\t→', A,'\t', B)
	# view(cif)
	cif = cif*(3,3,3)  # Expand to 3×3×3 (Sequence of ase expansion direction: Z, Y, X)
	# view(cif)
	A_index = [list(site.values())[0] for site in unique_sites if list(site.keys())[0] == A][0] # Index of non equivalent A atoms
	A_dis = cif.get_distances(A_index+56*13, [atom.index for atom in cif])    # List of distances between non equivalent A atoms and all atoms, sorted from smallest to largest by atomic index

	# print([atom for atom in cif])

	B_index = [list(site.values())[0] for site in unique_sites if list(site.keys())[0] == B][0] # Index of non equivalent B atoms
	B_dis = cif.get_distances(B_index+56*13, [atom.index for atom in cif])    # List of distances between non equivalent B atoms and all atoms, sorted from smallest to largest by atomic index

	X_index = [list(site.values())[0] for site in unique_sites if list(site.keys())[0] == "O"][0] # Index of non equivalent X atoms
	X_dis = cif.get_distances(X_index+56*13, [atom.index for atom in cif])    # List of distances between non equivalent X atoms and all atoms, sorted from smallest to largest by atomic index
# 	print('A_index:', A_index, 'B_index:', B_index, 'X_index:', X_index)
	symbol = cif.get_chemical_symbols()                                       # Obtain a list of element symbols, sorted from smallest to largest by atomic index
	# print(symbol)

	A_symbol_and_distance = pd.DataFrame({'symbol':symbol, 'A_dis':A_dis}) # Environmental atoms and distances of non equivalent A sites
	B_symbol_and_distance = pd.DataFrame({'symbol':symbol, 'B_dis':B_dis}) 
	X_symbol_and_distance = pd.DataFrame({'symbol':symbol, 'X_dis':X_dis}) 

	A_symbol_and_distance = A_symbol_and_distance.sort_values(by='A_dis',axis=0) # Sort by distance
	B_symbol_and_distance = B_symbol_and_distance.sort_values(by='B_dis',axis=0)
	X_symbol_and_distance = X_symbol_and_distance.sort_values(by='X_dis',axis=0)

	A_symbol_and_distance = A_symbol_and_distance.round(10)           # Retain ten decimal places
	B_symbol_and_distance = B_symbol_and_distance.round(10)           
	X_symbol_and_distance = X_symbol_and_distance.round(10)           

	A_col_count = dict(A_symbol_and_distance['A_dis'].value_counts()) # Count the number of atoms with the same atomic distance as the non equivalent A site
	B_col_count = dict(B_symbol_and_distance['B_dis'].value_counts())
	X_col_count = dict(X_symbol_and_distance['X_dis'].value_counts())
# 	print(type(A_symbol_and_distance),type(A_col_count))
	return (A_symbol_and_distance, B_symbol_and_distance, X_symbol_and_distance, A_col_count, B_col_count, X_col_count)

def AET(struc):

	unique_sites, cell_lengths = get_unique_sites(struc)
	A_symbol_and_distance, B_symbol_and_distance, A_col_count, B_col_count = getCifInfo(struc)

	formula = os.path.splitext(os.path.split(struc)[1])[0]
	A = A_symbol_and_distance.iloc[0,0] # Element A
	B = B_symbol_and_distance.iloc[0,0]

	A_col_count_sort = sorted(A_col_count.keys()) # Returns the sorted key value (nearest neighbor distance)
	B_col_count_sort = sorted(B_col_count.keys()
	# Set the CotOff (number of nearest neighbors) in the AET drawing
	A_neighbor = A_col_count_sort[15]
	B_neighbor = B_col_count_sort[15]
	A_col_count = {key : value for key, value in A_col_count.items() if 0 < key <= A_neighbor}
	B_col_count = {key : value for key, value in B_col_count.items() if 0 < key <= B_neighbor}

	A_gap1 = A_col_count_sort[1] # First gap
	B_gap1 = B_col_count_sort[1]

	Ax = [key/A_gap1 for key in A_col_count.keys()]
	Ay = [value for value in A_col_count.values()]
	Bx = [key/B_gap1 for key in B_col_count.keys()]
	By = [value for value in B_col_count.values()]

	# Draw AET results
	# with cbook.get_sample_data('logo2.png') as file:
	# 	im = image.imread(file)
	# im = plt.imread('crystal_structures_1.gif')
	im = plt.imread('AET.png')
	fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12, 6)) # sharex="all"
	fig.subplots_adjust(hspace=0.5)
	fig.figimage(im, 900, 410, zorder=300, alpha=.3,vmin=0.5, vmax=0.3) # Add AET image watermark

	ax1.bar(Ax, Ay, label='Number of Atoms', width=0.01, color='rgb')
	ax1.grid(True, linestyle='--', axis='y')
	ax1.text(1, 15, "$d_{min}$=%s"%(A_gap1), fontsize=15)
	# ax1.axis([0.5, 4.5, 0, 25])
	ax1.set_title("AET of %s"%(A), fontsize=15)
	ax1.set_xlabel("d/d$_{min}$", fontsize=15)
	ax1.set_ylabel("n", fontsize=15)
	ax1.legend(loc='upper left',shadow=True, fontsize=12)

	ax2.bar(Bx, By, label='Number of Atoms', width=0.01, color='rgb')
	ax2.grid(True, linestyle='--', axis='y')
	ax2.text(1, 7.5, "$d_{min}$=%s"%(B_gap1), fontsize=15)
	# ax2.axis([0.5, 4.5, 0, 12.5])
	ax2.set_title("AET of %s"%(B), fontsize=15)
	ax2.set_xlabel("d/d$_{min}$", fontsize=15)
	ax2.set_ylabel("n", fontsize=15)
	ax2.legend(loc='upper left',shadow=True, fontsize=12)

	neighbor = A_col_count_sort.index(A_neighbor)  # Number of Nearest Neighbors Considered
	plt.suptitle("The AET Model of %s%s${_2}$O${_4}$ [Consider the %s neighbors]"%(A,B,neighbor), fontsize=15)
	if not os.path.exists('AET'):
		os.mkdir('AET')
	plt.savefig("AET/%s.png"%(formula))
	plt.close()
	#plt.show()

def getElementsProperties():

	elementsProperties = pd.read_excel("ElementsProperties-1.xlsx", sheet_name=0, index_col=None, header=0)
	elements = pd.read_excel("ElementsProperties.xlsx", sheet_name=2, index_col=None, header = 0)
	elements = list(elements["element"])           
	df = elementsProperties.loc[:,elements]        
	df.dropna(axis=0, how='any', inplace=True)     
	df = df.reset_index(drop=True)                
	return df

def getTarget():

	target = pd.read_excel("target.xlsx", sheet_name=0, index_col=0, header=0)
	singlePointEnergy = target.loc[:,"Energy/atom"]
	formationEnergy   = target.loc[:,"FormationEnergy/atom"]
	latticeConstant   = target.loc[:,"LatticeConstant"]
	bandGap           = pd.read_excel("target.xlsx", sheet_name=1, index_col=0, header=0)
	return (singlePointEnergy,bandGap,formationEnergy,latticeConstant)

def main(path):
	"""
	构造机器学习训练集并写进excel表格,需要将元素性质文件和保存cif结构的文件夹与脚本放在同一路径下

	Parameters
    ----------

    Returns
    -------

	"""

	num_of_cif   = len(os.listdir(path))  # Total number of cif structures

	data_AB         = {}  
	data_AB_AB      = {}  
	data_AB_CE      = {}  
	data_AB_CECE    = {}  
	data_A          = {}  
	data_A_CE       = {}  
	data_B          = {}  
	data_B_CE       = {}  
	data_ABX_ABX    = {}  
	data_ABX_CECECE = {}  
	structure       = []  
	num = 1
	for i in os.listdir(path):
		if num < num_of_cif:

			sys.stdout.write("Processing% d/% d structure% s,% d remaining.\r"
				%(num,num_of_cif,i.split(".")[0],num_of_cif-num))
			num += 1
		else:
			sys.stdout.write("Processing% d/% d structure% s,% d remaining.\r"
				%(num,num_of_cif,i.split(".")[0],num_of_cif-num))
			print()
			print("Generating TrainData file···")

		struc = os.path.join(path, i)

		# =================================Save AET results=================================
		# AET(struc)

		formula = i.split(".")[0]
		structure.append(formula)
		A_symbol_and_distance, B_symbol_and_distance, X_symbol_and_distance, A_col_count, B_col_count, X_col_count = getCifInfo(struc)

		A_col_count_sort = sorted(A_col_count.keys())
		B_col_count_sort = sorted(B_col_count.keys())
		X_col_count_sort = sorted(X_col_count.keys())
# 		print(A_col_count_sort)
		# del A_col_count_sort[0] 
		# del B_col_count_sort[0]

		# =================================Set CotOff in AET=================================
		neighbor_num = 3 
		A_neighbor = A_col_count_sort[neighbor_num]  
		B_neighbor = B_col_count_sort[neighbor_num]
		X_neighbor = X_col_count_sort[neighbor_num]
# 		print(A_neighbor)
		A_symbol_and_distance = A_symbol_and_distance.loc[A_symbol_and_distance['A_dis'] <= A_neighbor]
		B_symbol_and_distance = B_symbol_and_distance.loc[B_symbol_and_distance['B_dis'] <= B_neighbor]
		X_symbol_and_distance = X_symbol_and_distance.loc[X_symbol_and_distance['X_dis'] <= X_neighbor]

		A_symbol   = A_symbol_and_distance['symbol']
		A_distance = A_symbol_and_distance['A_dis'] 
		B_symbol   = B_symbol_and_distance['symbol']
		B_distance = B_symbol_and_distance['B_dis'] 
		X_symbol   = X_symbol_and_distance['symbol']
		X_distance = X_symbol_and_distance['X_dis'] 

		prop = getElementsProperties()        

		# Center Environment Properties of Unequivalent A-sites
		A_weights     = sum([1/d for d in A_distance if d != 0])   
		A_envir_prop  = pd.Series(0 for i in range(prop.shape[0]))  
		for element, distance in zip(A_symbol, A_distance):
			if distance == 0:
				A_center_ele  = element                     
				A_center_prop = prop[A_center_ele]    
			else:
				A_weight      = (1/distance) / A_weights        
				A_envir_prop  = A_envir_prop.add(prop[element]*A_weight, axis=0) 
		A_CE_prop     = pd.concat([A_center_prop, A_envir_prop], axis=0, ignore_index=True) 
		A_CplusE_prop = A_center_prop.add(A_envir_prop, axis=0)  

		# Center Environment Properties of Unequivalent B-sites
		B_weights     = sum([1/d for d in B_distance if d != 0])     
		B_envir_prop  = pd.Series(0 for i in range(prop.shape[0])) 
		for element, distance in zip(B_symbol, B_distance):
			if distance == 0:
				B_center_ele  = element                   
				B_center_prop = prop[B_center_ele]          
			else:
				B_weight      = (1/distance) / B_weights       
				B_envir_prop  = B_envir_prop.add(prop[element]*B_weight,axis=0)            
		B_CE_prop     = pd.concat([B_center_prop, B_envir_prop], axis=0, ignore_index=True) 
		B_CplusE_prop = B_center_prop.add(B_envir_prop, axis=0)  

		# Center Environment Properties of Unequivalent X-sites
		X_weights     = sum([1/d for d in X_distance if d != 0])     
		X_envir_prop  = pd.Series(0 for i in range(prop.shape[0]))  
		for element, distance in zip(X_symbol, X_distance):
			if distance == 0:
				X_center_ele  = element                    
				X_center_prop = prop[X_center_ele]          
			else:
				X_weight      = (1/distance) / X_weights    
				X_envir_prop  = X_envir_prop.add(prop[element]*X_weight,axis=0)    
		X_CE_prop     = pd.concat([X_center_prop, X_envir_prop], axis=0, ignore_index=True)  
		X_CplusE_prop = X_center_prop.add(X_envir_prop, axis=0)   

		data_AB[formula]         = A_CplusE_prop.add(B_CplusE_prop, axis=0)  
		data_AB_AB[formula]      = pd.concat([A_CplusE_prop, B_CplusE_prop], axis=0) 
		data_AB_CE[formula]      = A_CE_prop.add(B_CE_prop, axis=0)
		data_AB_CECE[formula]    = pd.concat([A_CE_prop, B_CE_prop], axis=0) 
		data_A[formula]          = A_CplusE_prop
		data_A_CE[formula]       = A_CE_prop
		data_B[formula]          = B_CplusE_prop
		data_B_CE[formula]       = B_CE_prop
		data_ABX_ABX[formula]    = pd.concat([A_CplusE_prop, B_CplusE_prop, X_CplusE_prop], axis=0)
		data_ABX_CECECE[formula] = pd.concat([A_CE_prop, B_CE_prop, X_CE_prop], axis=0) 
	# print(data)


	data_AB         = pd.DataFrame(data_AB).T             # Convert the CE element properties of all structures to DataFrame format and transpose them
	data_AB_AB      = pd.DataFrame(data_AB_AB).T
	data_AB_CE      = pd.DataFrame(data_AB_CE).T
	data_AB_CECE    = pd.DataFrame(data_AB_CECE).T

	data_A          = pd.DataFrame(data_A).T
	data_A_CE       = pd.DataFrame(data_A_CE).T
	data_B          = pd.DataFrame(data_B).T
	data_B_CE       = pd.DataFrame(data_B_CE).T

	data_ABX_ABX    = pd.DataFrame(data_ABX_ABX).T
	data_ABX_CECECE = pd.DataFrame(data_ABX_CECECE).T

	# Add Column Name
	col = list(map(lambda x:"CE" + str(x), range(1,prop.shape[0]+1)))
	A   = list(map(lambda x:"A"  + str(x), range(1,prop.shape[0]+1)))
	B   = list(map(lambda x:"B"  + str(x), range(1,prop.shape[0]+1)))
	C   = list(map(lambda x:"C"  + str(x), range(1,prop.shape[0]+1)))
	E   = list(map(lambda x:"E"  + str(x), range(1,prop.shape[0]+1)))
	X   = list(map(lambda x:"X"  + str(x), range(1,prop.shape[0]+1)))
	AC  = list(map(lambda x:"AC" + str(x), range(1,prop.shape[0]+1)))
	AE  = list(map(lambda x:"AE" + str(x), range(1,prop.shape[0]+1)))
	BC  = list(map(lambda x:"BC" + str(x), range(1,prop.shape[0]+1)))
	BE  = list(map(lambda x:"BE" + str(x), range(1,prop.shape[0]+1)))
	XC  = list(map(lambda x:"XC" + str(x), range(1,prop.shape[0]+1)))
	XE  = list(map(lambda x:"XE" + str(x), range(1,prop.shape[0]+1)))
# 	print(col)
	col        = col
	col_AB     = A + B
	col_CE     = C + E
	col_CECE   = AC + AE + BC + BE
	col_ABX    = A + B + X
	col_CECECE = AC + AE + BC + BE + XC + XE

	data_AB.columns         = col
	data_AB_AB.columns      = col_AB
	data_AB_CE.columns      = col_CE
	data_AB_CECE.columns    = col_CECE
	data_A.columns          = col
	data_A_CE.columns       = col_CE
	data_B.columns          = col
	data_B_CE.columns       = col_CE
	data_ABX_ABX.columns    = col_ABX
	data_ABX_CECECE.columns = col_CECECE

	# 得到target数据
	target = pd.read_excel("target.xlsx", sheet_name=0, index_col=0, header=0)
	singlePointEnergy    = target.loc[structure,"Energy/atom"]   
	# bandGap              = bandGap.loc[structure,"gap"]
	formationEnergy      = target.loc[structure,"FormationEnergy/atom"]
	latticeConstant      = target.loc[structure,"LatticeConstant"]

	# Create a new trainData folder and save the output results
	if not os.path.exists('trainData-3/%s_neighbor'%(neighbor_num)):
		os.makedirs('trainData-3/%s_neighbor'%(neighbor_num))
	os.chdir('trainData-3/%s_neighbor'%(neighbor_num))             # Change the current working path to the TrainData storage path

	# Single point energy TrainData
	SPE_data_AB          = pd.concat([data_AB, singlePointEnergy], axis=1)
	SPE_data_AB_AB       = pd.concat([data_AB_AB, singlePointEnergy], axis=1)
	SPE_data_AB_CE       = pd.concat([data_AB_CE, singlePointEnergy], axis=1)
	SPE_data_AB_CECE     = pd.concat([data_AB_CECE, singlePointEnergy], axis=1)
	SPE_data_A           = pd.concat([data_A, singlePointEnergy], axis=1)
	SPE_data_A_CE        = pd.concat([data_A_CE, singlePointEnergy], axis=1)
	SPE_data_B           = pd.concat([data_B, singlePointEnergy], axis=1)
	SPE_data_B_CE        = pd.concat([data_B_CE, singlePointEnergy], axis=1)
	SPE_data_ABX_ABX     = pd.concat([data_ABX_ABX, singlePointEnergy], axis=1)
	SPE_data_ABX_CECECE  = pd.concat([data_ABX_CECECE, singlePointEnergy], axis=1)

	SPE_data_AB.to_excel('SPE_trainData_AB.xlsx')
	SPE_data_AB_AB.to_excel('SPE_trainData_AB_AB.xlsx')
	SPE_data_AB_CE.to_excel('SPE_trainData_AB_CE.xlsx')
	SPE_data_AB_CECE.to_excel('SPE_trainData_AB_CECE.xlsx')
	SPE_data_A.to_excel('SPE_trainData_A.xlsx')
	SPE_data_A_CE.to_excel('SPE_trainData_A_CE.xlsx')
	SPE_data_B.to_excel('SPE_trainData_B.xlsx')
	SPE_data_B_CE.to_excel('SPE_trainData_B_CE.xlsx')
	SPE_data_ABX_ABX.to_excel('SPE_trainData_ABX_ABX.xlsx')
	SPE_data_ABX_CECECE.to_excel('SPE_trainData_ABX_CECECE.xlsx')

	# band gap TrainData
	# GAP_data_AB          = pd.concat([data_AB, bandGap], axis=1)
	# GAP_data_AB_AB       = pa.concat([data_AB_AB, bandgap], axis=1)
	# GAP_data_AB_CE       = pd.concat([data_AB_CE, bandGap], axis=1)
	# GAP_data_AB_CECE     = pd.concat([data_AB_CECE, bandGap], axis=1)
	# GAP_data_A           = pd.concat([data_A, bandGap], axis=1)
	# GAP_data_A_CE        = pd.concat([data_A_CE, bandGap], axis=1)
	# GAP_data_B           = pd.concat([data_B, bandGap], axis=1)
	# GAP_data_B_CE        = pd.concat([data_B_CE, bandGap], axis=1)

	# GAP_data_AB.to_excel('GAP_trainData_AB.xlsx')
	# GAP_data_AB_AB.to_excel('GAP_trainData_AB_AB.xlsx')
	# GAP_data_AB_CE.to_excel('GAP_trainData_AB_CE.xlsx')
	# GAP_data_AB_CECE.to_excel('GAP_trainData_AB_CECE.xlsx')
	# GAP_data_A.to_excel('GAP_trainData_A.xlsx')
	# GAP_data_A_CE.to_excel('GAP_trainData_A_CE.xlsx')
	# GAP_data_B.to_excel('GAP_trainData_B.xlsx')
	# GAP_data_B_CE.to_excel('GAP_trainData_B_CE.xlsx')

	# formation energy TrainData
	FE_data_AB           = pd.concat([data_AB, formationEnergy], axis=1)
	FE_data_AB_AB        = pd.concat([data_AB_AB, formationEnergy], axis=1)
	FE_data_AB_CE        = pd.concat([data_AB_CE, formationEnergy], axis=1)
	FE_data_AB_CECE      = pd.concat([data_AB_CECE, formationEnergy], axis=1)
	FE_data_A            = pd.concat([data_A, formationEnergy], axis=1)
	FE_data_A_CE         = pd.concat([data_A_CE, formationEnergy], axis=1)
	FE_data_B            = pd.concat([data_B, formationEnergy], axis=1)
	FE_data_B_CE         = pd.concat([data_B_CE, formationEnergy], axis=1)
	FE_data_ABX_ABX      = pd.concat([data_ABX_ABX, formationEnergy], axis=1)
	FE_data_ABX_CECECE   = pd.concat([data_ABX_CECECE, formationEnergy], axis=1)

	FE_data_AB.to_excel('FE_trainData_AB.xlsx')
	FE_data_AB_AB.to_excel('FE_trainData_AB_AB.xlsx')
	FE_data_AB_CE.to_excel('FE_trainData_AB_CE.xlsx')
	FE_data_AB_CECE.to_excel('FE_trainData_AB_CECE.xlsx')
	FE_data_A.to_excel('FE_trainData_A.xlsx')
	FE_data_A_CE.to_excel('FE_trainData_A_CE.xlsx')
	FE_data_B.to_excel('FE_trainData_B.xlsx')
	FE_data_B_CE.to_excel('FE_trainData_B_CE.xlsx')
	FE_data_ABX_ABX.to_excel('FE_trainData_ABX_ABX.xlsx')
	FE_data_ABX_CECECE.to_excel('FE_trainData_ABX_CECECE.xlsx')

	# lattice constant TrainData
	LC_data_AB           = pd.concat([data_AB, latticeConstant], axis=1)
	LC_data_AB_AB        = pd.concat([data_AB_AB, latticeConstant], axis=1)
	LC_data_AB_CE        = pd.concat([data_AB_CE, latticeConstant], axis=1)
	LC_data_AB_CECE      = pd.concat([data_AB_CECE, latticeConstant], axis=1)
	LC_data_A            = pd.concat([data_A, latticeConstant], axis=1)
	LC_data_A_CE         = pd.concat([data_A_CE, latticeConstant], axis=1)
	LC_data_B            = pd.concat([data_B, latticeConstant], axis=1)
	LC_data_B_CE         = pd.concat([data_B_CE, latticeConstant], axis=1)
	LC_data_ABX_ABX      = pd.concat([data_ABX_ABX, latticeConstant], axis=1)
	LC_data_ABX_CECECE   = pd.concat([data_ABX_CECECE, latticeConstant], axis=1)

	LC_data_AB.to_excel('LC_trainData_AB.xlsx')
	LC_data_AB_AB.to_excel('LC_trainData_AB_AB.xlsx')
	LC_data_AB_CE.to_excel('LC_trainData_AB_CE.xlsx')
	LC_data_AB_CECE.to_excel('LC_trainData_AB_CECE.xlsx')
	LC_data_A.to_excel('LC_trainData_A.xlsx')
	LC_data_A_CE.to_excel('LC_trainData_A_CE.xlsx')
	LC_data_B.to_excel('LC_trainData_B.xlsx')
	LC_data_B_CE.to_excel('LC_trainData_B_CE.xlsx')
	LC_data_ABX_ABX.to_excel('LC_trainData_ABX_ABX.xlsx')
	LC_data_ABX_CECECE.to_excel('LC_trainData_ABX_CECECE.xlsx')

if __name__ == '__main__':
	main("cif_conventional_unrelax")

