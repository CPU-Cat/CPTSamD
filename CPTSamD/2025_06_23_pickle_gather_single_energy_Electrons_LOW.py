import datetime
import itertools
import math
import multiprocessing as mp
import os
import pickle
import random
import time

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from magpylib import Collection
from scipy.integrate import solve_ivp
import glob

# Original code goals:
#     import each file from within single energy & mag strength
#     get info out of it and append to new arrays
#     save to new pickle file
# 01_07_25  trying to go through multiple folders, on SCC
# 2/5/25 gathering 0mT 10k, 10energy set on SCC
# 2/9/25 note!!! this one has some special file name stuff because of messed up runcode index in naming so everything went in last folder
# 2/12 updated code, 1k with angles etc
# 2/22 updating for 10k updated code fullrun, edit runcodefoldernamestart for folder to look for files in, using copy paste from runcode folder file naming
#3/7  accidentally made first 0.3 file names with 0d6 real radius, so must fix and rerun
# 4/2 extremeley mild editing for 100kruns


# # copy pasted info for total # particles, energy + mag strength from runcode
# Required copy paste: totalnumberofparticles, Energy_range_eV, magnetSTRENGTH, radiuslist, foldernamestart


totalnumberofparticles = 25 #10000
Energy_range_eV = np.array(np.logspace(0, 3, num=4))#17))#17))#17 for 2 at each level 
magnetSTRENGTH = np.array([1000])#np.arange(0,2800, 250)) #250))#np.arange(500,2800, 250)) #11, 500]) # mT!!!! change to T in loop creation of magnet array
radiuslist = np.array([0.5])
travlength = 2.0 # in m travelling length for proton (used to calculate timing)
maxsteptimedivider = 40 #number divided by time for travel that is set as max step, have used 30

foldernamestart = f"../electron_data_06_22_25_straightdown_posset_LOW/"

# particle_simulator/electron_data_06_21_25_straightdown
# https://scc-ondemand2.bu.edu/pun/sys/dashboard/files/fs//projectnb/sw-prop/ckpawu/particle_simulator/electron_data_06_21_25_straightdown/2025-06-21_10000.0keV_1000mT/2025-06-21_1000mT_10000.0keV_pno00918.p

runcodefoldernamestart = foldernamestart

filedate = "2025-06-23"
gatheredpicklefolder= "2025_06_23_gatheredpfiles_electron25_LOW" #folder to put gathered p files into
check_folder = os.path.isdir(f"../{gatheredpicklefolder}")
if not check_folder:           # If folder doesn't exist, then create it.
    os.makedirs(f"../{gatheredpicklefolder}")

date = datetime.datetime.today().strftime('%Y-%m-%d') #date used to name files


for EnergyIndex in range(len(Energy_range_eV)):
    Energy = Energy_range_eV[EnergyIndex]/1000 #1/7 fix: MUST have ev-->keV

    for MagIndex in range(len(magnetSTRENGTH)):
        magstrengthRun = magnetSTRENGTH[MagIndex] 
        #data_02_09_25_straightdown/straight_100000.0keV_0.001mT/2025-02-09_straight_0.001mT_0.001keV_pno00001.p
        #MESSED UP FOLDER INDEX SO HAVE TO MAKE SPECIAL IN THIS GATHER FILE
        #   runcode had file names f"{today_date}_straight_{round(magstrengthRun,7)}mT_{round(Energy,7)}keV_"
        # data_02_17_25_updatedcode /

        foldername = f"{runcodefoldernamestart}{filedate}_{round(Energy,5)}keV_{round(magstrengthRun,5)}mT/"
        # /projectnb/sw-prop/ckpawu/particle_simulator/data_02_17_25_updatedcode/updatedcode_0.001keV_0mT
        #straight_{round(Energy,5)}keV_{round(magstrengthRun,5)}mT/"
        #f"../data_02_09_25_straightdown/straight_{round(Energy,5)}keV_{round(magstrengthRun,5)}mT/" #f"../data_02_09_25/{round(Energy,5)}keV_{round(magstrengthRun,5)}mT"

        fileenergy = round(Energy,5)
        filemag = round(magstrengthRun,5)
        run_name = f"../{gatheredpicklefolder}/{date}_pfilecomp_run{filedate}_updated_{fileenergy}keV_{filemag}mT"
        ###################################
        # grab files and put together new pickle
        ##################################
        #MESSED UP NAMES SO MUST DO THIS SPECIAL: _straight_0.001mT_0.001keV
        print("names looking for: ", foldername+f'/{filedate}') # for record + debugging
        files_of_energy = glob.glob(foldername+f'/{filedate}*')
        # print("names looking for: ", foldername+f'/{filedate}_straight_{round(magstrengthRun,7)}mT_{round(Energy,7)}keV') # for record + debugging
        # files_of_energy = glob.glob(foldername+f'/{filedate}_straight_{round(magstrengthRun,7)}mT_{round(Energy,7)}keV*')
        print("length files number of files: ", len(files_of_energy))
        dataset_save = { 'energy': [],
                'positions': [],
                'velocities': [],
                'maxstep' : [],
                'totaltime' : [],
                'randomseed': []  }
        for filenamenumber in range(len(files_of_energy)):
            run_name1 = files_of_energy[filenamenumber] 
            dataset1 = pickle.load(open(run_name1, "rb"))
            posxyz1a = dataset1['pvectorlist'] #xyz position data  FROM OLD CODE, NEW SHOULD HAVE NO NAN
            posxyz1b = posxyz1a[~np.isnan(posxyz1a)] #REMOVE THE NAN FROM data  FROM OLD CODE
            # posxyz1b = dataset1['pvectorlist']
            posxyz1 = np.reshape(posxyz1b , [int(len(posxyz1b)/3), 3])
            velxyz1a = dataset1['vvectorlist'] #xyz velocity data 
            velxyz1b = velxyz1a[~np.isnan(velxyz1a)] #REMOVE THE NAN FROM data
            # velxyz1b = dataset1['vvectorlist']
            velxyz1 = np.reshape(velxyz1b , [int(len(velxyz1b)/3), 3])
            maxstep = dataset1['maxstep']
            totaltime = dataset1['totaltime']
            randomseed = dataset1['randomseed']
            dataset_save['energy'].append(fileenergy)
            dataset_save['positions'].append(posxyz1)
            dataset_save['velocities'].append(velxyz1)
            dataset_save['maxstep'].append(maxstep)
            dataset_save['totaltime'].append(totaltime)
            dataset_save['randomseed'].append(randomseed)
        pickle.dump(dataset_save, open(f"{run_name}.p", "wb"))

# #############################################
# # READ pickle file to make sure it works
# #############################################
# datafilename = r'2023-08-09_pickledfilecompilation_1.01keV_TEST_500filesample.p'
# #"C:\Users\ckpawu\Documents\01_LEXI_Particle_Simulations\2023_08_09_paperprep_4plots_picklegather\2023-08-09_pickledfilecompilation_1.01keV_TEST.p"
# dataset1 = pickle.load(open(datafilename, "rb"))
# print(len(dataset1), len(dataset1['positions']))
# print('dataset')