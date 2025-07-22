import datetime
import glob
import os
import pickle

import numpy as np

"""
Script to gather and process single-energy, single-magnetic-strength particle files.

Overview:
    - Imports each file for a given energy and magnetic field strength.
    - Extracts relevant data and appends to new arrays.
    - Saves consolidated data to new pickle files.

Changelog:
    - 2025-01-07: Added support for processing multiple folders (SCC).
    - 2025-02-05: Gathered 0mT, 10k particles, 10 energy set on SCC.
    - 2025-02-09: Special filename handling due to runcode index issue.
    - 2025-02-12: Updated for 1k runs with angles.
    - 2025-02-22: Updated for 10k full run; folder/file naming improvements.
    - 2025-03-07: Fixed radius naming error in 0.3 files; rerun required.
    - 2025-04-02: Minor edits for 100k runs.
    - 2025-04-02: Minor edits for 100k runs.

Note:
    Requires manual copy-paste of: totalnumberofparticles, Energy_range_eV, magnetSTRENGTH,
    radiuslist, foldernamestart.
"""

totalnumberofparticles = 25  # 10000
Energy_range_eV = np.array(np.logspace(0, 3, num=4))  # 17))#17))#17 for 2 at each level
magnetSTRENGTH = np.array([1000])

radiuslist = np.array([0.5])
travlength = 2.0  # in m travelling length for proton (used to calculate timing)
maxsteptimedivider = 40  # number divided by time for travel that is set as max step, have used 30

foldernamestart = "../electron_data_06_22_25_straightdown_posset_LOW/"

# particle_simulator/electron_data_06_21_25_straightdown
runcodefoldernamestart = foldernamestart

filedate = "2025-06-23"
# folder to put gathered p files into
gatheredpicklefolder = "2025_06_23_gatheredpfiles_electron25_LOW"
check_folder = os.path.isdir(f"../{gatheredpicklefolder}")
if not check_folder:  # If folder doesn't exist, then create it.
    os.makedirs(f"../{gatheredpicklefolder}")

date = datetime.datetime.today().strftime("%Y-%m-%d")  # date used to name files

for EnergyIndex in range(len(Energy_range_eV)):
    Energy = Energy_range_eV[EnergyIndex] / 1000  # 1/7 fix: MUST have ev-->keV

    for MagIndex in range(len(magnetSTRENGTH)):
        magstrengthRun = magnetSTRENGTH[MagIndex]
        foldername = (
            f"{runcodefoldernamestart}{filedate}_{round(Energy,5)}keV_{round(magstrengthRun,5)}mT/"
        )

        fileenergy = round(Energy, 5)
        filemag = round(magstrengthRun, 5)
        run_name = f"../{gatheredpicklefolder}/{date}_pfilecomp_run{filedate}_updated_{fileenergy}keV_{filemag}mT"
        ###################################
        # grab files and put together new pickle
        ##################################
        print("names looking for: ", foldername + f"/{filedate}")  # for record + debugging
        files_of_energy = glob.glob(foldername + f"/{filedate}*")

        print("length files number of files: ", len(files_of_energy))
        dataset_save = {
            "energy": [],
            "positions": [],
            "velocities": [],
            "maxstep": [],
            "totaltime": [],
            "randomseed": [],
        }
        for filenamenumber in range(len(files_of_energy)):
            run_name1 = files_of_energy[filenamenumber]
            dataset1 = pickle.load(open(run_name1, "rb"))
            posxyz1a = dataset1["pvectorlist"]
            posxyz1b = posxyz1a[~np.isnan(posxyz1a)]  # REMOVE THE NAN FROM data
            posxyz1 = np.reshape(posxyz1b, [int(len(posxyz1b) / 3), 3])
            velxyz1a = dataset1["vvectorlist"]
            velxyz1b = velxyz1a[~np.isnan(velxyz1a)]
            velxyz1 = np.reshape(velxyz1b, [int(len(velxyz1b) / 3), 3])
            maxstep = dataset1["maxstep"]
            totaltime = dataset1["totaltime"]
            randomseed = dataset1["randomseed"]
            dataset_save["energy"].append(fileenergy)
            dataset_save["positions"].append(posxyz1)
            dataset_save["velocities"].append(velxyz1)
            dataset_save["maxstep"].append(maxstep)
            dataset_save["totaltime"].append(totaltime)
            dataset_save["randomseed"].append(randomseed)
        pickle.dump(dataset_save, open(f"{run_name}.p", "wb"))
