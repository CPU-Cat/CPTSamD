import datetime
import itertools
import math
import multiprocessing as mp
import os
import pickle
import random

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from magpylib import Collection
from scipy.integrate import solve_ivp

#02_08 version messed up the folder 
#02_12 making multiple changes and corrections -- angles, initialized x y, time package removal, folder name, solve-ivp maxlength increase, 
#02_17 running larger range to check hitrate analysis patterns
#03_05 editting so that testing new angle range --- going symmetric hemisphere for x0 y0 z0 and angle choice w/in mpo array hit
#  https://mathworld.wolfram.com/SphericalCoordinates.html https://mathworld.wolfram.com/SpherePointPicking.html
#03_06 trying larger hemisphere radius -0.3m-->0.6m and travel length -->1.8m
#3/7 trying diff radius lists, removing unused vals like holderedge and halflengthfov, adding loops for radius changing and finding initialization in first loop of radius
# 3//12 updating to keep edits for diff radius  but changing the initialization to center intersection with radius
#   MPO array sidelength ~0.14m, and if need radius at least 0.3m away from array, radius 0.34 min
# 3/13 doing fullrun since hemi intersection initialization seems to have worked to be radii independent
# 4/1 adjust random sphere picking to restrict to angle limited by CAD to 47 degrees; also doing run w/100k particles, for 15 items (5x nrg, 3x mag strength)
# 4/4 adjust for 28 omp, reduce the number of print statements to only on the number of particle ==1
# 4/8 adjust code for electrons

totalnumberofparticles = 25 #10000
Energy_range_eV = np.array(np.logspace(0, 3, num=4))#17))#17))#17 for 2 at each level 
magnetSTRENGTH = np.array([1000])#np.arange(0,2800, 250)) #250))#np.arange(500,2800, 250)) #11, 500]) # mT!!!! change to T in loop creation of magnet array
radiuslist = np.array([0.5])
travlength = 2.0 # in m travelling length for proton (used to calculate timing)
maxsteptimedivider = 40 #number divided by time for travel that is set as max step, have used 30

foldernamestart = f"../electron_data_06_22_25_straightdown_posset_LOW/"
# datafilename = "../2025_04_22_gatheredpfiles_electron100kALL/2025-04-21_pfilecomp_run2025-04-08_updated_0.01keV_1000mT.p"
# https://scc-ondemand2.bu.edu/pun/sys/dashboard/files/fs//projectnb/sw-prop/ckpawu/particle_simulator/2025_04_22_gatheredpfiles_electron100kALL
#datafile for getting initial positions from
today_date = datetime.datetime.today().strftime('%Y-%m-%d') #date used to name files

Energy_range = Energy_range_eV/1000 #keV
#make the folders for the runs before the runs
for radiusval in radiuslist:
    for Energy_indie in Energy_range:
        for magstrengthRun_indie in magnetSTRENGTH:
            folder_name = f"{foldernamestart}{today_date}_{round(Energy_indie,5)}keV_{round(magstrengthRun_indie,5)}mT/" #f"../data/{round(Energy, 3)}Kev/"
            # Check if directory exists, if not, create it
            check_folder = os.path.isdir(folder_name)
            if not check_folder:           # If folder doesn't exist, then create it.
                os.makedirs(folder_name)

m = (9.109*10**-31) #(1.67262192*10**-27) # proton (1.67262192*10**-27) #electrons (9.109*10**-31)# mass in kg
mass =  (9.109*10**-31) #(1.67262192*10**-27)
q    = (1.602176634*10**-19)  #  charge in coulumb #do positive for electron, negative for proton#  since was made og for electron
halflength = 5.5120/2*2.54*1/100 #in to cm to m

c = 2.99e8  # speed of light m/s
kev  = 1.0e3*1.6e-19 #convert kev to joules

# Dimensions of lengths in mm --> 11/1 changed to m
LEXIside  = 36/1000   # length of side of one lens - shelf = 40-2-2
LEXIsuppt =  5/1000   # shelf +between lens = 2+2+1
magnetdimx = 3.175/1000 #mag dim = 12.7 x 6.35 x 3.175 mm^3
magnetdimy = 12.7*3/1000  # three mag stacked
mdimy      = 12.7 /1000 # y mag dim for 1 magnet
magnetdimz = 6.35/1000
xedge    = 2.286/1000 #mm,= 0.09" =0.08648802"
spacing  = 1.016/1000 #mm
yspacing = 1.016/1000 #mm (same as x spacing?)
y_up     = mdimy + 2*yspacing
y_down   = mdimy
maghold  = 3.4544/1000 #mm, =0.136"
yhold    = 13.5678418/1000 #13.5678418mm 0.534167"
suptadjust = 1.04775/1000
x_inneredge = ( (LEXIside + magnetdimx)/2+(LEXIside+LEXIsuppt+xedge)                   )
x_outeredge = ( (LEXIside - magnetdimx)/2+(LEXIside+LEXIsuppt+xedge+spacing+2*maghold) )
y_inner     = ( (LEXIside + yhold     )/2+(LEXIsuppt - suptadjust)                         )
y_mid       = ( (LEXIside + yhold     )/2+(LEXIsuppt -suptadjust) + (yhold+yspacing)      )
y_outer     = ( (LEXIside + yhold     )/2+(LEXIsuppt -suptadjust) + 2*(yhold+yspacing)    )

def magnetarraycreation(magnet_strength):
    # create magnets
    s1a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, 0, 0))
    s1b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, mdimy, 0))
    s1c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, -mdimy, 0))

    s2a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, 0, 0))
    s2b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, mdimy, 0))
    s2c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, -mdimy, 0))

    s3a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+magnetdimx)/2+(LEXIside+LEXIsuppt+xedge), 0, 0))
    s3b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+magnetdimx)/2+(LEXIside+LEXIsuppt+xedge), mdimy, 0))
    s3c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+magnetdimx)/2+(LEXIside+LEXIsuppt+xedge), -mdimy, 0))

    s4a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+magnetdimx)/2-(LEXIside+LEXIsuppt+xedge), 0, 0))
    s4b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+magnetdimx)/2-(LEXIside+LEXIsuppt+xedge), mdimy, 0))
    s4c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+magnetdimx)/2-(LEXIside+LEXIsuppt+xedge), -mdimy, 0))

    s5a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, (LEXIside+LEXIsuppt), 0))
    s5b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, (LEXIside+LEXIsuppt)+ mdimy, 0))
    s5c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, (LEXIside+LEXIsuppt)-mdimy, 0))

    s6a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, (LEXIside+LEXIsuppt), 0))
    s6b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, (LEXIside+LEXIsuppt)+ mdimy, 0))
    s6c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, (LEXIside+LEXIsuppt)-mdimy, 0))

    s7a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_inneredge,   y_mid, 0))
    s7b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_inneredge, y_inner, 0))
    s7c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_inneredge, y_outer, 0))

    s8a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -x_inneredge,   y_mid, 0))
    s8b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -x_inneredge, y_inner, 0))
    s8c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -x_inneredge, y_outer, 0))

    s9a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, -(LEXIside+LEXIsuppt), 0))
    s9b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, -(LEXIside+LEXIsuppt+ mdimy), 0))
    s9c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( (LEXIside+LEXIsuppt)/2, -(LEXIside+LEXIsuppt- mdimy), 0))

    s10a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, -(LEXIside+LEXIsuppt), 0))
    s10b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, -(LEXIside+LEXIsuppt+ mdimy), 0))
    s10c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -(LEXIside+LEXIsuppt)/2, -(LEXIside+LEXIsuppt- mdimy), 0))

    s11a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_inneredge, -y_mid, 0))
    s11b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_inneredge, -y_inner, 0))
    s11c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_inneredge, -y_outer, 0))

    s12a  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -x_inneredge, -y_mid, 0))
    s12b  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -x_inneredge, -y_inner, 0))
    s12c  = magpy.magnet.Cuboid(polarization=(-magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( -x_inneredge, -y_outer, 0))

    s13a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_outeredge,   y_mid, 0))
    s13b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_outeredge,   y_inner, 0))
    s13c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_outeredge,   y_outer, 0))

    s14a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_outeredge,   -y_mid, 0))
    s14b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_outeredge,   -y_inner, 0))
    s14c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = ( x_outeredge,   -y_outer, 0))

    s15a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = (-x_outeredge,   y_mid, 0))
    s15b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = (-x_outeredge,   y_inner, 0))
    s15c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = (-x_outeredge,   y_outer, 0))

    s16a  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = (-x_outeredge,   -y_mid, 0))
    s16b  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = (-x_outeredge,   -y_inner, 0))
    s16c  = magpy.magnet.Cuboid(polarization=(magnet_strength,0,0), dimension=(magnetdimx,mdimy,magnetdimz), position = (-x_outeredge,   -y_outer, 0))
    
    # create collection
    return( Collection(s1a, s1b, s1c, s2a, s2b, s2c, s3a, s3b, s3c, s4a, s4b, s4c, s5a, s5b, s5c, s6a, s6b, s6c,
        s7a, s7b, s7c, s8a, s8b, s8c, s9a, s9b, s9c, s10a, s10b, s10c, s11a, s11b, s11c, s12a, s12b, s12c,
        s13a, s13b, s13c, s14a, s14b, s14c, s15a, s15b, s15c, s16a, s16b, s16c) )


def function45 (timestep, variables_array):
    #function that goes into scipy rk45
    #takes in time and array of variables, needs to return same size array of variables
    x0 = variables_array[0]
    y0 = variables_array[1]
    z0 = variables_array[2]
    vx0 = variables_array[3]
    vy0 = variables_array[4]
    vz0 = variables_array[5]
    #deriv of pos is the velocities BUT deriv of velocity must be calc'd based on lorentz force
    positionxyz = [x0, y0, z0]
    v = [vx0, vy0, vz0]
    Bfield = magnetfield(positionxyz)
    force_lorentz = (q/m)*(np.cross(v, Bfield))/gamma
    axi = force_lorentz[0]
    ayi = force_lorentz[1]
    azi = force_lorentz[2]
    variables_array_return = [vx0, vy0, vz0, axi, ayi, azi] #return derivatives of pos (velocity) and of velocity (acceleration)
    return(variables_array_return)

def magnetfield(position):
    #input position [x, y, z] and mag field return real!! bx by bz returned
    Bfield = c_mag_array.getB([position[0],  position[1], position[2]]) # getb input = m, output = T
    return(Bfield)


##############QUDSI CODE###################
def compute_vars(*args):
    r"""
    The function takes particle number as input and gives a bunch of output and saves them an HDF
    and pickle file.

    Parameters
    ----------
    args : integer
        Input particle number for computation

    Raises
    ------
    None

    Returns
    -------
    pvectorlist : array of shape steps x 3
        p Vector list

    vvectorlist : array of shape steps x 3
        V vector list

     """

    numberofparticles = args[0][0]
    numberofparticles = numberofparticles +1

    # trying to change the x and y
    x_0 = x0list[numberofparticles - 1]  # 0.01 # 1cm -->m
    y_0 = y0list[numberofparticles - 1]  # 0.00
    z_0 = z0list[numberofparticles - 1]  
    # eventually change v directions also
    vx_0 = vx0list[numberofparticles - 1]  # m/s
    vy_0 = vy0list[numberofparticles - 1]
    vz_0 = vz0list[numberofparticles - 1]
    pos_vel_0 = [x_0,  y_0,  z_0, vx_0, vy_0, vz_0]

    output_of_ivprk45 = solve_ivp(function45, [0,timend], pos_vel_0, method= 'RK45' , max_step = maxstep, rtol = 0.00001, atol = 0.000001)#DEFAULT: rtol = 0.001, atol = 0.000001)
    #save data to pickle file
    fullpos_veloutput = np.array(output_of_ivprk45.y)

    folder_name =  f"{foldernamestart}{today_date}_{round(Energy,5)}keV_{round(magstrengthRun,5)}mT/" 
    #f"{foldernamestart}_{radiusval}m_{round(Energy_indie,5)}keV_{round(magstrengthRun_indie,5)}mT/" #f"../data/{round(Energy, 3)}Kev/"
            
    # Check if directory exists, if not, create it
    # check_folder = os.path.isdir(folder_name)
    # if not check_folder:           # If folder doesn't exist, then create it.
    #     os.makedirs(folder_name)
        
    run_name = f"{folder_name}{date}pno{str(numberofparticles).zfill(5)}" 
    if numberofparticles == 1:
        print("run: ", run_name)
    # Save np arrays to file
    # Create a dictionary with all the datasets so that the data can be saved in the pickle file
    # with associated keys
    fullpos_veloutput_use = fullpos_veloutput.transpose()
    pvectorlist = fullpos_veloutput_use[ : , 0:3]
    vvectorlist = fullpos_veloutput_use[ : , 3:6]
    dataset = {}
    dataset['pvectorlist'] = pvectorlist
    dataset['vvectorlist'] = vvectorlist
    dataset['maxstep'] = maxstep
    dataset['totaltime'] = timend
    dataset['randomseed'] = random_seed
    pickle.dump(dataset, open(f"{run_name}.p", "wb"))

######################################################################################################################

inipos_x = []
inipos_y = []
inipos_z = []
inivel_x = []
inivel_y = [] 
inivel_z = []
initialpositionlist = []
initialvelocitylist = []
initialvelocitymultiplierlist = []
# #get initial positions from old file
# dataset1 = pickle.load(open(datafilename, "rb"))
# xyzpositions = np.array(dataset1['positions'], dtype = object )
# xyzvelocities = np.array(dataset1['velocities'], dtype = object)
# get all of the initial values of positions and velocities
xiniposrange = np.linspace(-halflength, halflength, 5)
xinipos, yinipos  = np.meshgrid(xiniposrange, xiniposrange)
xinipos = np.concatenate(xinipos)
yinipos = np.concatenate(yinipos)
for i in range(totalnumberofparticles):
    initialposition_x = xinipos[i]
    initialposition_y = yinipos[i]
    initialposition_z = 0.5
    initialposition = np.array([initialposition_x, initialposition_y, initialposition_z])
    # initialvelocity = xyzvelocities[i][0]
    initialvelocitymultiplier =np.array([0,0,-1]) #Straightdown Debug # initialvelocity/np.linalg.norm(initialvelocity)
    #
    initialpositionlist.append(initialposition)
    # initialvelocitylist.append(initialvelocity)
    initialvelocitymultiplierlist.append(initialvelocitymultiplier)
    #
    inipos_x.append(initialposition_x)
    inipos_y.append(initialposition_y)
    inipos_z.append(initialposition_z)
    #
    inivel_x.append(initialvelocitymultiplier[0])
    inivel_y.append(initialvelocitymultiplier[1])
    inivel_z.append(initialvelocitymultiplier[2])
# print("length of initialized vals saved: ", len(inipos_x), len(inivel_x))
# print("inipos and vel length", len(initialposition), len(initialvelocity))
# print("inipos, vel, multiplier", initialpositionlist[0], initialvelocitylist[0], initialvelocitymultiplierlist[0])


##############################################################
for radiusvalindex in range(len(radiuslist)):
    zstart = radiuslist[radiusvalindex]
    #initialize the locations and angles of particles at 0th time step
    x0list = []
    y0list = []
    z0list = []
    vx0_multiplier_list = []
    vy0_multiplier_list = []
    vz0_multiplier_list = []
    random_seed = 9999
    # random_seed = 10 # Added a seed for reproducibility
    # random.seed(random_seed)
    # zintersect = 0
    # anglerestrictmin = 0
    # anglerestrictmax = 47 # #in degrees
    # vmin = ( (1.+math.cos(anglerestrictmin*math.pi/(180.)))/2. )
    # vmax = ( (1.+math.cos(anglerestrictmax*math.pi/(180.)))/2. )
    for i in range(totalnumberofparticles):
        x0_i = inipos_x[i]#adding intersect makes the position relative to the positions
        y0_i = inipos_y[i]
        z0_i = inipos_z[i]
        x0list.append(x0_i)
        y0list.append(y0_i) #0, halflength)))
        z0list.append(z0_i)
        #for these x y vals, get angle
        vx0_i = inivel_x[i]
        vy0_i = inivel_y[i]
        vz0_i = inivel_z[i]
        vz0_multiplier_list.append(vz0_i)
        vx0_multiplier_list.append(vx0_i)
        vy0_multiplier_list.append(vy0_i)

    # print("length of initialized vals: ", len(x0list), len(vx0_multiplier_list))
    # print("vel multiplier", vx0_multiplier_list[0], vy0_multiplier_list[0], vz0_multiplier_list[0]) 
    # print("pos", x0list[0], y0list[0], z0list[0])

    for magstrengthRun in magnetSTRENGTH:
        tesla_magstrength = magstrengthRun/1000 #the magnet strength in T
        c_mag_array =magnetarraycreation(tesla_magstrength) # make the magnet array of strength magstrengthRun and set to variable called later to get mag field
        #################################################################################################
        for Energy in Energy_range:
            KE_0 = Energy*kev #translating keV to J
            gamma_0 = 1.+(KE_0/(mass*c*c)) #gamma used in conversion of energy to velocity, not classic relativity gamma
            v0 = c*math.sqrt(1.-1./(gamma_0*gamma_0))
            vtot = v0 #total velocity magnitude needs to match v0 for energy to be correct
            gamma = 1/np.sqrt(1-vtot**2/c**2) #classic definition of relativity gamma

            timend = travlength/v0 # time for proton at vel to travel length travlength
            maxstep = timend/maxsteptimedivider #timend/15 #max time step allowed: timend/10 seems reasonable

            vx0list = []
            vy0list = []
            vz0list = []
            for positionindex in range(totalnumberofparticles):
                # straightdown initial velocity
                vz0 = vtot*vz0_multiplier_list[positionindex]
                vx0 = vtot*vx0_multiplier_list[positionindex]
                vy0 = vtot*vy0_multiplier_list[positionindex]
                vz0list.append(vz0)
                vx0list.append(vx0)
                vy0list.append(vy0)
            
            date = f"{today_date}_{round(magstrengthRun,7)}mT_{round(Energy,7)}keV_"

            if __name__ == '__main__': #runcode that does the multiprocessing
                p = mp.Pool(8)#16 #60 #2 #10
                input = (i for i in itertools.combinations_with_replacement(range(totalnumberofparticles), 1))
                res = p.map(compute_vars, input)
                p.close()
                p.join()