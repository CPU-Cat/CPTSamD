import datetime
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

"""
Analysis Script for Single-Energy Particle Files

Overview:
    - Imports and processes single-energy particle files.
    - Extracts relevant data and appends to new arrays.
    - Saves processed data to new pickle files.
    - Generates plots for hit locations and particle trajectories.
    - Outputs summary text files with streamlined information.

Changelog:
    - 2023-08-28: Added bar plot for hit locations by energy.
    - 2024-08-22: Added 25-particle xz/yz plots for detailed trajectory visualization.
    - 2025-01-21: Updated detector to be square and flat, matching focal point and FOV.
    - 2025-02-03: Added streamlined text output and magvallist (magnetic values without "mT"); outputs now saved in dedicated folders.
    - 2025-02-05: 0mT run analysis.
    - 2025-02-09: Analyzed new 0.001mT data; updated PNG save file locations.
    - 2025-02-10: Fixed straight-down trajectory issues; resumed angle checks.
    - 2025-02-11: Resolved central detector artifact; addressed multiple bottomplate hits with no detection.
    - 2025-02-15: Cleaned up code, removed unnecessary plt.close() and comments.
    - 2025-02-18: Expanded analysis range; disabled .eps output for easier file management.
    - 2025-03-06: Implemented hemisphere-initialized code (radius = 0.3m).
    - 2025-05-22: Fixed energy and velocity issues; improved 4-panel plots. (Ensure correct mass
"""
date = datetime.datetime.today().strftime("%Y-%m-%d")

# Particle type and shade configuration

# PARTICLE TYPE: proton, electron, or antiproton
particletype = "electron"

# SHADE CONFIGURATION: rotated or original position
shadeisrotated = False

# SPECIAL ID: special string ID to put into newly made folder names to delineate from other similar
# runs
specialid = f"{particletype}_90deg{shadeisrotated}_electronstraightdown"

# PFILE BASE NAME: base name of the pfile, should be the same for all pfiles in the folder
pfilenamebase = "C:/Users/ckpawu/Documents/000_SCC_downloads/2025_06_21_gatheredpfiles_electron1k_straightdown/2"

# FOLDER NAME: folder name for saving the analysis results
folder_name = f"C:/Users/ckpawu/Documents/000_SCC_downloads/{date}_singlepfileanalysis_{specialid}/"

# NUMBER OF PARTICLES TESTING: manually add, don't have way to automate this to the max
numberofparticlestesting = 1000

# ALL ENERGY PFILES: gather all pfiles in the folder with the base name
alltheenergypfiles = glob.glob(pfilenamebase + "*")

# Check if directory exists, if not, create it
check_folder = os.path.isdir(folder_name)
if not check_folder:  # If folder doesn't exist, then create it.
    os.makedirs(folder_name)

# Set the mass of the particle based on the type
if particletype == "proton":
    mass = 1.67262192 * 10**-27  # kg mass of proton
elif particletype == "antiproton":
    mass = 1.67262192 * 10**-27  # kg mass of antiproton
else:
    mass = 9.109 * 10**-31  # kg mass of electron

print("number of files available: ", len(alltheenergypfiles))

# use len(alltheenergypfiles) for all files, use less for sample
numberoftypestesting = int(len(alltheenergypfiles))
print("All the energy pfiles: ", alltheenergypfiles[:numberoftypestesting])

if numberofparticlestesting < 25:
    # number of trajectories that will be plotted as example trajectories, if checking more than 25
    # trajectories, only plot 25, otherwise, plot as many as checking. mostly for the case of
    # testing large # of particles
    exampletrajnumber = numberofparticlestesting
else:
    exampletrajnumber = 25


# Constants and parameters for the simulation
halflength = 5.5120 / 2 * 2.54 * 1 / 100  # in to cm to m, half of wall/MPO side of LEXI
zbodyheight = 37.5 / 100.0  # cm to m of focal length #15.847*2.54/100. #inch to cm to m
# cm to m, detector is half of length of MPO array = 3 x 4cm lenses, with 1mm between each
detectorabsolutelength = ((4 + 0.1 + 4 + 0.1 + 4) / 100.0) / 2
detectorsidelength = detectorabsolutelength / 2  # since is on either side of 0
# changed lengths and body height to ideal radii/2 distance etc
alpha_val = 0.6  # value for alpha in most plots
alpha_val2 = 0.95  # alpha_val*5 # value for need see better, less transparent
alpha_val3 = 0.2  # value for too many points or trajectories in plot, need more transparent
finerr = 0.0381 / 100  # 0.0381 cm thickness fins

# rotation matrix for 90 deg
rotquat = Rotation.from_rotvec(-np.pi / 2 * np.array([0, 0, 1]))
rotmat = rotquat.as_matrix()


def rotated(xyzarray):
    """
    Rotate the input xyz array by 90 degrees around the z-axis.

    Parameters:
    -----------
    xyzarray : np.ndarray
        An array of shape (n, 3) representing the coordinates to be rotated.

    Returns:
    --------
    np.ndarray
        The rotated array of coordinates.
    """
    rotatedxyzarray = xyzarray.dot(rotmat)
    return rotatedxyzarray


def findplanepointandnormal(point1, point2, point3):
    """
    Takes in 3 points from edges of fins and returns the plane point and normal vector.

    Parameters:
    -----------
    point1 : np.ndarray
        First point in the plane.
    point2 : np.ndarray
        Second point in the plane.
    point3 : np.ndarray
        Third point in the plane.

    Returns:
    --------
    tuple
        A tuple containing the plane point and the normal vector.
    """
    thisplanepoint = point1
    p12 = point2 - point1  # vector between point 1 and 2
    p13 = point3 - point1  # vector between point 1 and 3
    nonnormalplanenormal = np.cross(p12, p13)
    thisplanenormal = nonnormalplanenormal / np.linalg.norm(nonnormalplanenormal)
    return (thisplanepoint, thisplanenormal)


# face width of fins/shades, divided by 2 cuz spans 0 #in convert to cm
yedge = 2.54 * 6.6141 / 2
ESz = 14.0  # earthshade height, in cm
SSz = 27.0  # sunshade height, in cm

# use find plane point and normal to take in points from the fins and find the fin plane point and
# normal
cmfin_xpoints = np.array(
    [
        [-7.637, -7.637, -7.637],
        [-4.75, -5.61, -5.60],
        [-3.43, -4.0, -4.0],
        [-2.04, -2.32, -2.33],
        [-0.66, -0.735, -0.74],
        [0.67, 0.74, 0.74],
        [2.05, 2.25, 2.24],
        [3.44, 3.72, 3.74],
        [4.76, 5.04, 5.08],
        [7.93, 7.93, 7.93],
    ]
)
cmfin_ypoints = np.array(
    [
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
        [yedge, yedge, 0],
    ]
)
cmfin_zpoints = np.array(
    [
        [0, ESz, ESz],
        [0.384, 5.48, 5.631],
        [0.438, 4.939, 5.094],
        [0.474, 4.372, 4.529],
        [0.493, 3.803, 3.963],
        [0.493, 3.571, 3.732],
        [0.475, 3.251, 3.414],
        [0.437, 2.873, 3.036],
        [0.384, 2.210, 2.377],
        [0, SSz, SSz],
    ]
)

# convert the fins to cm
cmfin_zpoints[1:-1] = cmfin_zpoints[1:-1] * 2.54
cm_finpointslist = np.stack((cmfin_xpoints, cmfin_ypoints, cmfin_zpoints), axis=2)
# first point is bottom edge, second point is high left, third point is high middle
# earthshade, fins 1, 2, 3, 4, 5, 6, 7, 8, sunshade
finpointslist = cm_finpointslist / 100  # convert cm to m
# first point is bottom left/edge, second point is high left/edge, third point is high middle
shadelimitlow_list = finpointslist[:, 0]
shadelimithighcorner_list = finpointslist[:, 1]
shadelimithighmiddle_list = finpointslist[:, 2]
finzpoints = finpointslist[:, :, 2]
shadelimitlowestz = np.min(finzpoints)  # lowest shade lowest point
# highest shade highest point (top of sunshade)
shadelimithighestz = np.max(finzpoints)
sideofshadelist = (
    np.array(
        [
            [[-8, 8.4, 0], [-7, 8.4, ESz], [8.2, 8.4, SSz]],
            [[-8, -8.4, 0], [-7, -8.4, ESz], [8.2, -8.4, SSz]],
        ]
    )
    / 100
)

sideplane_point_list = []
sideplane_normal_list = []
for sideindex in range(len(sideofshadelist)):
    sidepoint3 = sideofshadelist[sideindex]
    sidepoint, sidenormal = findplanepointandnormal(sidepoint3[0], sidepoint3[1], sidepoint3[2])
    sideplane_point_list.append(sidepoint)
    sideplane_normal_list.append(sidenormal)
numberofsides = len(sideplane_point_list)
sideplane_point_list = np.array(sideplane_point_list)
sideplane_normal_list = np.array(sideplane_normal_list)

plane_point_list = []
plane_normal_list = []
for finpointindex in range(len(finpointslist)):
    finpoint3 = finpointslist[finpointindex]
    planepoint, planenormal = findplanepointandnormal(finpoint3[0], finpoint3[1], finpoint3[2])
    plane_point_list.append(planepoint)
    plane_normal_list.append(planenormal)
numberofshades = len(plane_point_list)
plane_point_list = np.array(plane_point_list)
plane_normal_list = np.array(plane_normal_list)


def findtrajplaneintersection(xyzsingletraj, plane_point, plane_normal):
    """
    Find the intersection of a trajectory with a plane defined by a point and a normal vector.

    Parameters:
    -----------
    xyzsingletraj : np.ndarray
        The trajectory of the particle.
    plane_point : np.ndarray
        A point on the plane.
    plane_normal : np.ndarray
        The normal vector of the plane.

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - shadewashit (bool): Whether the trajectory intersects the plane.
        - intersectpositionxyz (np.ndarray): The position of the intersection.
        - tparameter (float): The parameter of the intersection.
        - numberofintersections (int): The number of intersections.
        - otherhitlocations (list): A list of other intersection locations.
    """

    shadewashit = False  # initialize return outputs in case no shade was hit
    intersectpositionxyz = None
    tparameter = None
    numberofintersections = 0
    otherhitlocations = []

    trajectory = np.array(xyzsingletraj)
    trajectorydirections = np.diff(trajectory, axis=0)
    # vector from plane point to traj points
    plane_trajvector = trajectory[:-1] - plane_point
    # dot products vectors between traj and plane's norm
    planenormal_directions = np.dot(plane_normal, trajectorydirections.T)
    planenormal_trajectory = np.dot(plane_normal, plane_trajvector.T)
    # if dot close to 0 then it was parallel to plane
    maskoutparallels = np.abs(planenormal_directions) < 0.0000000000000000001
    tparamvals = np.divide(-planenormal_trajectory, planenormal_directions, where=~maskoutparallels)
    # 6/4 edit for trashing parallels
    tparamvals = np.array([np.nan if x is None else x for x in tparamvals])
    # masks for intersection within the traj path
    maskouttvalid = (tparamvals >= 0) & (tparamvals <= 1)
    numberofintersections = np.sum(maskouttvalid)

    if numberofintersections > 0:
        indexofvalidt = np.argmax(maskouttvalid)
        tparameter = tparamvals[indexofvalidt]
        intersectpositionxyz = (
            trajectory[indexofvalidt] + tparameter * trajectorydirections[indexofvalidt]
        )
        shadewashit = True
        if numberofintersections > 1:
            # if we have more than one intersection, then look for the other locations and return
            # in the other hitlocations output
            allhitindices = np.where(maskouttvalid)
            # start at 1 so we don't get the first index that was already gotten with First
            # intersection point
            for indexofindices in range(1, len(allhitindices[0])):
                indicesofvalidtOTHER = allhitindices[0][indexofindices]
                othertparameter = tparamvals[indicesofvalidtOTHER]
                otherhitposxyz = (
                    trajectory[indicesofvalidtOTHER]
                    + othertparameter * trajectorydirections[indicesofvalidtOTHER]
                )
                otherhitlocations.append(otherhitposxyz)

    return (shadewashit, intersectpositionxyz, tparameter, numberofintersections, otherhitlocations)


def findshadeintersection(xyzsingletraj):
    """
    Find the intersection of a trajectory with the shade planes and side planes.

    Parameters:
    -----------
    xyzsingletraj : np.ndarray
        The trajectory of the particle.

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - shadewashit (bool): Whether the trajectory intersects any shade.
        - intersectpositionxyz (np.ndarray): The position of the intersection.
        - tparameter (float): The parameter of the intersection.
        - numberofshadehits (int): The number of shade hits.
        - otherhitlocations (list): A list of other intersection locations.
    """
    trajectory = np.array(xyzsingletraj)
    shadewashit = False  # initialize return outputs in case no shade was hit
    intersectpositionxyz = None
    tparameter = None
    numberofshadehits = 0
    otherhitlocations = []

    for i_sides in range(numberofsides):
        # only if didn't already hit something
        if not shadewashit:
            sidehit_i, intersectpos_i, tparam_i, numberofintersections_i, otherhitlocations_i = (
                findtrajplaneintersection(
                    trajectory, sideplane_point_list[i_sides], sideplane_normal_list[i_sides]
                )
            )  # check if hit plane
            # is a list, must edit into array
            otherhitlocations_i = np.array(otherhitlocations_i)
            # for the plane intersection, check if was inside the plane's shade side limits
            if (
                sidehit_i
            ):  # if do hit a plane, make sure to change return parameters to details of hit
                if numberofintersections_i > 1:
                    allintersectpoints = np.concatenate(
                        (intersectpos_i, otherhitlocations_i.flatten())
                    )
                else:
                    allintersectpoints = intersectpos_i
                # for each point that intersect, check if within max and min shade
                for intersectindex in range(numberofintersections_i):
                    # check if within y limit, z limit, and x limit of this plane/shade
                    # the xyz point checking
                    interposcheck_i = allintersectpoints[
                        3 * intersectindex : (3 * intersectindex + 3)
                    ]
                    # check x, y, and then z has equation
                    # should give set of 3 points
                    sidepoints_i = sideofshadelist[i_sides]
                    # get the lowest of the x vals for this shade
                    sidex_low = np.min(sidepoints_i[:, 0]) - finerr
                    # get the highest of the x vals for this shade
                    sidex_high = np.max(sidepoints_i[:, 0]) + finerr
                    if (interposcheck_i[0] < sidex_high) and (interposcheck_i[0] > sidex_low):
                        # since pass x check, now get y limits and check y limits
                        # get the lowest of the x vals for this shade
                        sidey_low = np.min(sidepoints_i[:, 1]) - finerr
                        # get the highest of the x vals for this shade
                        sidey_high = np.max(sidepoints_i[:, 1]) + finerr
                        if (interposcheck_i[1] < sidey_high) and (interposcheck_i[1] > sidey_low):
                            # should always pass this if got this far because of the y vals being the same for the 3 points making up the plane
                            # calculate z(x) for use as the top limit of z
                            if interposcheck_i[0] < sidepoints_i[:, 0][1]:
                                # earthshade if in earthshade shelf region
                                sidez_at_x = sidepoints_i[:, 2][1]
                            else:
                                sidez_at_x = (sidepoints_i[:, 2][2] - sidepoints_i[:, 2][1]) / (
                                    sidepoints_i[:, 0][2] - sidepoints_i[:, 0][1]
                                ) * (interposcheck_i[0] - sidepoints_i[:, 0][1]) + sidepoints_i[
                                    :, 2
                                ][
                                    1
                                ]
                            sidez_high = sidez_at_x + finerr
                            sidez_low = sidepoints_i[:, 2][0]
                            if (interposcheck_i[2] < sidez_high) and (
                                interposcheck_i[2] > sidez_low
                            ):
                                if numberofshadehits == 0:
                                    # if haven't hit shade yet, log as intersect
                                    intersectpositionxyz = interposcheck_i
                                    tparameter = tparam_i
                                else:
                                    # if have hit shade already, log as other hit
                                    otherhitlocations.append(interposcheck_i)
                                shadewashit = True
                                numberofshadehits = numberofshadehits + 1

    # iterate through each shade individually
    for i_planes in range(numberofshades):
        if not shadewashit:
            shadehit_i, intersectpos_i, tparam_i, numberofintersections_i, otherhitlocations_i = (
                findtrajplaneintersection(
                    trajectory, plane_point_list[i_planes], plane_normal_list[i_planes]
                )
            )  # check if hit plane
            # is a list, must edit into array
            otherhitlocations_i = np.array(otherhitlocations_i)
            # for the plane intersection, check if was inside the plane's shade limits
            if (
                shadehit_i
            ):  # if do hit a plane, make sure to change return parameters to details of hit
                if numberofintersections_i > 1:
                    allintersectpoints = np.concatenate(
                        (intersectpos_i, otherhitlocations_i.flatten())
                    )
                else:
                    allintersectpoints = intersectpos_i
                # for each point that intersect, check if within max and min shade
                for intersectindex in range(numberofintersections_i):
                    # check if within y limit, z limit, and x limit of this plane/shade
                    # the xyz point checking
                    interposcheck_i = allintersectpoints[
                        3 * intersectindex : (3 * intersectindex + 3)
                    ]
                    planeylimit = abs(shadelimitlow_list[i_planes][1]) + finerr

                    # check y Note this WONT work for the side walls, only the major earthshade, sunshade, and fins
                    if abs(interposcheck_i[1]) < planeylimit:
                        planezlimhigh = shadelimithighmiddle_list[i_planes][2] + finerr
                        planezlimlow = shadelimitlow_list[i_planes][2] - finerr
                        # print("limits: z low, zhigh",  planezlimlow, planezlimhigh )
                        # check z
                        if (interposcheck_i[2] > planezlimlow) and (
                            interposcheck_i[2] < planezlimhigh
                        ):
                            planexlimupper = np.max(finpointslist[i_planes][:, 0]) + finerr
                            planexlimlower = np.min(finpointslist[i_planes][:, 0]) - finerr
                            # print("limits: x low, x high",  planexlimlower , planexlimupper  )
                            if (interposcheck_i[0] > planexlimlower) and (
                                interposcheck_i[0] < planexlimupper
                            ):
                                # this is final check, not doing z curves top and bottom because of tiny portion it affects
                                # z as function of interposcheck_i[1] based on radii
                                if numberofshadehits == 0:
                                    intersectpositionxyz = interposcheck_i
                                    tparameter = tparam_i
                                else:
                                    otherhitlocations.append(interposcheck_i)
                                shadewashit = True
                                numberofshadehits = numberofshadehits + 1

    return (shadewashit, intersectpositionxyz, tparameter, numberofshadehits, otherhitlocations)


energylist = []
energyvallist = []
magfieldlist = []
mcphitratelist = []
energyfailurelist = []

randomseed_list = []

MCPhitlist = []
bodybaffleshitlist = []
shadehitlist = []
numberoffileslist = []
nonfailed_particle_list = []
duplicate_particle_list = []  # added 1/9
magfieldvallist = []

# Constants for the simulation
error = 0.5  # percent error allowed, so 0.05 = 100+-0.05 error allowed
c = 2.99e8  # speed of light m/s
kev = 1.0e3 * 1.6e-19  # convert kev to joules
# mcp_diameter = 80.0/1000.0 #80mm in m
# mcp_radius   = mcp_diameter/2.
# top of range for finding hits that went to bottom + MCP
zbodylim = -zbodyheight * 0.75
LEXIside = 36  # mm length of side of one lens - shelf = 40-2-2
LEXIsuppt = 5  # mm shelf +between lens = 2+2+1
magnetdimx = 3.175  # mag dim = 12.7 x 6.35 x 3.175 mm^3
magnetdimy = 12.7 * 3  # three mag stacked
mdimy = 12.7  # y mag dim for 1 magnet
magnetdimz = 6.35
magholes = 0.27 * 2.54 / 100.0  # inch to cm to m
MAGholder = 0.375 * 2.54 / 100.0  # inch to cm to m
MPOholder = 0.138 * 2.54 / 100.0  # inch to cm to m
lowheight = 0.0 - (MAGholder - 0.5 * magholes)  # bottom side MAGholder
# lowest possible top side of MPO holder
highheight = 0.0 + (MPOholder + 0.5 * magholes)
a1 = 0.5 * LEXIside / 1000  # used as determining near side of first magholder bar
a2 = 0.5 * LEXIside / 1000 + LEXIsuppt / 1000  # far side of first magholder bar
b1 = 1.5 * LEXIside / 1000 + LEXIsuppt / 1000  # inner magholder far edge
# mag+mpo holder edges for 3d plotting
x_min, x_max = -1 * b1, b1
y_min, y_max = -1 * b1, b1
z_min, z_max = lowheight, highheight
vertices = np.array(
    [
        [
            x_min,
            y_min,
            z_min,
        ],  # vertices that will call with magarrayedges, actual values for edge of mag and mpo holders
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]
)
magarrayedges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],  # bottom rectangle indices for plotting 3d rectangle prism
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],  # top rectangle indices
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]  # vertical edges indices


hitposz = -1 * zbodyheight  # for finding bottomplate hit
hitposz1 = 0  # for finding magnet holder hit

# file_indexes = np.array(range(3))
# int(len(alltheenergypfiles))):#-1*file_indexes:#len(alltheenergypfiles)):
for index_of_pfiles in range(numberoftypestesting):

    datafilename = alltheenergypfiles[index_of_pfiles]
    print("name of file: ", datafilename)
    dataset1 = pickle.load(open(datafilename, "rb"))

    filename_split = datafilename.split("_")
    print("filename split: ", filename_split)
    fileenergy = filename_split[-2]
    print("file energy: ", fileenergy)
    filemagfield = filename_split[-1][:-2]
    # fileenergy = filename_split[-2]
    # print("file energy: ", fileenergy)
    energykeV_value = float(fileenergy[:-3])
    energyvallist.append(energykeV_value)
    # filemagfield = filename_split[-1][:-2]
    magfieldlist.append(filemagfield)
    print("mag field and val: ", filemagfield)
    magfieldvallist.append(float(filemagfield[:-2]))
    print("mag field and val: ", filemagfield[:-2])
    totalnumberofparticles = len(dataset1["positions"])

    randomseedlong = dataset1["randomseed"]
    randomseed = randomseedlong[0]
    randomseed_list.append(randomseed)
    print("randseed: ", randomseed)
    date_name_basepng = (
        f"{folder_name}{date}_{specialid}_{fileenergy}_{filemagfield}_randseed_{randomseed}"
    )
    print("base of png file name: ", date_name_basepng)

    xyzpositions = np.array(dataset1["positions"], dtype=object)
    xyzvelocities = np.array(dataset1["velocities"], dtype=object)

    single_energy_lists = {"xlists": [], "ylists": [], "zlists": []}
    mcphits = 0  # for this energy, initialize number of hits
    failed_particles = 0
    nonfailed_particles = 0
    bodybafflehits = 0
    shadesturnaround = 0

    hitlocations_x = []  # location hit detector plane
    hitlocations_y = []
    mcphitlocationx = []  # location hit detector itself
    mcphitlocationy = []
    # hit magholder/mpoholder (lowest point of top of mpoholder, not includes spherical curve) or turnaround I think
    cadhitx = []
    cadhity = []
    shadehitlocationx = []  # locations hit shades
    shadehitlocationy = []
    shadehitlocationz = []

    # DEBUG ###############################################################
    mcphittrajx = []  # saving the trajectories of the ones that hit MCP
    mcphittrajy = []
    mcphittrajz = []
    mcphittrajxyz = []
    nothittrajxyz = []  # MOVED
    nonfailed_trajxyz = []  # trajectories that haven't failed energy check
    nonfailed_trajx = []
    nonfailed_trajy = []
    nonfailed_trajz = []
    ##########################################################################

    # initialize position checking list -- collect the initial positions of particles and check
    # against the previous
    inipos_list = [[100, 100, 100], [200, 200, 200]]

    # len(xyzpositions)): #go through each individ trajectory
    for trajectory_index in range(numberofparticlestesting):
        # velxyz1 = xyzvelocities[trajectory_index] #for use in trajectory-only testing
        # posxyz1 = xyzpositions[trajectory_index]
        velxyz0 = xyzvelocities[trajectory_index]
        posxyz0 = xyzpositions[trajectory_index]
        if shadeisrotated:
            velxyz1 = rotated(velxyz0)  # rotate 90 deg
            posxyz1 = rotated(posxyz0)
        else:
            velxyz1 = velxyz0
            posxyz1 = posxyz0
        xpositionlist_i = posxyz1[:, 0]
        ypositionlist_i = posxyz1[:, 1]
        zpositionlist_i = posxyz1[:, 2]
        xvelocitylist_i = velxyz1[:, 0]
        yvelocitylist_i = velxyz1[:, 1]
        zvelocitylist_i = velxyz1[:, 2]
        # xpositionlist_i = xyzpositions[trajectory_index][:,0]
        # ypositionlist_i = xyzpositions[trajectory_index][:,1]
        # zpositionlist_i = xyzpositions[trajectory_index][:,2]
        # xvelocitylist_i = xyzvelocities[trajectory_index][:,0]
        # yvelocitylist_i = xyzvelocities[trajectory_index][:,1]
        # zvelocitylist_i = xyzvelocities[trajectory_index][:,2]
        # #with the x, y, z lists, find the hits and whatnot

        energy_failed = False  # initialize trajectory as not failed
        mcp_is_hit = False  # initialize trajectory as not hit mcp
        mcpHIT = False
        MPOholderhit = False
        shadehit = False
        cadhit = False
        turnaround = False

        inipos_array = np.array(inipos_list)
        if posxyz1[0][0] in inipos_array[:, 0]:
            if posxyz1[0][1] in inipos_array[:, 1]:
                duplicate_particle_list.append(trajectory_index)

        inipos_list.append(posxyz1[0])

        # get initial velocity magnitude in order to get initial energy
        vtot0 = np.linalg.norm(velxyz1[0])
        gamma0 = 1 / np.sqrt(1 - vtot0**2 / c**2)
        # UPDATE 1/9 to relativistic!!! 0.5*mass*np.linalg.norm(velxyz1[0])**2 #kinetic energy from first step 1/2mv^2
        energystart = (gamma0 - 1) * mass * c**2
        hitposx = 2  # initialize the hit positions to make it out of bounds if particle not get to -zbodyheight
        hitposy = 2  # to make it out of bounds

        shadehit, shadehitpos, shadehitparameter, numberofintersections, otherhitlocations = (
            findshadeintersection(posxyz1)
        )
        # print("shade intersection output: ", shadehit, shadehitpos, shadehitparameter, numberofintersections, otherhitlocations)
        if shadehit:
            shadehitlocationx.append(shadehitpos[0])
            shadehitlocationy.append(shadehitpos[1])
            shadehitlocationz.append(shadehitpos[2])
            # print("number of intersections: ", numberofintersections, "\n")
            if numberofintersections > 1:
                # since already have one intersection accounted for, number of otherhitlocations is numberofintersections-1
                for iother in range(numberofintersections - 1):
                    # print("otherhitlocations", otherhitlocations[0])
                    # print("other hit locaitions: ", otherhitlocations, otherhitlocations[iother], iother)
                    otherhitlocationsjustarray = otherhitlocations[iother]
                    shadehitlocationx.append(otherhitlocationsjustarray[0])
                    shadehitlocationy.append(otherhitlocationsjustarray[1])
                    shadehitlocationz.append(otherhitlocationsjustarray[2])
                    # ADD otherhitlocationsto the shade hit positions

        # for each step in one file (one trajectory)
        for indexi in range(len(velxyz1)):
            vtot = np.linalg.norm(velxyz1[indexi])
            if vtot**2 > c**2:
                energy_failed = True
            else:  # changed debug -- only calc rest if vtot hasn't surpassed c
                gamma = 1 / np.sqrt(1 - vtot**2 / c**2)
                # UPDATE 1/9!!!! 0.5*mass*np.linalg.norm(velxyz1[indexi])**2  #kinetic energy from
                # last step
                energyend = (gamma - 1) * mass * c**2
                energyerrorratio = abs(energyend - energystart) / energystart * 100.0
                if energyerrorratio > error:
                    energy_failed = True
                else:  # CHANGED DEBUG ##### #only check rest of things if energy hasn't failed
                    if indexi > 0:  # check for hitting magholder
                        # since may need to check the interpolated position, cannot use the last indices
                        # if the location of position z is within magholder or mpo holder, or ...
                        # OR if if this position is before (above) MPO holder and next position is after (below) magholder
                        znow = posxyz1[indexi][2]
                        znext = posxyz1[indexi - 1][2]
                        xnow = posxyz1[indexi][0]
                        ynow = posxyz1[indexi][1]
                        if znow > lowheight and znow < highheight:
                            # if it's within height of magholder+MPOholder
                            # WITHIN BARS:
                            if (
                                (a1 < abs(xnow) < a2)
                                or (a1 < abs(ynow) < a2)
                                or (abs(xnow) > b1)
                                or (abs(ynow) > b1)
                            ):
                                MPOholderhit = True
                        elif (znow > highheight and znext < lowheight) or (
                            znext > highheight and znow < lowheight
                        ):
                            # if don't have point within mcp z width, calc point with z = 0 to see if x and y hit bars
                            zratio1 = (hitposz1 - znext) / (znext - znow)
                            xnext = posxyz1[indexi - 1][0]
                            ynext = posxyz1[indexi - 1][1]
                            xest = xnext + zratio1 * (xnext - xnow)
                            yest = ynext + zratio1 * (ynext - ynow)
                            # WITHIN BARS:
                            if (
                                (a1 < abs(xest) < a2)
                                or (a1 < abs(yest) < a2)
                                or (abs(xest) > b1)
                                or (abs(yest) > b1)
                            ):
                                MPOholderhit = True

                        # check for MCP plane hit: using less than because it's negative, so less
                        # is lower
                        if znow < (hitposz / 2):
                            # get the mcphits, don't need info on if hit cad or walls or turned
                            # around
                            # old "currentpos", nextpos is now xnow ynow znow
                            zlast = posxyz1[indexi - 1, 2]
                            xlast = posxyz1[indexi - 1, 0]
                            ylast = posxyz1[indexi - 1, 1]
                            if znow == hitposz:
                                hitposx = xnow
                                hitposy = ynow
                                hitlocations_x.append(hitposx)
                                hitlocations_y.append(hitposy)
                            # last point was above and current point is below
                            elif (zlast > hitposz) and (znow < hitposz):
                                zratio = (hitposz - zlast) / (znow - zlast)
                                hitposx = xlast + zratio * (xnow - xlast)
                                hitposy = ylast + zratio * (ynow - ylast)
                                hitlocations_x.append(hitposx)
                                hitlocations_y.append(hitposy)

        # DETECTOR HIT CHANGE TO SQUARE!!!  #check the bottomplate hit is within the MCP x and y
        # hitposy**2) < mcp_radius**2:
        if ((-1 * detectorsidelength) < hitposx < detectorsidelength) and (
            (-1 * detectorsidelength) < hitposy < detectorsidelength
        ):
            mcpHIT = True

        if posxyz1[-1][2] > 0:
            turnaround = True
            # check if the last z point is positive which is above the mag holder

        if mcpHIT:
            mcp_is_hit = True
        if MPOholderhit or shadehit:
            mcp_is_hit = False
            cadhit = True

        ####################################
        if energy_failed:
            failed_particles = failed_particles + 1
        else:
            nonfailed_particles = nonfailed_particles + 1
            nonfailed_trajxyz.append(posxyz1)
            nonfailed_trajx.append(xpositionlist_i)
            nonfailed_trajy.append(ypositionlist_i)
            nonfailed_trajz.append(zpositionlist_i)

            if mcp_is_hit:
                mcphits = mcphits + 1
                mcphitlocationx.append(hitposx)
                mcphitlocationy.append(hitposy)
                mcphittrajxyz.append(posxyz1)
                mcphittrajx.append(xpositionlist_i)
                mcphittrajy.append(ypositionlist_i)
                mcphittrajz.append(zpositionlist_i)
            elif cadhit:
                cadhitx.append(hitposx)
                cadhity.append(hitposy)
                nothittrajxyz.append(posxyz1)
                if (
                    shadehit
                ):  # log as shadeturn around if CAD hit is shade, otherwise go to body baffle hit
                    shadesturnaround = shadesturnaround + 1
                else:
                    bodybafflehits = bodybafflehits + 1
            elif turnaround:
                shadesturnaround = shadesturnaround + 1
                nothittrajxyz.append(posxyz1)
            else:
                bodybafflehits = bodybafflehits + 1
                nothittrajxyz.append(posxyz1)

    if len(xyzpositions) == 0:
        hitrate = 0.999
    else:
        hitrate = mcphits / nonfailed_particles

    energylist.append(fileenergy)
    mcphitratelist.append(hitrate)
    energyfailurelist.append(failed_particles)
    MCPhitlist.append(mcphits)
    numberoffileslist.append(totalnumberofparticles)
    nonfailed_particle_list.append(nonfailed_particles)
    bodybaffleshitlist.append(bodybafflehits)
    shadehitlist.append(shadesturnaround)
    print("adds up to 10k?: ", shadesturnaround + bodybafflehits + mcphits + failed_particles)
    print("length of initial positions +2: ", len(inipos_list))
    print(
        "energy, mag strength, nonfailed, hitnumber, hitrate: ",
        fileenergy,
        filemagfield,
        nonfailed_particles,
        mcphits,
        hitrate,
    )
    print("number of hit trajectories recorded: ", len(mcphittrajx))
    print(
        "number of not-hit traj and sum hit not-hit: ",
        len(nothittrajxyz),
        len(nothittrajxyz) + len(mcphittrajx),
    )
    print(
        "number of non-nrg-fail + non-failed traject recorded: ",
        nonfailed_particles,
        len(nonfailed_trajx),
    )

    if len(xyzpositions) != 0:
        # make 3d plot with trajectories and intersection points with shades
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(shadehitlocationx, shadehitlocationy, shadehitlocationz, alpha=alpha_val3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        for edge in magarrayedges:
            x_edge = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y_edge = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z_edge = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, color="cyan")
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {nonfailed_particles} nonfailed particles {hitrate} hitrate 3D"
        )
        plt.savefig(f"{date_name_basepng}_3dparticlehits.eps", dpi=300, format="eps")
        plt.savefig(f"{date_name_basepng}_3dparticlehits.png", dpi=300)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(inipos_array[2:, 0], inipos_array[2:, 1], inipos_array[2:, 2], alpha=alpha_val3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        for edge in magarrayedges:
            x_edge = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y_edge = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z_edge = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, color="cyan")
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {nonfailed_particles} nonfailed particles {hitrate} hitrate 3D"
        )
        # plt.savefig(f"{date_name_basepng}_3dparticlehits.eps", dpi=300, format='eps')
        plt.savefig(f"{date_name_basepng}_3d_inipos.png", dpi=300)
        # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(inipos_array[2:, 0], inipos_array[2:, 1], alpha=alpha_val3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {nonfailed_particles} nonfailed particles {hitrate} hitrate 3D"
        )
        # plt.savefig(f"{date_name_basepng}_3dparticlehits.eps", dpi=300, format='eps')
        plt.savefig(f"{date_name_basepng}_inipos_xy.png", dpi=300)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(inipos_array[2:, 0], inipos_array[2:, 2], alpha=alpha_val3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {nonfailed_particles} nonfailed particles {hitrate} hitrate 3D"
        )
        # plt.savefig(f"{date_name_basepng}_3dparticlehits.eps", dpi = 300, format='eps')
        plt.savefig(f"{date_name_basepng}_inipos_xz.png", dpi=300)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(inipos_array[2:, 1], inipos_array[2:, 2], alpha=alpha_val3)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {nonfailed_particles} nonfailed particles {hitrate} hitrate 3D"
        )
        # plt.savefig(f"{date_name_basepng}_3dparticlehits.eps", dpi = 300, format='eps')
        plt.savefig(f"{date_name_basepng}_inipos_yz.png", dpi=300)

        #################

        fig, ax = plt.subplots()
        plt.plot(hitlocations_x, hitlocations_y, "bx", alpha=0.3)
        plt.plot(mcphitlocationx, mcphitlocationy, "co", alpha=0.3)
        plt.plot(cadhitx, cadhity, "go", alpha=0.7)
        mcpsquare = plt.Rectangle(
            (-detectorsidelength, -detectorsidelength),
            detectorabsolutelength,
            detectorabsolutelength,
            color="r",
            fill=False,
        )
        ax.add_artist(mcpsquare)
        ax.set_xlim([-1.5 * b1, 1.5 * b1])
        ax.set_ylim([-1.5 * b1, 1.5 * b1])
        ax.set_aspect("equal")
        plt.title(f"{fileenergy} {filemagfield} {particletype} plot {hitrate} hr x y")
        plt.savefig(
            f"{date_name_basepng}_bottomplatehits_highlightedhitsandcadhit_ZOOMED.png", dpi=300
        )
        plt.savefig(
            f"{date_name_basepng}_bottomplatehits_highlightedhitsandcadhit_ZOOMED.eps",
            dpi=300,
            format="eps",
        )

        fig, ax = plt.subplots()
        plt.plot(hitlocations_x, hitlocations_y, "bx", alpha=0.3)
        plt.plot(mcphitlocationx, mcphitlocationy, "co", alpha=0.3)
        mcpsquare = plt.Rectangle(
            (-detectorsidelength, -detectorsidelength),
            detectorabsolutelength,
            detectorabsolutelength,
            color="r",
            fill=False,
        )
        ax.add_artist(mcpsquare)
        ax.set_aspect("equal")
        plt.title(f"{fileenergy} {filemagfield} {particletype} plot {hitrate} hr x y")
        plt.savefig(f"{date_name_basepng}_bottomplatehits_highlightedhits.png", dpi=300)
        plt.savefig(
            f"{date_name_basepng}_bottomplatehits_highlightedhits.eps", dpi=300, format="eps"
        )

        fig, ax = plt.subplots()
        plt.plot(hitlocations_x, hitlocations_y, "bx", alpha=0.3)
        plt.plot(mcphitlocationx, mcphitlocationy, "co", alpha=0.3)
        mcpsquare = plt.Rectangle(
            (-detectorsidelength, -detectorsidelength),
            detectorabsolutelength,
            detectorabsolutelength,
            color="r",
            fill=False,
        )
        ax.add_artist(mcpsquare)
        ax.set_aspect("equal")
        ax.set_xlim([-1.5 * b1, 1.5 * b1])
        ax.set_ylim([-1.5 * b1, 1.5 * b1])
        plt.title(f"{fileenergy} {filemagfield} {particletype} plot {hitrate} hr x y")
        plt.savefig(f"{date_name_basepng}_bottomplatehits_highlightedhits_ZOOMED.png", dpi=300)
        plt.savefig(
            f"{date_name_basepng}_bottomplatehits_highlightedhits_ZOOMED.eps", dpi=300, format="eps"
        )

        # all after this point have dark bkg
        plt.style.use("dark_background")
        ############################

        fig = plt.figure()
        ax = plt.axes()
        for indie in range(len(nonfailed_trajx)):
            # ax.plot(single_energy_lists['xlists'][indie],single_energy_lists['zlists'][indie], linewidth = 0.15, alpha = alpha_val, color = 'm')
            ax.plot(
                nonfailed_trajx[indie],
                nonfailed_trajz[indie],
                linewidth=0.15,
                alpha=alpha_val,
                color="m",
            )
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        LEXIbody = plt.Rectangle(
            (-halflength, -zbodyheight), 2.0 * halflength, zbodyheight, color="y", fill=False
        )
        ax.add_artist(LEXIbody)
        ax.set_xlabel("X pos (m)")
        ax.set_ylabel("Z pos (m)")
        ax.axis("equal")
        ax.set(xlim=(-1.4 * halflength, 1.4 * halflength), ylim=(-1.3 * zbodyheight, 0.35))
        plt.title(f"{fileenergy} {filemagfield} {particletype} plot {hitrate} hr x z")
        plt.savefig(f"{date_name_basepng}_x_z.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_x_z.eps", dpi=300, format="eps")

        fig = plt.figure()
        ax = plt.axes()
        for indie in range(len(nonfailed_trajy)):
            ax.plot(
                nonfailed_trajy[indie],
                nonfailed_trajz[indie],
                linewidth=0.15,
                alpha=alpha_val,
                color="m",
            )
        # ax.plot(single_energy_lists['ylists'][start: start+particlesshowing],single_energy_lists['zlists'][start: start+particlesshowing], linewidth = 0.15, alpha = alpha_val, color = 'm')
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        LEXIbody = plt.Rectangle(
            (-halflength, -zbodyheight), 2.0 * halflength, zbodyheight, color="y", fill=False
        )
        ax.add_artist(LEXIbody)
        ax.set_xlabel("Y pos (m)")
        ax.set_ylabel("Z pos (m)")
        ax.axis("equal")
        ax.set(xlim=(-1.4 * halflength, 1.4 * halflength), ylim=(-1.3 * zbodyheight, 0.35))
        plt.title(f"{fileenergy} {filemagfield} {particletype} plot {hitrate} hr y z")
        plt.savefig(f"{date_name_basepng}_y_z.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_y_z.eps", dpi=300, format="eps")

        # detector hit trajectories specifically
        fig = plt.figure()
        ax = plt.axes()
        for indie in range(len(mcphittrajz)):
            ax.plot(
                mcphittrajx[indie], mcphittrajz[indie], linewidth=0.25, alpha=alpha_val, color="m"
            )
        # ax.plot(single_energy_lists['ylists'][start: start+particlesshowing],single_energy_lists['zlists'][start: start+particlesshowing], linewidth = 0.15, alpha = alpha_val, color = 'm')
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        LEXIbody = plt.Rectangle(
            (-halflength, -zbodyheight), 2.0 * halflength, zbodyheight, color="y", fill=False
        )
        ax.add_artist(LEXIbody)
        ax.set_xlabel("X pos (m)")
        ax.set_ylabel("Z pos (m)")
        ax.axis("equal")
        ax.set(xlim=(-1.4 * halflength, 1.4 * halflength), ylim=(-1.3 * zbodyheight, 0.55))
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} {len(mcphittrajz)} particles hit detector"
        )
        plt.savefig(f"{date_name_basepng}_x_z_dethitonly.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_x_z_dethitonly.eps", dpi=300, format="eps")

        fig = plt.figure()
        ax = plt.axes()
        for indie in range(len(mcphittrajz)):
            ax.plot(
                mcphittrajy[indie], mcphittrajz[indie], linewidth=0.25, alpha=alpha_val, color="m"
            )
        # ax.plot(single_energy_lists['ylists'][start: start+particlesshowing],single_energy_lists['zlists'][start: start+particlesshowing], linewidth = 0.15, alpha = alpha_val, color = 'm')
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        LEXIbody = plt.Rectangle(
            (-halflength, -zbodyheight), 2.0 * halflength, zbodyheight, color="y", fill=False
        )
        ax.add_artist(LEXIbody)
        ax.set_xlabel("Y pos (m)")
        ax.set_ylabel("Z pos (m)")
        ax.axis("equal")
        ax.set(xlim=(-1.4 * halflength, 1.4 * halflength), ylim=(-2.3 * zbodyheight, 0.55))
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} {len(mcphittrajz)} particles hit detector"
        )
        plt.savefig(f"{date_name_basepng}_y_z_dethitonly.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_y_z_dethitonly.eps", dpi=300, format="eps")

        # runs hit and not hit side by side x z and y z
        fig, axis = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for indie in range(len(nonfailed_trajx)):
            axis[0].plot(
                nonfailed_trajx[indie],
                nonfailed_trajz[indie],
                linewidth=0.15,
                alpha=alpha_val,
                color="c",
            )
            axis[2].plot(
                nonfailed_trajy[indie],
                nonfailed_trajz[indie],
                linewidth=0.15,
                alpha=alpha_val,
                color="c",
            )
        for indie in range(len(mcphittrajz)):
            axis[1].plot(
                mcphittrajx[indie], mcphittrajz[indie], linewidth=0.15, alpha=alpha_val2, color="m"
            )
            axis[3].plot(
                mcphittrajy[indie], mcphittrajz[indie], linewidth=0.15, alpha=alpha_val2, color="m"
            )
        axis[0].set_ylabel("Z (m)")
        axis[0].set_xlabel("X (m)")
        axis[1].set_xlabel("X (m)")
        axis[2].set_xlabel("Y (m)")
        axis[3].set_xlabel("Y (m)")
        for axisindex in range(4):
            LEXIbody = plt.Rectangle(
                (-halflength, -zbodyheight),
                2.0 * halflength,
                zbodyheight,
                color="y",
                fill=False,
                zorder=2,
            )
            Sqdetector = plt.Rectangle(
                (-detectorsidelength, -zbodyheight),
                detectorabsolutelength,
                0.001,
                color="r",
                fill=True,
                zorder=3,
            )  # detectorsidelength
            axis[axisindex].set(
                xlim=(-2 * halflength, 2 * halflength), ylim=(-1.3 * zbodyheight, 0.5)
            )
            axis[axisindex].add_artist(LEXIbody)
            axis[axisindex].add_artist(Sqdetector)
        fig.suptitle(
            f"{fileenergy} {filemagfield} {particletype} {len(mcphittrajz)} particles hit detector All and mcphit"
        )
        plt.savefig(f"{date_name_basepng}_4colzxy.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_4colzxy.eps", dpi=300, format="eps")

        fig, axis = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for indie in range(len(nothittrajxyz)):
            axis[0].plot(
                nothittrajxyz[indie][:, 0],
                nothittrajxyz[indie][:, 2],
                linewidth=0.15,
                alpha=alpha_val,
                color="c",
            )
            axis[2].plot(
                nothittrajxyz[indie][:, 1],
                nothittrajxyz[indie][:, 2],
                linewidth=0.15,
                alpha=alpha_val,
                color="c",
            )
        for indie in range(len(mcphittrajz)):
            axis[1].plot(
                mcphittrajx[indie], mcphittrajz[indie], linewidth=0.15, alpha=alpha_val2, color="m"
            )
            axis[3].plot(
                mcphittrajy[indie], mcphittrajz[indie], linewidth=0.15, alpha=alpha_val2, color="m"
            )
        axis[0].set_ylabel("Z (m)")
        axis[0].set_xlabel("X (m)")
        axis[1].set_xlabel("X (m)")
        axis[2].set_xlabel("Y (m)")
        axis[3].set_xlabel("Y (m)")
        for axisindex in range(4):
            LEXIbody = plt.Rectangle(
                (-halflength, -zbodyheight),
                2.0 * halflength,
                zbodyheight,
                color="y",
                fill=False,
                zorder=2,
            )
            Sqdetector = plt.Rectangle(
                (-detectorsidelength, -zbodyheight),
                detectorabsolutelength,
                0.001,
                color="r",
                fill=True,
                zorder=3,
            )  # detectorsidelength
            axis[axisindex].set(
                xlim=(-2 * halflength, 2 * halflength), ylim=(-1.3 * zbodyheight, 0.5)
            )
            axis[axisindex].add_artist(LEXIbody)
            axis[axisindex].add_artist(Sqdetector)
        fig.suptitle(
            f"{fileenergy} {filemagfield} {particletype} {len(mcphittrajz)} particles hit detector Nothit and mcphit"
        )
        plt.savefig(f"{date_name_basepng}_4colzxy_nothitandhit.png", dpi=300)

        # sampled runs
        fig = plt.figure()
        ax = plt.axes()
        for indie in range(exampletrajnumber):
            ax.plot(
                nonfailed_trajx[indie],
                nonfailed_trajz[indie],
                linewidth=0.80,
                alpha=alpha_val2,
                color="m",
            )
        ax.plot(shadehitlocationx, shadehitlocationz, "c.", alpha=alpha_val3)
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        LEXIbody = plt.Rectangle(
            (-halflength, -zbodyheight), 2.0 * halflength, zbodyheight, color="y", fill=False
        )
        ax.add_artist(LEXIbody)
        ax.set_xlabel("X pos (m)")
        ax.set_ylabel("Z pos (m)")
        ax.axis("equal")
        ax.set(xlim=(-1.4 * halflength, 1.4 * halflength), ylim=(-1.3 * zbodyheight, 0.35))
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {exampletrajnumber} particles x z"
        )
        plt.savefig(f"{date_name_basepng}_x_z_25.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_x_z_25.eps", dpi=300, format="eps")

        fig = plt.figure()
        ax = plt.axes()
        for indie in range(exampletrajnumber):
            ax.plot(
                nonfailed_trajy[indie],
                nonfailed_trajz[indie],
                linewidth=0.80,
                alpha=alpha_val2,
                color="m",
            )
        ax.plot(shadehitlocationy, shadehitlocationz, "c.", alpha=alpha_val3)
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        LEXIbody = plt.Rectangle(
            (-halflength, -zbodyheight), 2.0 * halflength, zbodyheight, color="y", fill=False
        )
        ax.add_artist(LEXIbody)
        ax.set_xlabel("Y pos (m)")
        ax.set_ylabel("Z pos (m)")
        ax.axis("equal")
        ax.set(xlim=(-1.4 * halflength, 1.4 * halflength), ylim=(-1.3 * zbodyheight, 0.35))
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {exampletrajnumber} particles y z"
        )
        plt.savefig(f"{date_name_basepng}_y_z_25.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_y_z_25.eps", dpi=300, format="eps")

        fig = plt.figure()
        ax = plt.axes()
        for indie in range(exampletrajnumber):
            ax.plot(
                nonfailed_trajx[indie],
                nonfailed_trajy[indie],
                linewidth=0.80,
                alpha=alpha_val2,
                color="m",
            )
        ax.plot(shadehitlocationx, shadehitlocationy, "c.", alpha_val3)
        # ax.plot(x1_points, z1_points, 'b+-', alpha = 0.5)
        ax.set_xlabel("X pos (m)")
        ax.set_ylabel("Y pos (m)")
        ax.axis("equal")
        # ax.set(xlim = (-1.4*halflength, 1.4*halflength), ylim = (-1.3*zbodyheight, 0.35))
        plt.title(
            f"{fileenergy} {filemagfield} {particletype} plot {exampletrajnumber} particles x y"
        )
        plt.savefig(f"{date_name_basepng}_x_y_shadehitloc.png", dpi=300)
        plt.savefig(f"{date_name_basepng}_x_y_shadehitloc.eps", dpi=300, format="eps")
        plt.close("all")

    plt.close("all")

# save text files with info of energy, energy messed up, hits to mcp, hit rate (mcp hit/ energy
# conserved)
TXTdate_name_base = f"{folder_name}{date}_{specialid}_hitrate_info.txt"
file_maxdiff = open(f"{TXTdate_name_base}", "a+")
L = [
    "\n energy from name\n",
    str(energylist),
    "\n",
    str(energyvallist),
    "\n magnet field base \n",
    str(magfieldlist),
    "\n",
    str(magfieldvallist),
    " \n mcp hit rate \n",
    str(mcphitratelist),
    " \n energy failure \n",
    str(energyfailurelist),
    "\n MCP hit \n",
    str(MCPhitlist),
    "\n total number files \n",
    str(numberoffileslist),
    "\n body/baffles hit \n",
    str(bodybaffleshitlist),
    "\n turned around/shades hit \n",
    str(shadehitlist),
    "\n notfailed particles \n",
    str(nonfailed_particle_list),
    "\n duplicate particles \n",
    str(duplicate_particle_list),
    "\n randomseed \n",
    str(randomseed_list),
]

file_maxdiff.writelines(L)
file_maxdiff.close()

TXTdate_name_base = f"{folder_name}{date}_{specialid}_info_BASIC.txt"
file_maxdiff = open(f"{TXTdate_name_base}", "a+")
L = [
    "\n energy \n",
    str(energyvallist),
    "\n magnet field base \n",
    str(magfieldvallist),
    " \n mcp hit rate \n",
    str(mcphitratelist),
    "\n MCP hit total \n",
    str(MCPhitlist),
    " \n energy failure total \n",
    str(energyfailurelist),
    "\n number of notfailed particles \n",
    str(nonfailed_particle_list),
    "\n total number of files \n",
    str(numberoffileslist),
]

file_maxdiff.writelines(L)
file_maxdiff.close()
