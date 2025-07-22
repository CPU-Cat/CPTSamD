# File: pickle_gather_single_energy_Electrons_LOW.py


---

# File: __init__.py

**Module Docstring:**

CPTSamD: Charged Particle Trajectory Simulation and analysis in a magnetic Diverter field.

Provides tools for simulating and analyzing charged particle motion in complex magnetic geometries.


---

# File: multiedit_RUNCODE_Electrons_25_posset_straightdown_LOWERenergy.py

## Function: magnetarraycreation

Function to create a collection of magnets arranged in a specific pattern.

Parameters
----------
magnet_strength : float
    The strength of the magnets in mT.

Returns
-------
magpy.Collection
    A collection of magnets arranged in the specified pattern.

## Function: function45

The function calculates the Lorentz force acting on a charged particle in a magnetic
field.

Parameters
----------
timestep : float
    The current time step in the simulation.
    NOTE: This parameter is not used in the function but is required by the `solve_ivp` 
    function.
variables_array : array-like
    An array containing the current position and velocity of the particle.

Returns
-------
variables_array_return : array-like
    An array containing the derivatives of the position and velocity of the particle.

## Function: magnetfield

The function calculates the magnetic field at a given position using a predefined collection of
magnets.

Parameters
----------
position : array-like
    A list or array containing the x, y, and z coordinates of the position where the magnetic
    field is to be calculated.

Returns
-------
Bfield : array-like
    An array containing the magnetic field components (Bx, By, Bz) at the specified position.

## Function: compute_vars

The function takes particle number as input and gives a bunch of output and saves them an HDF
and pickle file.

Parameters
----------
args : integer
    Input particle number for computation

Returns
-------
pvectorlist : array of shape steps x 3
    p Vector list

vvectorlist : array of shape steps x 3
    V vector list


---

# File: shadeincl_singlepfile_hitrate_analysis_False90_1kelectron_straightdown_iniposcheck.py

## Function: rotated

Rotate the input xyz array by 90 degrees around the z-axis.

Parameters:
-----------
xyzarray : np.ndarray
    An array of shape (n, 3) representing the coordinates to be rotated.

Returns:
--------
np.ndarray
    The rotated array of coordinates.

## Function: findplanepointandnormal

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

## Function: findtrajplaneintersection

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

## Function: findshadeintersection

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


---
