import os


def test_import_magpylib():
    """Test that magpylib can be imported."""
    import magpylib

    assert hasattr(magpylib, "Collection")


def test_import_numpy():
    """Test that numpy can be imported and basic array works."""
    import numpy as np

    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_import_matplotlib():
    """Test that matplotlib can be imported and a figure can be created."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    assert fig is not None


def test_import_scipy():
    """Test that scipy.integrate.solve_ivp can be imported and used."""
    from scipy.integrate import solve_ivp

    def f(t, y):
        return -0.5 * y

    sol = solve_ivp(f, [0, 1], [1.0])
    assert sol.y.shape[1] > 0


def test_scripts_run():
    """Test that main scripts can be imported as modules (if refactored)."""
    # This will pass if the scripts are refactored into importable modules.
    # For now, just check that the files exist.
    base_dir = os.path.dirname(os.path.dirname(__file__))
    assert os.path.exists(
        os.path.join(
            base_dir,
            "fixedshadeincl_singlepfile_hitrate_analysis_False90_1kelectron_straightdown_iniposcheck.py",
        )
    )
    assert os.path.exists(os.path.join(base_dir, "pickle_gather_single_energy_Electrons_LOW.py"))
    assert os.path.exists(
        os.path.join(base_dir, "multiedit_RUNCODE_Electrons_25_posset_straightdown_LOWERenergy.py")
    )
