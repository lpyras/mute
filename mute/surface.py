#########################
#########################
###                   ###
###  MUTE             ###
###  William Woodley  ###
###  15 July 2022     ###
###                   ###
#########################
#########################

# Import packages

import os

import numpy as np
import scipy.integrate as scii
from tqdm import tqdm
import mute.constants as constants

# Calculate surface fluxes


def calc_s_fluxes(
    primary_model="GSF",
    interaction_model="SIBYLL-2.3c",
    atmosphere="CORSIKA",
    location="USStd",
    month=None,
    output=None,
    file_name="",
    force=False,
    test=False,
    year=None,
    day=None,
):

    """
    Calculate surface fluxes in units of [(cm^2 s sr MeV)^-1] for default surface energy grid and surface flux zenith angles.

    The default surface energy grid is given by constants.ENERGIES, and the default zenith angles are given by constants.ANGLES_FOR_S_FLUXES.

    Parameters
    ----------
    primary_model : str in {"GSF", "HG", "GH", "ZS", "ZSP", "PL27"} or tuple, optional (default: "GSF")
        The primary flux model to use in MCEq. Options:
        GSF  = GlobalSplineFitBeta
        HG   = HillasGaisser2012 (H3a)
        GH   = GaisserHonda
        ZS   = Zatsepin-Sokolskaya (Default)
        ZSP  = Zatsepin-Sokolskaya (PAMELA)
        PL27 = SimplePowerlaw27
        Alternatively, this can be set with a tuple. For example: (pm.GaisserStanevTilav, "3-gen").

    interaction_model : str, optional (default: "SIBYLL-2.3c")
        The hadronic interaction model to use in MCEq. See the Tutorial or MCEq documentation for a list of options.

    atmosphere : {"CORSIKA", "MSIS00"}, optional (default: "CORSIKA")
        The atmospheric model to use in MCEq. For US Standard Atmosphere, use "CORSIKA". For seasonal variations, use "MSIS00".

    location : str, optional (default: "USStd")
        The name of the location for which to calculate the surface fluxes. See the Tutorial or MCEq documentation for a list of options.

    month : str, optional (default: None)
        The month for which to calculate the surface fluxes. For US Standard Atmosphere, use None. For seasonal variations, use the month name.

    output : bool, optional (default: taken from constants.get_output())
        If True, an output file will be created to store the results.

    file_name : str, optional (default: constructed from input parameters)
        The name of the file in which to store the results. If output is False or None, this is ignored and the file is named automatically Surface_Fluxes_{location}_{month}_{interaction_model}_{pm_sname}_{year}_{day}.txt
                ),

    force : bool, optional (default: False)
        If True, force the creation of new directories if required.

    test: bool, optional (default: False)
        For use in the file test_s_fluxes.py. If True, this will calculate surface fluxes for only three angles.

    Returns
    -------
    s_fluxes : NumPy ndarray
        A two-dimensional array containing the surface fluxes. The shape will be (n_energies, n_surface_angles), and the fluxes will be in units of [(cm^2 s sr MeV)^-1].
    """

    # Import packages

    from MCEq.core import MCEqRun
    import mceq_config
    if location == 'SouthPole':
        mceq_config.h_obs = 2835.

    import crflux.models as pm

    # Check values

    constants._check_constants(force=force)

    if output is None:
        output = constants.get_output()

    assert atmosphere in [
        "CORSIKA",
        "MSIS00",
        "AIRS",
    ], 'atmosphere must be set to either "CORSIKA" or "MSIS00".'

    primary_models = {
        "GSF": (pm.GlobalSplineFitBeta, None),
        "HG": (pm.HillasGaisser2012, "H3a"),
        "GH": (pm.GaisserHonda, None),
        "ZS": (pm.ZatsepinSokolskaya, "default"),
        "ZSP": (pm.ZatsepinSokolskaya, "pamela"),
        "PL27": (pm.SimplePowerlaw27, None),
    }

    if isinstance(primary_model, str):

        assert (
            primary_model in primary_models
        ), "Set primary model not available. See the available options in the Tutorial at {0}.".format(
            "https://github.com/wjwoodley/mute/blob/main/docs/Tutorial.md#changing-the-primary-model"
        )

        primary_model_for_MCEq = primary_models[primary_model]
        pm_sname = primary_model

    elif isinstance(primary_model, tuple) and len(primary_model) == 2:

        primary_model_for_MCEq = primary_model
        pm_sname = primary_model[0](primary_model[1]).sname

    else:

        raise TypeError(
            "Primary model not set correctly. For an explanation, see the Tutorial at {0}.".format(
                "https://github.com/wjwoodley/mute/blob/main/docs/Tutorial.md#changing-the-primary-model"
            )
        )

    # Set the angles for use in MCEq

    angles = constants.ANGLES_FOR_S_FLUXES

    if test:

        angles = [0, 30, 60]

    # Set MCEq up

    mceq_config.enable_default_tracking = False

    if constants.get_verbose() > 1:

        print(
            "Calculating surface fluxes for {0} using {1} and {2} {3} {4}.".format(
                location, pm_sname, interaction_model, year, day
            )
        )

        mceq_run = MCEqRun(
            interaction_model=interaction_model,
            primary_model=primary_model_for_MCEq,
            theta_deg=0.0,
            **mceq_config.config,
        )
        mceq_run.set_density_model((atmosphere, (location, month)))
        if atmosphere == 'AIRS' and year is not None and day is not None:
            mceq_run.density_model.set_date(year, day)
            print('Setting date to {0} {1}'.format(year, day))

    else:

        from contextlib import redirect_stdout

        with open(os.devnull, "w") as suppress, redirect_stdout(suppress):

            mceq_run = MCEqRun(
                interaction_model=interaction_model,
                primary_model=primary_model_for_MCEq,
                theta_deg=0.0,
                **mceq_config.config,
            )
            mceq_run.set_density_model((atmosphere, (location, month)))
            if atmosphere == 'AIRS' and year is not None and day is not None:
                mceq_run.density_model.set_date(year, day)
                print('Setting date to {0} {1}'.format(year, day))



    # Calculate the surface fluxes
    # Zeroth index = Surface energy
    # First index  = Zenith angle

    s_fluxes = np.zeros((len(constants.ENERGIES), len(angles)))

    # Run MCEq
    # Convert the surface fluxes from default [GeV] to [MeV]

    for j in (
        tqdm(range(len(angles))) if constants.get_verbose() >= 1 else range(len(angles))
    ):

        mceq_run.set_theta_deg(angles[j])
        mceq_run.solve()

        s_fluxes[:, j] = (
            1e-3
            * (
                mceq_run.get_solution("total_mu+", mag=0)
                + mceq_run.get_solution("total_mu-", mag=0)
            )[:-30]
        )

    if constants.get_verbose() > 1:
        print("Finished calculating surface fluxes.")

    # Write the results to the file

    if output:

        constants._check_directory(
            os.path.join(constants.get_directory(), "surface"), force=force
        )

        if file_name == "" or not isinstance(file_name, str):

            file_name = os.path.join(
                constants.get_directory(),
                "surface",
                "Surface_Fluxes_{0}_{1}_{2}_{3}_{4}_{5}.txt".format(
                    location, str(month), interaction_model, pm_sname, year, day
                ),
            )

        file_out = open(file_name, "w")

        for i in range(len(constants.ENERGIES)):

            for j in range(len(angles)):

                file_out.write(
                    "{0:1.14f} {1:1.5f} {2:1.14e}\n".format(
                        constants.ENERGIES[i],
                        angles[j],
                        s_fluxes[i, j],
                    )
                )

        file_out.close()

        if constants.get_verbose() > 1:
            print(f"Surface fluxes written to {file_name}.")

    return s_fluxes


# Get surface fluxes


def load_s_fluxes_from_file(file_name=""):

    """
    Retrieve a surface fluxes matrix from a file in units of [(cm^2 s sr MeV)^-1] stored in data/surface based on the input parameters. The file is usually generated by calc_s_fluxes().

   
    Parameters
    ----------
    file_name : str
        The name of the file from which to retrieve the surface fluxes. Run calc_s_fluxes() to generate this file.

    Returns
    -------
    s_fluxes : NumPy ndarray
        A two-dimensional array containing the surface fluxes. The shape will be (n_energies, n_surface_angles), and the fluxes will be in units of [(cm^2 s sr MeV)^-1].
    """

    if os.path.exists(file_name):
        if constants.get_verbose() > 1:
            print(
                "Loading file {0} to retrieve surface fluxes.".format(file_name)
                )

        # Check that the file has the correct numbers of energies and zenith angles
        file = open(file_name, "r")
        n_lines = len(file.read().splitlines())
        file.close()

        # If so, read the surface fluxes in from it
        if n_lines == len(constants.ENERGIES) * len(constants.ANGLES_FOR_S_FLUXES):

            s_fluxes = np.reshape(
                np.loadtxt(file_name)[:, 2], (len(constants.ENERGIES), len(constants.ANGLES_FOR_S_FLUXES))
            )

            if constants.get_verbose() > 1:
                print("Loaded surface fluxes.")
            return s_fluxes

        else:
            print("The file {0} does not contain the correct number of energies and zenith angles.".format(file_name))
    else:
        print('The file {0} does not exist.'.format(file_name))



def calc_s_intensities(
    s_fluxes=None,
    output_file_name="",
    force=False,
):

    """
    Calculate surface intensities in units of [(cm^2 s sr)^-1] for default surface energy grid and surface flux zenith angles.

    The default surface energy grid is given by constants.ENERGIES, and the default zenith angles are given by constants.ANGLES_FOR_S_FLUXES.

    Parameters
    ----------
    s_fluxes : NumPy ndarray, optional (default: taken from surface.load_s_fluxes_from_file())
        A surface flux matrix of shape (n_energies, n_surface_angles).

    store_output : bool, optional (default: True)

    output_file_name : 
        The name of the file in which to store the results. If None the file will not be written.

    Returns
    -------
    s_intensities : NumPy ndarray
        A one-dimensional array containing the surface intensities. The length will be n_surface_angles, and the intensities will be in units of [(cm^2 s sr)^-1].
    """

    # Check values
    constants._check_constants()

    # Get the surface flux matrix

    if constants.get_verbose() > 1:
        print("Calculating surface intensities.")

    # Check that the surface flux matrix has been loaded properly
    if s_fluxes is None:
        raise Exception(
            "Surface intensities not calculated. The surface flux matrix was not provided or loaded correctly."
        )
    s_fluxes = np.atleast_2d(s_fluxes)

    # Calculate the surface intensities
    s_intensities = [
        scii.simpson(s_fluxes[:, j], constants.ENERGIES)
        for j in range(len(constants.ANGLES_FOR_S_FLUXES))
    ]

    if constants.get_verbose() > 1:
        print("Finished calculating surface intensities.")

    # Write the results to the file
    if output_file_name != "":

        constants._check_directory(
            os.path.join(constants.get_directory(), "surface"), force=force)

        if not isinstance(output_file_name, str):
            if constants._get_verbose() > 1:
                print("No valid output file name provided. Writing to default file. surface/Surface_Intensities.txt")

            output_file_name = os.path.join(constants.get_directory(),"surface","Surface_Intensities.txt")

        file_out = open(output_file_name, "w")

        for j in range(len(constants.ANGLES_FOR_S_FLUXES)):

            file_out.write(
                "{0:1.5f} {1:1.14e}\n".format(
                    constants.ANGLES_FOR_S_FLUXES[j], s_intensities[j]
                )
            )

        file_out.close()

        if constants.get_verbose() > 1:
            print(f"Surface intensities written to {output_file_name}.")

    return s_intensities


# Calculate total surface fluxes


def calc_s_tot_flux(
    s_fluxes=None,
    force=False,
):

    """
    Calculate a total surface flux in units of [(cm^2 s)^-1] for default surface energy grid and surface flux zenith angles.

    The default surface energy grid is given by constants.ENERGIES, and the default zenith angles are given by constants.ANGLES_FOR_S_FLUXES.

    Parameters
    ----------
    s_fluxes : NumPy ndarray, optional (default: taken from surface.load_s_fluxes_from_file())
        A surface flux matrix of shape (n_energies, n_surface_angles).

    force : bool, optional (default: False)
        If True, force the calculation of new arrays or matrices and the creation of new directories if required.

    Returns
    -------
    s_tot_flux : float
        The total surface flux in units of [(cm^2 s)^-1].
    """

    # Check values

    constants._check_constants()

    # Calculate the surface intensities

    if constants.get_verbose() > 1:
        print("Calculating total surface flux.")

    s_intensities = calc_s_intensities(
        s_fluxes,
        output_file_name="",
        force=force,
    )

    if s_intensities is None:

        raise Exception(
            "Total surface flux not calculated. The surface intensities were not calculated properly."
        )

    # Calculate the total surface flux
    # Because constants.ANGLES_FOR_S_FLUXES goes (0..89), cos(angles) goes (1..0)
    # cos(constants.ANGLES_FOR_S_FLUXES) is decreasing, but scii.simpson() wants an increasing array
    # Therefore, integrate backwards, using [::-1] on the integrand and steps
    # Otherwise, the answer will be negative

    s_tot_flux = (
        2
        * np.pi
        * scii.simpson(
            s_intensities[::-1], np.cos(np.radians(constants.ANGLES_FOR_S_FLUXES[::-1]))
        )
    )

    if constants.get_verbose() > 1:
        print("Finished calculating total surface flux.")

    return s_tot_flux
