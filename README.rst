Introduction
============

.. image:: https://img.shields.io/discord/327254708534116352.svg
    :target: https://adafru.it/discord
    :alt: Discord


.. image:: https://github.com/furbrain/CircuitPython_mag_cal/workflows/Build%20CI/badge.svg
    :target: https://github.com/furbrain/CircuitPython_mag_cal/actions
    :alt: Build Status


.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style: Black

This library allows you to calibrate magnetometers when used with a 3 axis accelerometer.
It is designed for use with circuitpython and `ulab <https://github.com/v923z/micropython-ulab>`_,
but can be used in any python environment where `numpy <https://numpy.org/>`_ is present.


Dependencies
=============
This driver depends on:

* `Adafruit CircuitPython <https://github.com/adafruit/circuitpython>`_
* `Numpy <https://numpy.org/>`_ *or* `ulab <https://github.com/v923z/micropython-ulab>`_

Installing from PyPI
=====================

On supported GNU/Linux systems like the Raspberry Pi, you can install the driver locally `from
PyPI <https://pypi.org/project/circuitpython-mag-cal/>`_.
To install for current user:

.. code-block:: shell

    pip3 install circuitpython-mag-cal

To install system-wide (this may be required in some cases):

.. code-block:: shell

    sudo pip3 install circuitpython-mag-cal

To install in a virtual environment in your current project:

.. code-block:: shell

    mkdir project-name && cd project-name
    python3 -m venv .venv
    source .env/bin/activate
    pip3 install circuitpython-mag-cal

Installing to a Connected CircuitPython Device with Circup
==========================================================

Make sure that you have ``circup`` installed in your Python environment.
Install it with the following command if necessary:

.. code-block:: shell

    pip3 install circup

With ``circup`` installed and your CircuitPython device connected use the
following command to install:

.. code-block:: shell

    circup install mag_cal

Or the following command to update an existing version:

.. code-block:: shell

    circup update

Usage Example
=============
.. code-block:: python

    from mag_cal.calibration import Calibration
    from mag_cal.utils import read_fixture

    PATH = "../tests/fixtures/cal_data/hj2.json"

    with open(PATH) as f:
        aligned, grav, mag = read_fixture(f.read())

    calib = Calibration()
    calib.fit_ellipsoid(mag, grav)
    calib.fit_to_axis(aligned)
    calib.fit_non_linear_quick(aligned, param_count=5)

    #calib.fit_non_linear(aligned, param_count=3)
    for m, g in zip(mag,grav):
        azimuth, inclination, roll = calib.get_angles(m, g)
        print(f"{azimuth:05.1f}° {inclination:+05.1f}° {roll:+04.0f}°")

Documentation
=============
API documentation for this library can be found on `Read the Docs <https://circuitpython-mag-cal.readthedocs.io/>`_.

For information on building library documentation, please check out
`this guide <https://learn.adafruit.com/creating-and-sharing-a-circuitpython-library/sharing-our-docs-on-readthedocs#sphinx-5-1>`_.

Contributing
============

Contributions are welcome! Please read our `Code of Conduct
<https://github.com/furbrain/CircuitPython_mag_cal/blob/HEAD/CODE_OF_CONDUCT.md>`_
before contributing to help this project stay welcoming.
