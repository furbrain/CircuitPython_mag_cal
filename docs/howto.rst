**********************************
How to build an electronic compass
**********************************

Author: Phil Underwood

This document discusses the considerations needed for building an electronic compass.
I have spent many years designing a device known as the *Shetland Attack Pony* [#a]_

This device is designed for use in cave surveying - it needs to be precise (ideally to within 0.5° for both
azimuth and inclination), easy to use and rugged. It needs to also be able to measure distance.

Theory
======
There are two sensors needed, a magnetometer and an accelerometer. A magnetometer measures the direction and strength
of the earth's magnetic field, whereas an accelerometer measures how fast the device is accelerating, and also the
direction and strength of earth's gravity. (Gravity can be simply considered as a form of acceleration.) We will be
discussing sensors that have three perpendicular axes so we can fully measure the gravitational and magnetic fields.

The earth's magnetic field is fairly weak, typically only 25-60µT [#b]_ (a strong fridge magnet is about 10000µT).
It is also usually inclined - thsi can vary from parallel to the ground to very steeply inclined near the magnetic
poles. In northern europe it is usually inclined by about 60-70°.

This means that simply measuring the magnetic field vector is insufficient if the device is tilted or rolled. This can
be corrected using an accelerometer to measure the tilt and roll of the device.

Coordinate Systems
------------------
We will need to define two coordinate systems to use, *device* and *world* coordinates

World coordinates
   X is due east, Y is due North, Z is Up

Device coordinates
  Y is the primary axis - the direction of
  travel of direction or direction of a pointer. Z is up, and X is to the right if Y is
  facing away from you and Z is up.


Ideal algorithm
---------------
Let ``M`` and ``G`` be the magnetic field vectors respectively, in device coordinates.

.. math::
    E & = M \times -G \\
    N & = -G \times E \\
    R & = (\hat{E} \space \hat{N} \space -\hat{G})

Where ``E`` is East, ``N`` is North and ``R`` is an orthonormal rotation matrix that converts device coordinates
to world coordinates. From here it is simple to obtain the direction of the Y-axis in world coordinates, and from there
use simple trigonometry to calculate the azimuth and inclination.

Sources of Error
================

If our measurements of ``M`` and ``G`` are correct, then the above calculation is sufficient. However there are multiple
potential sources of error present

Sensor Errors
-------------
All sensors have some element of error. They will usually have at least an offset error (so reading 0nT as 30nT)  and
some scale errors (reading 100nT as 120nT, and 200nT as 240nT). There can also be some non-linearity, where for example
the difference between 0 and 100nT would be read as 100nT but the difference between 400 and 500nT would be read as
90nT.

There can also be mechanical errors as the sensors may not be exactly at 90° to each other or properly aligned to their
enclosing package. The sensors may also not be "pure" to their own axis and respond to changing magnetic fields in a
different axis.

Some sensors can also have significant noise - i.e. variation in readings when there is no change in the external
magnetic field.

Accelerometers also have these intrinsic errors, but they tend to be better behaved than magnetometers.

Device Magnetic Errors
----------------------

Hard Iron
^^^^^^^^^
*Hard iron* refers to the presence of any permanent magnets within the device. These will add a fixed offset to the
sensor readings. We can visualise this as follows. As the magnetic field is constant, then the sensor reading should
exist on the surface of a sphere of radius ``|M|``. Hard iron effects will change the centre of this sphere, but
not otherwise affect it.

Soft Iron
^^^^^^^^^
*Soft iron* refers to the presence of ferromagnetic material within the device. These respond to external magnetic
fields by generating their own magnetic field, which produces a variable effect on the magnetic field of the device.
This can be visualised as the sphere of possible sensor readings becomes an ovoid (like an american football), which
may have it's long axis in any direction.

Electromagnetic Effects
^^^^^^^^^^^^^^^^^^^^^^^
Any electrical current will also generate a magnetic field which is proportional both to the current and the inverse
of the distance from the current. This will act in the same way as hard iron distortion, but if the current is variable
then the error will also vary.

External Magnetic Errors
------------------------
If readings are taken near sources of hard or soft iron distortion then this will affect the compass readings - because
the external magnetic field *has changed*. This is not possible to calibrate, as it will depend on the precise location
of the material causing the distortion, relative to the device which will vary during use. It may be possible to detect
this as we should be able to see if the magnetic field strength or inclination has changed

Device Gravitational Errors
---------------------------
These are very unlikely unless your device enclosure is *exceptionally* dense.

External Gravitational Errors
-----------------------------
There is some minor variation in the value of *g* over the surface of the earth, but these are unlikely to be
significant enough to affect your measurements.

Mechanical Errors
-----------------
The sensor package may not be mounted precisely on the PCB, and the PCB may not be mounted precisely within
the enclosure. The pointer (if there is one) may also not be mounted correctly. If you have a separate
accelerometer and magnetometer, then these may also not be mounted parallel to each other. These errors can
all be combined together as a single rotational transformation for each sensor.

Design Considerations
=====================

There are a few design considerations that can help reduce the errors described above.

PCB Considerations
------------------
Avoid running high current power lines near the magnetometer. If you use a ground plane, make a window in this near the
magnetometer so you do not get significant current running underneath your sensor. You may want to consider turning
off high current peripherals (e.g. display or laser) during measurement.

Enclosure Considerations
------------------------
It is worth trying to avoid using any ferromagnetic material in the construction of the device. This will reduce
the impact of hard and soft iron errors. Most enclosures come with steel fixings, but it is generally possible to
replace these with brass fixings. You may want to consider using a LiPo battery - normal alkaline batteries usually
have significant hard and/or soft iron effects. If you have any wires carrying significant current within the device,
ensure that they do not move with respect to the sensors - using quite stiff wire can help here.

It is important that your pointer device is held rigidly with respect to the magnetometer and accelerometer, and that
the PCB is supported such that it does not flex or move if held in various angles. Make sure that the battery is held
stable with respect to the PCB.

So long as the sensors and pointer device are held rigidly with respect to each other, you don't need to worry too much
about ensuring that they are lined up correctly - there's going to be some rotational error with the sensor placement
on the PCB, so this will all get calibrated out in any event.

Calibration
===========

Coming soon


.. [#a] `<https://shetlandattackpony.co.uk/>`_.
.. [#b] `<https://www.ngdc.noaa.gov/geomag/faqgeom.shtml#What_are_the_magnetic_elements>`_
