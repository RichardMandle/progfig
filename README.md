# progfig
_progfig_ is a GUI tool using PyQT for generating images that depict liquid crystalline ordering and/or phase types. It uses the mayavi visualisation engine for 3D rendering and, through clever choice of parameters, enables the user to generate nice images that represent a variety of phase types.

To load it, simply:
~~~
python main.py
~~~

##How it Works##
_progfig_ starts by generating some initial coordinates in 3D space based on the user specified spacings and randomness parameters. We add vectors begining at each of these coordinates, the orientation of which is controlled so that the final order parameter of the vector space is close to the user requested value of <P2>.<br><br>

The user can also add _tilt_ in the X and Y directions, as well as _splay_. The idea of _blocks_ is that periodic structures can be generated. Combining all these options allows for complex phase types to be generated programatically.<br><br>

You can use a large number of different colourmaps. You can colour according to position in X/Y/Z, rotation with respect to the different axes, individual order paramters (<P1>, <P2>) of the vectors. If you use "cylinder" mode, you can colour each vector individually with a gradient of the chosen colourmap along its length. <br><br>

You can select from a few different drawing styles:<br>
###quiver### - A nice 3D arrow; useful for showing polar or directional things.<br>
###fast cylinder### - a quick to draw cylinder with end-caps; looks nice, but isn't quite as flexible as...<br>
###cylinder### - slow to draw, but good resolution control and support for colourmap-along-vector gradients which are really nice for polar/directional ordering.<br>
###boomerang### - a V-shaped thing; useful for representing dimers or bent-cores. You can control the bend angle (!) with the _Boomerang Angle_ parameter. The _biaxiality()_ parameter controls how much the "short" axis of the boomerang is aligned. If we term this biaxial director _B_, then we must consider that for non-polar phases +B == -B, whereas for polar +B != -B:<br>

| ![image](https://github.com/user-attachments/assets/e3d2f948-772b-4b05-ad47-70db32d66b96) | ![image](https://github.com/user-attachments/assets/08976723-3e3c-4f4d-9603-385570f7b573)   |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
|A _non polar nematic_ phase formed of boomerangs with an angle of 60 and _biaxiality_ of 0 |A _polar nematic_ phase formed of boomerangs with an angle of 60 and _biaxiality_ of 1    |


Many default phase types are provided, for example, nematic with polar = True gives this:
![image](https://github.com/RichardMandle/progfig/assets/101199234/6b341003-857c-42b8-b377-8785fa9c6044)

<br>You can also play around with "blocks", enabling modulated phase types such as the SmA_AF phase from https://arxiv.org/abs/2402.07305:
![image](https://github.com/RichardMandle/progfig/assets/101199234/872577e0-8fd8-49a6-aff3-0321509cdddd)

<br> Clever choice of parameters allow heliconical structures to be generated, for example an N_TB phase composed of rods (fast cylinder) with a cylic colourmap (HSV): 
![image](https://github.com/user-attachments/assets/a6a9b6b9-bf12-4a5b-9dee-96b0c9498344)

<br> Or even things that don't (yet) exist; This TGBA_F type structure:
![image](https://github.com/RichardMandle/progfig/assets/101199234/4c8fbe41-63d0-4d8f-bb8d-e3f0aa7dd192)

# Known issues
* Some variables pertaining to visual quality are hardcoded in visualisation.py; this will be addressed in future once the code is less flimsy
* Layer planes ignore block orientation, so this produces visually unsatisfying results.
* The "cylinder" method is quite slow, but it does allow colouring by gradient along the cylinder length. Its best to preview with "fast cylinder" and once happy, switch to "cylinder".

# Notes
an earlier version of _progfig.py_ is in this repo; this uses matplotlib as the visualiser and is surpassed by this version.
