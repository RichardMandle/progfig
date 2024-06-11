# progfig
_progfig_ is a GUI tool using PyQT for generating images that depict liquid crystalline ordering and/or phase types. It uses the mayavi visualisation engine for 3D rendering and, through clever choice of parameters, enables the user to generate nice images that represent a variety of phase types.

To load it, simply:
~~~
python main.py
~~~

Now, change settings to your hearts content! Some defaults are provided, for example, nematic with polar = True gives this:
![image](https://github.com/RichardMandle/progfig/assets/101199234/6b341003-857c-42b8-b377-8785fa9c6044)


<br>You can also play around with "blocks", enabling modulated phase types such as the SmA_AF phase from https://arxiv.org/abs/2402.07305:
![image](https://github.com/RichardMandle/progfig/assets/101199234/872577e0-8fd8-49a6-aff3-0321509cdddd)

<br> Or even things that don't (yet) exist; This TGBA_F type structure:
![image](https://github.com/RichardMandle/progfig/assets/101199234/4c8fbe41-63d0-4d8f-bb8d-e3f0aa7dd192)

# Known issues
* The input values of <P2> and the output values are not really meaningful currently, but the user supplied value of <P2> does allow control over orientation although you probably want to set it a lot higher than you might think.
* While testing/fixing, some variables pertaining to visual quality are hardcoded in visualisation.py; this will be addressed in future once the code is less flimsy
* Layer planes ignore block orientation, so this produces visually unsatisfying results.
* Drawing with cylinders is sort of slow; best to preview with a quiver, then when you are happy, you can draw cylinders and go make a cup of tea while you wait.

# Notes
an earlier version of _progfig.py_ is in this repo; this uses matplotlib as the visualiser and is surpassed by this version.
