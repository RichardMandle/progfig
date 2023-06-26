# progfig
_progfig_ is a small tool for generating images that depict the local structure of liquid crystals. These images are generated programatically, and the user can supply arguments that control the local structure.

# How it works
Initially we generate xyz coordinates for a set of points with _generate_regular_points_; this takes two arguments, the first sets the spacing between points and the second sets the extent. We can visualise this with the _plot_points_ function:
~~~
points = generate_regular_points(0.25,0.75)
plot_points(points)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/594888a7-4e46-475a-8397-9e392597df8b)


If we want a hexatic local packing then we use the _hexatic_offset_ function, and visualise with _plot_points_:

~~~
points = hexatic_offset(points)
plot_points(points)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/76f3b33d-beb6-43ba-b8fd-e48a4ab2564c)

Next, we can add some random displacement _via_ the _add_randomness_ function. We can specify the random displacement in x,y, and z seperately. 
~~~
points = add_randomness(points,randomness_x=0.25,randomness_y=0.00,randomness_z=0.00)
points = add_randomness(points,randomness_x=0.25,randomness_y=0.25,randomness_z=0.00)
points = add_randomness(points,randomness_x=0.25,randomness_y=0.25,randomness_z=0.50)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/e4951eed-4f6a-4604-b1f8-312e33d8b5bc)
![image](https://github.com/RichardMandle/progfig/assets/101199234/54041d39-d7fa-4ef0-8a1d-99ada0db4cbe)
![image](https://github.com/RichardMandle/progfig/assets/101199234/418514d1-316a-4878-b8e9-f698628e7731)
![image](https://github.com/RichardMandle/progfig/assets/101199234/14c518f4-a89f-4203-ae12-aa3e4c342c24)

The _define_vectors_ function takes the points generated above (with hexatic and/or randomisation) and returns a list of vectors of of length _vector_length_. The user supplies _P2_, the desired nematic order parameter, and the vectors are generated so that the angle between them and the average vector of the system has is equivilent to this value of _P2_.

In the first case, let's try to make a fairly ordinary nematic-like structure:

~~~
points = generate_regular_points(0.5,3)
points = add_randomness(points,
                        randomness_x=0.375,
                        randomness_y=0.375,
                        randomness_z=0.075)

vectors = define_vectors(points=points,vector_length=0.5,P2=0.6)
~~~

We can plot these 3D vectors using the _plot_vectors_ function: this takes _points_ and _vectors_ as arguments; we also passs figure and axis handles. The argument _colors_ controls the colouring of the vectors, either by the _P1_ or _P2_ order parameters of the individual vectors, or not at all. The argument _apolar_ is boolean; if _false_, the system will be polar wit all vectors oriented the same direction; if _true_, then the orientation of the vectors is randomised. The user can select any colourmap they like from matplotlib with the argument _color_map_,  I think bwr and bwr_r work nicely. _box_ puts a nice box around the image if _True_, or doesn't if _False_

~~~

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_vectors(points, 
             vectors, 
             fig = fig,
             ax = ax,
             colors='p2',
             apolar=True,
             box=True,
             color_map='bwr')

~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/34a1f03c-b4fe-4957-b634-b8bab2bb6ef3)

The program also tells us that this image has a <P1> order parameter of ~ 0.05 and <P2> of ~ 0.604; pretty good; we asked for an apolar nematic with P2=0.6, so this is acceptable.

We can also construct images of _polar_ nematic phases (RM734 and the like):
~~~

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p1',
             apolar=False,
             box=True,
             color_map='coolwarm')
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/df1c6d56-8cf5-41e6-8882-44940e5d5083)

  
Lastly, we can plot with an ellipsoidal representation. We take _points_ and _vectors_ as inputs, as above. _aspect_ratio_ controls the aspect ratio of the ellipsoids. _overlap_threshold_ removes overlapping ellipsoids according to a distance-based cutoff. other arguments like _apolar_, _box_, _color_map_, _colors_, work as above. 

For _plot_ellipsoid_ to work nicely you want a sightly larger point spacing from _generate_regular_points_ than you'd use for vector-based plots:

~~~
points = generate_regular_points(0.75,5)
points = add_randomness(points,
                        randomness_x=0.375,
                        randomness_y=0.375,
                        randomness_z=0.375)

vectors = define_vectors(points,0.5,0.6)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_ellipsoid(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p2',
             apolar=True,
             box=True,
             overlap_threshold=0.75,
             color_map='bwr')
~~~
![image](https://github.com/RichardMandle/progfig/assets/101199234/e38d0156-7f34-48b6-9234-724a5a0ad070)

This rather nice image has the following order paramter values:

~~~
P1 = 0.594
P2 = 0.012
~~~

By playing around with different values of _add_randomness_ we can quite easily generate layered (i.e. pseudo smectic) structures:

~~~
points = generate_regular_points(0.5,3)
points = add_randomness(points,
                        randomness_x=0.325,
                        randomness_y=0.325,
                        randomness_z=0.05) # keep this fairly small; the reduced displacement from initial positions in z- will mimic a layered structure

vectors = define_vectors(points=points,vector_length=0.5,P2=0.8)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p1',
             apolar=True,
             box=True,
             color_map='coolwarm')
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/304785b2-f5ac-4574-9d6e-d00d584e3591)

We can use the _add_tilt_ function to create tilted structures. Our arguments here are _points_ and _vectors_, as well as _tilt_angle_ (self-explanatory) and _rotation_axis_, which is the 3-vector against which we are going to apply the tilt (e.g. [1,0,0] is X-axis):

~~~
points,vectors = add_tilt(points, vectors, tilt_angle=33, rotation_axis=[1, 0, 0])

plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p1',
             apolar=True,
             box=True,
             color_map='coolwarm')

~~~
![image](https://github.com/RichardMandle/progfig/assets/101199234/28f4820d-69d7-4bea-aba6-ce372b4bacf0)

Lastly, we can make something like a hexatic smectic phase by using the _hexatic_offset_ function; this takes a set of input points (which are presumed to be cubic as produced by _generate_regular_points_) and offsets each 'row' by 1/2 the spacing, giving a hexatic packing. The argument _plane_offset_ will offset adjacent layers by 1/2 of spacing so as to give a different packing.

To preserve the hexagonal packing you'll want to use small randomness valiues in _add_randomness_:

~~~

points = hexatic_offset(points,plane_offset=False)

points = add_randomness(points,
                        randomness_x=0.1,
                        randomness_y=0.1,
                        randomness_z=0.05)

~~~

### ODF

You can extract something sort-of like the ODF using _inspect_angles_:

~~~
points = generate_regular_points(0.5,20)
vectors = define_vectors(points,0.5,0.6)
p2 = inspect_angles(vectors)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/6457e9c2-5cd8-42e9-9f0e-fee3da1d7665)

This also returns P1 and P2:



~~~
P1 = 0.824
P2 = 0.597
~~~


### Future plans

Some additional colouring options; by tilt, by layer-membership for example.
Ability to plot discs (for columnar type structures).
Rewrite of generate_regular_points to give control over spacing in different dimensions; we might sometimes want z- to be different.
