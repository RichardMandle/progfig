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

The _define_vectors_ function takes the points generated above (with hexatic and/or randomisation) and returns a list of vectors of of length _vector_length_. The user supplies _P2_, the desired nematic order parameter, and the vectors are generated so that the angle between them and the average vector of the system has is equivilent to this value of _P2_:

~~~
points = generate_regular_points(0.5,2)
points = add_randomness(points,0)
vectors = define_vectors(points,0.5,0.6)
~~~

We can plot these 3D vectors using the _plot_vectors_ function: this takes _points_ and _vectors_ as arguments; we also passs figure and axis handles. The argument _colors_ controls the colouring of the vectors, either by the _P1_ or _P2_ order parameters of the individual vectors, or not at all. The argument _apolar_ is boolean; if _false_, the system will be polar wit all vectors oriented the same direction; if _true_, then the orientation of the vectors is randomised.

~~~

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p1',
             apolar=False)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/53ad54bb-2650-452e-82f9-6a8cf15fdc52)

~~~

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p1',
             apolar=True)
~~~


  
Lastly, we can plot with an ellipsoidal representation. We take _points_ and _vectors_ as inputs, as above. _aspect_ratio_ controls the aspect ratio of the ellipsoids. _overlap_threshold_ removes overlapping ellipsoids according to a distance-based cutoff. _apolar_ is boolean, and works as above. Colouring is done automatically based on wether or not we have a polar or apolar system, as the lack of arrowheads makes them indistinguishable without colour.

~~~
plot_ellipsoid(points,
               vectors,
               aspect_ratio=4,
               overlap_threshold=0.75,
               apolar=False)
~~~
![image](https://github.com/RichardMandle/progfig/assets/101199234/5b7b20bb-101e-41ad-ab63-1727ac06f058)

~~~
P1 = 0.835
P2 = 0.614
~~~

~~~
plot_ellipsoid(points,
               vectors,
               aspect_ratio=4,
               overlap_threshold=0.75,
               apolar=True)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/866f8b6c-1dc6-49c8-a0d5-764dcd9f2e26)
~~~
P1 = 0.008
P2 = 0.614
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
### Sepcial Cases

You can represent something like a smectic A phase by judicious choice of spacing and add_randomness:

~~~
points = generate_regular_points(1,5) # increased spacing for visibility
points = add_randomness(points,
                        randomness_x=0.25,
                        randomness_y=0.25,
                        randomness_z=0.125) # reduced randomness in Z

vectors = define_vectors(points,0.5,0.75) # same, but increase P2 for realism

plot_vectors(points,
             vectors,
             fig = fig,
             ax = ax,
             colors='p1',
             apolar=True)
~~~

![image](https://github.com/RichardMandle/progfig/assets/101199234/3921dd71-65a7-4ad3-a01e-d4b6cfb2c8c0)

### Future plans

Better code for hexatic packing would be good. 
Code for tilted smectics needed
What happens if aspect_ratio is >1? Do we get discs?
