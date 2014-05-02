snow
====

Take a look at the project site: http://wyegelwel.github.io/snow

Project dependencies:

* Cuda (at least compute capacity 2.1)
* glm
* OpenGL (probably works if cuda does...)
* GLEW
* QtCreator

To build the project, open project/snow.pro in QtCreator and then look at the file. You will likely need to change the path of a few of the dependencies. Then run build. If all goes well, you should be able to run. 

Take a look at snow_math.pdf and snow_implicit_math.pdf for a sense of the implementation.

We implemented some features that may be of interest to people not interested in snow. These include:

* A 3x3 svd solver for cuda (__device__) which as been snow tested
* A vec3 and mat3 class with most features you would expect


