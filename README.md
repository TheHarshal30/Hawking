<div align="center"> 
   <img  src="/temp/deep-learning.svg">
   <img  src="/temp/astronomy.svg">
   
</div>

<div align="center"> 
  <h1>Near Earth Object Classifier</h1>
   <img height=300 width=900 src="/temp/comets.gif">
</div>

## Context
There is an infinite number of objects in the outer space. Some of them are closer than we think. Even though we might think that a distance of 70,000 Km can not potentially harm us, but at an astronomical scale, this is a very small distance and can disrupt many natural phenomena. These objects/asteroids can thus prove to be harmful. Hence, it is wise to know what is surrounding us and what can harm us amongst those.

## Dataset
This dataset compiles the list of NASA certified asteroids that are classified as the nearest earth object.
<img width="750" height="200" src="/temp/dataset.png">

## Models:

<img align="right" src="/temp/Near Earth Asteroids.jpg" width="350" height="200">

| Model                        |  Accuracy      
| -----------------------------|  -------------| 
| Using Bayes Theorem          |  90.41        |
| Neural Network using Pytorch |  91.186       |
|XG Boost                      |  91.23        |
|Random Forest                 |  92.07        |
|DecisionTreeClassifier        |  89.33        |
|KNeighborsClassifier 	       |  88.06        |


## TechStack Used:
1.Pytorch <br />
3.Scikitlearn <br />
4.Numpy <br />
5.Pandas <br />
2.Pytorch-Tabnet - https://arxiv.org/abs/1908.07442 <br />
6.NASA Open API - https://api.nasa.gov <br />
7.NEO Earth Close Approches - https://cneos.jpl.nasa.gov/ca/ <br />


