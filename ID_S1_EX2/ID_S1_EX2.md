# ID_S1_EX2 exercise writeup

## 6 examples of vehicles 
Find and display 6 examples of vehicles with varying degrees of visibility in the point-cloud

### 1: regular car very visible incl sidemirrors

 ![alt text](vehicle-examples/regular-car-very-visible-incl-sidemirrors.png)

### 2: pickup truck with trailer

![alt text](vehicle-examples/pickup-truck-with-trailer.png)

### 3: clearly a car despite occlusions

![alt text](vehicle-examples/clearly-a-car-despite-occlusions.png)

### 4: probably car but hard to say

![alt text](vehicle-examples/probably-car-but-hard-to-say.png)

### 5: probably a car but few features

![alt text](vehicle-examples/probably-a-car-but-few-features.png)

### 6: mini van with large window high roof

![alt text](vehicle-examples/mini-van-with-large-window-high-roof.png)

### 7: vehicle which seems maybe parked

![alt text](vehicle-examples/vehicle-maybe-parked.png)

### 8: Vehicle very close to ego / ourselves

![alt text](vehicle-examples/vehicle-very-close-to-ego.png)

### 9: possibly a pedestrian, very close by

![alt text](vehicle-examples/possibly-pedestrian.png)

### 10: potentially occluded unseen vehicles

![alt text](vehicle-examples/potentially-occluded-unseen-vehicles.png)




## Identify vehicle features
Identify vehicle features that appear as a stable feature on most vehicles (e.g. rear-bumper, tail-lights) and describe them briefly. Also, use the range image viewer from the last example to underpin your findings using the lidar intensity channel.

### Front Grill & Bumber
many points reflected from the front grille bumber area.
![alt text](vehicle-features/front-grille-and-bumber.png)
![alt text](vehicle-features/front-grille-bumber.png)

### Back Bumber
Also a big area from the back of the car incl bumber. High intensity.

![alt text](vehicle-features/back-bumber.png)
![alt text](vehicle-features/bumber-2.png)

### Side mirrors are sometimes also clearly seen
depending on the angle, side mirrors are clearly visible and a good distinguishing feature

![alt text](vehicle-features/side-mirrors.png)


### Wheels
also depends on angle, but can sometimes be clearly seen as a distinguishing feature.

![alt text](vehicle-features/wheels.png)

### Windows often clearly visible
![alt text](vehicle-features/windows.png)