### description
In order to get the foraging area of bumblebees in a low-cost method, this project aims to track bumblebees by putting a retroreflector on the bumblebee and took a group of images about 60 metres away. The retroreflector will appear as a dot in the image when flash is true. The group of images is like below:
1.no-flash photo
2.flash photo
3.no-flash photo
4.last flash photo

This repository extracts features from each group of images and uses several machine learning algorithms to classify the real dots among the top5 brightest spots. The true spot may not be the top5 brightest, but this situation will be ignored in order to make sure the low false-positive rate.

### Extract features from a group:
Firstly, convert three images and the combinations of them to numpy array like below and condiser them as basic features:
* no_flash1
* no_flash2
* flash
* flash-no_flash1
* flash-no_flash2
* flash-lst_flash

Secondly, get the co-ordinates of the top5 brightest spots in **flash-no_flash2** and make following implements on five co-ordinates of basis features:
1. get the n by n cut images;
2. get the mean and standard deviation of 1;
3. get the mean and standard deviation of concentric ring;
4. get the edge detection of 1 by convolution;
5. get the mean and standard deviation of 4.

There are $6 \times 7 = 42$ features totally.