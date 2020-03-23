## Why

Data augmentation is an integral process in deep learning, as in deep learning we need large amounts of data and in some cases it is not feasible to collect thousands or millions of images, so data augmentation comes to the rescue.

It helps us to increase the size of the dataset and introduce variability in the dataset.

## Operations

- Rotation
- Shearing, shear-X and shear-Y
- Zooming
- Cropping, selecting a specific area from image
- Flipping, use this carefully because it might not make much sense, e.g. a face, a person will not be upside down to a camera.
- Changing the brightness level

## Code

```python
# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,  
array_to_img, img_to_array, load_img 
   
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5)) 
    
# Loading a sample image  
img = load_img('image.jpg')  
# Converting the input sample image to an array 
x = img_to_array(img) 
# Reshaping the input image 
x = x.reshape((1, ) + x.shape)  
   
# Generating and saving 5 augmented samples  
# using the above defined parameters.  
i = 0
for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir ='preview',  
                          save_prefix ='image', save_format ='jpeg'): 
    i += 1
    if i > 5: 
        break
```

## Useful Links

[GeeksforGeeks - Python Data Augmentation](https://www.geeksforgeeks.org/python-data-augmentation/)   
[tutorialspoint - 2D Transformations](https://www.tutorialspoint.com/computer_graphics/2d_transformation.htm)