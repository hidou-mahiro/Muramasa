import os
import cv2

array_of_img = [] 

class picturin(object):
    def __init__(self, directory_name):
        self.directory_name = directory_name
        
    def __call__(self):
        # this loop is for read each image in this foder,directory_name is the foder name with images.
        for filename in os.listdir(r"./"+self.directory_name):
            #print(filename) #just for test
            #img is used to store the image data 
            img = cv2.imread(self.directory_name + "/" + filename)
            array_of_img.append(img)
            #print(img)
        return array_of_img
    
    
