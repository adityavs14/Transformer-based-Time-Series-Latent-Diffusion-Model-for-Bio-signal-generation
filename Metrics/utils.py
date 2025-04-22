import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

class utility:

    def __init__(self):
        process = "True"

    def load_image(self,path):
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img/255.
    

    def convert(self,data):

        IMG = []

        for i in data:
            fig, ax = plt.subplots()
            ax.plot(i) 

            # Remove axes and background for clean image
            fig.patch.set_visible(False)
            ax.axis('off')

            # Render the plot and extract the image data
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Convert to HxWx3

            # Convert to grayscale using OpenCV
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            IMG.append(gray_img)

            # Close the plot
            plt.close(fig)


        IMG = np.array(IMG)
        
        return IMG
    
    