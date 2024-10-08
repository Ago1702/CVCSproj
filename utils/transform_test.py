import torch
from matplotlib import pyplot as plt
import transform
import numpy as np
from torchvision.transforms import v2
import os
"""
The purpose of this test is evaluating if our custom transformations introduce any kind of bias towards specific regions of the photos
Since the original data has dimensions in the range: [256 , 640]
The following formats will be tested
640 x 400
640 x 640
504 x 504
400 x 400
304 x 304
256 x 256
NB: REMEMBER TO TEST GLOBAL AND LOCAL CROPS EVERY TIME THE CROPPING SIZE IS CHANGED DURING THE DEVELOPMENT PHASE.
Both pacman==True and pacman==False will be tested
A grid with 64 different values will be initialized. Zero will be reserved so that average padding size can be evaluated in the 
pacman==False case.
"""

if __name__ == "__main__":
    sizes: list[tuple[int,int]]=[(256,256),(304,304),(400,400),(504,504),(640,640),(400,640)]
    for size in sizes:
        y = size[0]
        x = size[1]
        image = torch.zeros(1,1,y,x,dtype=torch.int)
        for i in range(8):
            for j in range(8):
                image[:,:,i*int(y/8):(i+1)*int(y/8),j*int(x/8):(j+1)*int(x/8)]=(i+1)+8*(j+1)
        # now the image should be filled with the correct values

        x_values = torch.arange(73).numpy()

        #local vanilla testing
        localvanilla = np.zeros((73),dtype=np.int64) 
        print("Progress with local vanilla:")
        for k in range(1000):
            if k%100==0:
                print(f'{k/10}%')
            transformer = transform.RandomTransform(p=1,scale=1,pacman=False,cropping_mode=transform.RandomTransform.LOCAL_CROP,
                                                          transform=(v2.RandomPerspective(p=1), v2.RandomAffine(90), v2.RandomAdjustSharpness(2, 1),
                                    v2.RandomHorizontalFlip(p=1),v2.RandomVerticalFlip(p=1)))
            output = transformer(image)
            histogram_values = torch.histc(input=output.float(),bins=73,min=0,max=72).numpy()
            localvanilla+=histogram_values.astype(np.int64)

        # bar chart
        plt.figure()
        plt.bar(x_values, localvanilla, width=0.8)  # Use bars instead of lines
        plt.title(f"Bar Plot for Histogram of size {size}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # saving the plot
        save_path = os.path.expanduser(f"~/CVCSproj/outputs/histograms/vanilla/local/bar_plot_{y}x{x}.png")
        plt.savefig(save_path)

        #global vanilla testing
        globalvanilla = np.zeros((73),dtype=np.int64) 
        print("Progress with global vanilla:")
        for k in range(1000):
            if k%100==0:
                print(f'{k/10}%')
            transformer = transform.RandomTransform(p=1,scale=1,pacman=False,cropping_mode=transform.RandomTransform.GLOBAL_CROP,
                                                          transform=(v2.RandomPerspective(p=1), v2.RandomAffine(90), v2.RandomAdjustSharpness(2, 1),
                                    v2.RandomHorizontalFlip(p=1),v2.RandomVerticalFlip(p=1)))
            output = transformer(image)
            histogram_values = torch.histc(input=output.float(),bins=73,min=0,max=72).numpy()
            globalvanilla+=histogram_values.astype(np.int64)

        # bar chart
        plt.figure()
        plt.bar(x_values, globalvanilla, width=0.8)  # Use bars instead of lines
        plt.title(f"Bar Plot for Histogram of size {size}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # saving the plot
        save_path = os.path.expanduser(f"~/CVCSproj/outputs/histograms/vanilla/global/bar_plot_{y}x{x}.png")
        plt.savefig(save_path)

        #local pacman testing
        localpacman = np.zeros((73),dtype=np.int64) 
        print("Progress with local pacman:")
        for k in range(1000):
            if k%100==0:
                print(f'{k/10}%')
            transformer = transform.RandomTransform(p=1,scale=1,pacman=True,cropping_mode=transform.RandomTransform.LOCAL_CROP,
                                                          transform=(v2.RandomPerspective(p=1), v2.RandomAffine(90), v2.RandomAdjustSharpness(2, 1),
                                    v2.RandomHorizontalFlip(p=1),v2.RandomVerticalFlip(p=1)))
            output = transformer(image)
            histogram_values = torch.histc(input=output.float(),bins=73,min=0,max=72).numpy()
            localpacman+=histogram_values.astype(np.int64)

        # bar chart
        plt.figure()
        plt.bar(x_values, localpacman, width=0.8)  # Use bars instead of lines
        plt.title(f"Bar Plot for Histogram of size {size}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # saving the plot
        save_path = os.path.expanduser(f"~/CVCSproj/outputs/histograms/pacman/local/bar_plot_{y}x{x}.png")
        plt.savefig(save_path)

        #global pacman testing
        globalpacman = np.zeros((73),dtype=np.int64) 
        print("Progress with global pacman:")
        for k in range(1000):
            if k%100==0:
                print(f'{k/10}%')
            transformer = transform.RandomTransform(p=1,scale=1,pacman=True,cropping_mode=transform.RandomTransform.GLOBAL_CROP,
                                                          transform=(v2.RandomPerspective(p=1), v2.RandomAffine(90), v2.RandomAdjustSharpness(2, 1),
                                    v2.RandomHorizontalFlip(p=1),v2.RandomVerticalFlip(p=1)))
            output = transformer(image)
            histogram_values = torch.histc(input=output.float(),bins=73,min=0,max=72).numpy()
            globalpacman+=histogram_values.astype(np.int64)

        # bar chart
        plt.figure()
        plt.bar(x_values, globalpacman, width=0.8)  # Use bars instead of lines
        plt.title(f"Bar Plot for Histogram of size {size}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # saving the plot
        save_path = os.path.expanduser(f"~/CVCSproj/outputs/histograms/pacman/global/bar_plot_{y}x{x}.png")
        plt.savefig(save_path)

        '''
        x_values = torch.arange(73).numpy()  # x-axis values (bins)
        histogram_values = histogram.numpy()  # y-axis values (frequencies)
        
        print(histogram_values)
        # bar chart
        plt.figure()
        plt.bar(x_values, histogram_values, width=0.8)  # Use bars instead of lines
        plt.title(f"Bar Plot for Histogram of size {size}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # saving the plot
        plt.savefig(f"outputs/histograms/bar_plot_{y}x{x}.png")'''


