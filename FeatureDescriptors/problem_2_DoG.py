'''
Created on Mar 4, 2023

@author: aelsayed
'''
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 

def gaussian_filter(kernel_size, sigma=1, muu=0):
    x = np.linspace(-1*kernel_size/2.0, kernel_size/2.0, np.uint8(kernel_size))
    y = np.linspace(-1*kernel_size/2.0, kernel_size/2.0, np.uint8(kernel_size))
    x, y = np.meshgrid(x,y)
    dst = np.sqrt(x**2+y**2)
    normal = 1/(2.0 * np.pi * sigma**2)
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return gauss

def DoG_filter(sigma1=1, sigma2=2):
    #kernel_size = 3         #Edit this line to choose the correct kernel size
    kernel_size = int(np.ceil(6 * max(sigma1, sigma2)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    filter1 = gaussian_filter(kernel_size, sigma=sigma1)
    filter2 = gaussian_filter(kernel_size, sigma=sigma2)
    DoG = filter1-filter2
    return DoG
    
def convolve2D(img, kernel, padding=1, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel)) # If the kernel is symmetric we can ignore this step

    # Gather Shapes of Kernel + Image + Padding
    x_kernshape = kernel.shape[0]
    y_kernshape = kernel.shape[1]
    x_imgshape = img.shape[0]
    y_imgshape = img.shape[1]
    # Shape of Output Convolution
    x_output = int(((x_imgshape - x_kernshape + 2 * padding) / strides)+1)     #Edit this line if the padding has a dimension problem
    y_output = int(((y_imgshape - y_kernshape + 2 * padding) / strides)+1)     #Edit this line if the padding has a dimension problem
    output = np.zeros((x_output, y_output))
    # Apply Equal Padding to All Sides
    if padding != 0:
        imgPadded = np.zeros((img.shape[0] + padding*2, img.shape[1] + padding*2))
        imgPadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = img
    else:
        imgPadded = img
    # Iterate through image
    for y in range(img.shape[1]):
        # Exit Convolution
        if y > imgPadded.shape[1] - y_kernshape + 1:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(img.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imgPadded.shape[0] - x_kernshape + 1:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imgPadded[x: x + x_kernshape, y: y + y_kernshape]).sum()
                except:
                    break

    return output


def main():
    image_filename = 'D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\sunflower_small.jpg'
    img_color=Image.open(image_filename)
    img = ImageOps.grayscale(img_color)
    img_color = np.array(img_color)
    npimg = np.array(img)

    blob_radius = 28  # Estimated radius of sunflower cores in pixels
    sigma1 = blob_radius / np.sqrt(2)
    sigma2 = 0.9 * sigma1

    kernel = DoG_filter(sigma1, sigma2=sigma2)                       #Calculate and edit the correct sigma value based on the relation between the required blob size and sigma
    padding = kernel.shape[0] // 2
    output = convolve2D(np.float32(npimg),kernel,padding=padding)   #Calculate and edit  the correct padding based on the kernel filter size
    #img_color[output>0.6*output.max()]=[255,0,0]
    # Add your code here to save the output marked image
    threshold = 0.68 * output.max()  # Adjust threshold as needed
    img_color[output > threshold] = [255, 0, 0]
    output_image = Image.fromarray(img_color)
    output_image.save('D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\sunflower_dog_detected.jpg')

    plt.imshow(img_color)
    plt.axis('off')
    plt.show()
    pass


if __name__ == '__main__':
    main()
    pass