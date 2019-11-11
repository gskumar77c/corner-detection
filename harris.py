import sys
import numpy as np
import pylab
import scipy.ndimage.filters as ndi
import skimage.io
import skimage.transform
import cv2
from random import shuffle
from operator import itemgetter



def gaussian1D(n, sigma=3.0):
    result = np.zeros(n)
    mid = int(n / 2)
    result = [(1 / (np.sqrt(2 * np.pi * sigma**2)) * (np.e**(-((i**2) / (2 * sigma**2))))) for i in
              range(-mid, mid + 1)]

    return result

#helper function
def remove_eig(list_tuples):
    new_list = []
    for i in range(len(list_tuples)):
        x,y,e = list_tuples[i]
        new_list.append([x, y])
    return new_list


def find_corners(image, dx, dy, thresh, output, m=4):

    x2 = dx ** 2
    xy = dx*dy
    y2 = dy ** 2

    offset = 2*m+1

    corners = []
    percentCorners = .1 #percentage of corners to actually draw

    print ("Looking for corners...")
    for y in range(offset, image.shape[0]-offset):
        for x in range(offset, image.shape[1]-offset):

            Fx2 = x2[y - offset:y + offset, x - offset:x + offset]
            Fxy = xy[y - offset:y + offset, x - offset:x + offset]
            Fy2 = y2[y - offset:y + offset, x - offset:x + offset]

            Fx2 = Fx2.sum()
            Fxy = Fxy.sum()
            Fy2 = Fy2.sum()

            C = np.array([[Fx2, Fxy],[Fxy, Fy2]]) #covariance matrix
            e,v = np.linalg.eig(C) #calc eigenvalues
            e = np.amin(e) #smallest eigenvalue = corner response

            if e > thresh:
                corners.append([x,y,e])

    corners.sort(key=itemgetter(2), reverse=True)

    corner_points = remove_eig(corners)
    corner_points_copy = list(corner_points)

    for i in range(len(corner_points)):

        x,y = corner_points[i]
        corner_points_copy.append([x,y])

        #Remove surrounding neighbors from list as they are not the maximum
        if [x+1, y] in corner_points_copy: corner_points_copy.remove([x+1, y])
        if [x+1,y+1] in corner_points_copy: corner_points_copy.remove([x+1, y+1])
        if [x + 1, y - 1] in corner_points_copy: corner_points_copy.remove([x+1, y-1])
        if [x, y + 1] in corner_points_copy: corner_points_copy.remove([x, y+1])
        if [x, y - 1] in corner_points_copy: corner_points_copy.remove([x, y-1])
        if [x - 1, y] in corner_points_copy: corner_points_copy.remove([x-1, y])
        if [x - 1, y - 1] in corner_points_copy: corner_points_copy.remove([x-1, y-1])
        if [x - 1, y + 1] in corner_points_copy: corner_points_copy.remove([x-1, y+1])

    shuffle(corner_points_copy)

    for i in range(int(len(corner_points_copy)*percentCorners)):
        x,y = corner_points_copy[i]
        cv2.circle(output,(x,y),2,(1,0,0),-1)


    return output


def main():

	l = ['bicycle.bmp','bird.bmp','dog.bmp','einstein.bmp','plane.bmp','toy_image.jpg']
	g = ['bicycle','bird','dog','einstein','plane','toy_image']
	s = "data/"
	for i in range(0,len(l)):
		I = skimage.img_as_float(skimage.io.imread(s+l[i]))

		I = I.astype('float32')
		I_grey = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

		sigma = 1
		kernel_size = 9
		H = np.array(gaussian1D(kernel_size, sigma))
		V = H

		dv = np.array([-1, 0, 1])

		H_dx = ndi.convolve1d(H, dv)
		V_dy = ndi.convolve1d(V, dv)

		G_x = ndi.convolve1d(I_grey, H_dx, 0)
		G_y = ndi.convolve1d(I_grey, V_dy, 1)

		threshold = 0.55

		result = find_corners(I_grey, G_x, G_y, threshold, I)

		pylab.imshow(result)

		skimage.io.imsave('houtput/'+g[i]+'.png', result.astype('float32'))

		pylab.show()


if __name__ == '__main__':
    main()
