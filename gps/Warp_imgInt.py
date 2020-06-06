#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:50:02 2020

@author: braso
"""

import numpy           as np
import matplotlib.pyplot as plt
import cv2
from glob import glob


#folder = "/home/braso/Agricultura_UNQ/CodigosStiching/Frames/*.jpg"
folder='/home/braso/Agricultura_UNQ/gps/Para SURF/*.PNG'
files =glob(folder)
files.sort()

img=list()

for file in files:
    print()
    img.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

img[0]=img[0][210:,0:1346]
img[1]=img[1][:,0:1346]

pts_src=[]
pts_dst=[]

# Selecciona 4 puntos iguales en cada foto, elegis uno en una y el mismo en la otra. 
# Asi hasta completar 4 veces(Trato de que sea las 4 esquinas de una parcela)
for i in range(1,4):
	print(i)
	plt.figure('Seleccionar puntos imagen 1')
	plt.imshow(img[0],'gray')
	print('ELegir un punto de interés:\t\n',end=' ')
	cx1,cy1=np.array(np.round(plt.ginput()[0]),dtype=np.int32)
	print('Punto en x {:.4f}\n punto en y {:.4f} \n'.format(cx1,cy1))
	pts_src.append([cx1,cy1])
	plt.figure('Seleccionar puntos imagen 2')
	plt.imshow(img[1],'gray')
	print('ELegir un punto de interés:\t\n',end=' ')
	cx2,cy2=np.array(np.round(plt.ginput()[0]),dtype=np.int32)
	print('Punto en x {:.4f}\n punto en y {:.4f} \n'.format(cx2,cy2))
	pts_dst.append([cx2,cy2])
	
	
# %% Transformacion con Homografia , CAMBIAR EL FOR de arriba para que agarre 4 puntos. 
pts_src=np.array(pts_src)
pts_dst=np.array(pts_dst)
h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(img[0], h, (img[1].shape[1],img[1].shape[0]))
cv2.imshow("Source Image", cv2.pyrDown(img[0]))
cv2.imshow("Destination Image",cv2.pyrDown(img[1]))
cv2.imshow("Warped Source Image",cv2.pyrDown(im_out))

res= (im_out/2)+(img[1]/2)
plt.figure('Imagenes ALineadas')
plt.imshow(res,'gray')

# %% Transformacion AFFINE con 3 puntos, sin tener encuenta cambios de amplitud solo traslacion
pts_src=np.array(np.float32(pts_src))
pts_dst=np.array(np.float32(pts_dst))
M = cv2.getAffineTransform(pts_src,pts_dst)
im_out = cv2.warpAffine(img[0], M, (img[1].shape[1],img[1].shape[0]))
cv2.imshow("Source Image", cv2.pyrDown(img[0]))
cv2.imshow("Destination Image",cv2.pyrDown(img[1]))
cv2.imshow("Warped Source Image",cv2.pyrDown(im_out))

res= (im_out/2)+(img[1]/2)
plt.figure('Imagenes ALineadas')
plt.imshow(res,'gray')


