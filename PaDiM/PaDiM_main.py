import sys
import os
import numpy as np
import tensorflow as tf
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from mvtec_loader import MVTecADLoader

mvtec_loader = MVTecADLoader()
bottle = mvtec_loader.load('bottle', repeat=4, max_rot=10)

print(type(bottle.x))
