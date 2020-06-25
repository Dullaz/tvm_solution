import tvm
from tvm.contrib import graph_runtime
import numpy as np
import cv2

#Load compiled tvm model and parameters
loaded_lib = tvm.runtime.load_module("deploy.dll")
loaded_json = open("deploy.json").read()
loaded_params = bytearray(open("deploy.params", "rb").read())

#create tvm module
module = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu(0))

#load params into module
module.load_params(loaded_params)

#load input data, and reshape
input_data = np.asarray(cv2.imread("input.bmp") / 255,dtype="float32")[:,:,0]
input_data = np.reshape(input_data,(1,1,256,256))

#set the input for the tvm module
module.set_input(input_1=input_data)

module.run()

#get the output
out = module.get_output(0)

#convert to numpy array, show and save
final = np.asarray(out.asnumpy()[0,0,:,:] * 255,dtype="uint8")
cv2.imwrite("output.bmp",final)
cv2.imshow("test",final)
cv2.waitKey(0)

