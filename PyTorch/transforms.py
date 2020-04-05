# Understanding how transform function works


## First is mean,  then is std!
tranforms.Normalize([ 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  )
'''
Normalize does the following for each channel:

image = (image - mean) / std

The parameters mean, std are passed as 0.5, 0.5 in your case. This will normalize the image in the range [-1,1]. For example, the minimum value 0 will be converted to (0-0.5)/0.5=-1, the maximum value of 1 will be converted to (1-0.5)/0.5=1.

if you would like to get your image back in [0,1] range, you could use,

image = ((image * std) + mean)

'''

image.transpose((1,2,0))

# It is important to understand how images are processed by different libraries.
# NUMPY:  HxWXC
# pyTorch: BxCxHxW