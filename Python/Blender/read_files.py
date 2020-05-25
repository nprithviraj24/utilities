import os
#print(os.path.basename(os.getcwd()))

#dirpath = os.getcwd()
#print("current directory is : " + dirpath)
#foldername = os.path.basename(dirpath)
#print("Directory name is : " + foldername)
#scriptpath = os.path.realpath(__file__)
#print("Script path is : " + scriptpath)
import subprocess

from os import walk

#f = []
dirs = []
#dirpaths = []
for (dirpath, dirnames, filenames) in walk(os.getcwd()):
    dirs.extend(dirnames)
    #f.extend(filenames)
    #print(dirpath)
    #print(dirnames)
    break
#print(dirs)
#print(direc)

for each in dirs:
    a = each.split("-")
    blenderfile = str(each)+"/"+str(a[0])+"_d.blend"
    subprocess.call(["blender","--background", blenderfile, "--python", "orient&render.py","--", a[0]])
    
    

