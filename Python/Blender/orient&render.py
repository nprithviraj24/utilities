import bpy
import os
import sys
fullpath = os.getcwd()
idname = os.path.basename(fullpath)

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

print(argv[0])  # --> ['example', 'args', '123']
obj = bpy.context.active_object
obj.rotation_mode = 'XYZ'

#left almost straight (90)
orientation = "l_90_"+str(argv[0])
tx=-1; ty = 1; tz=-1
bpy.ops.transform.translate(value=(tx, ty, tz))

X=0; Y=0; Z=1
obj.rotation_euler = (X,Y,Z)

bpy.data.scenes['Scene'].render.filepath = str(fullpath)+'/z_Rendered Images/'+str(orientation)+'.png'
bpy.ops.render.render(write_still=True)

tx=1; ty =-1; tz=1
bpy.ops.transform.translate(value=(tx, ty, tz))


#right profile (45)
orientation = "r_45_"+str(argv[0])
tx=-0.5; ty = 0; tz=0
bpy.ops.transform.translate(value=(tx, ty, tz))

X=0; Y=0; Z=1
obj.rotation_euler = (X,Y,Z)

bpy.data.scenes['Scene'].render.filepath = str(fullpath)+'/z_Rendered Images/'+str(orientation)+'.png'
bpy.ops.render.render(write_still=True)

tx=0.5; ty =0; tz=0
bpy.ops.transform.translate(value=(tx, ty, tz))


#right almost straight (90)
orientation = "r_90_"+str(argv[0])
tx=-1; ty = -0.5; tz=0
bpy.ops.transform.translate(value=(tx, ty, tz))

X=0.1; Y=0; Z=0.6
obj.rotation_euler = (X,Y,Z)

bpy.data.scenes['Scene'].render.filepath = str(fullpath)+'/z_Rendered Images/'+str(orientation)+'.png'
bpy.ops.render.render(write_still=True)

tx=1; ty =0.5; tz=0
bpy.ops.transform.translate(value=(tx, ty, tz))


#straight (90)
orientation = "s_90_"+str(argv[0])
X=0.1; Y=0; Z=0.9
obj.rotation_euler = (X,Y,Z)

bpy.data.scenes['Scene'].render.filepath = str(fullpath)+'/z_Rendered Images/'+str(orientation)+'.png'
bpy.ops.render.render(write_still=True)


#left profile (45)
orientation = "l_45_"+str(argv[0])
tx=0; ty = 0.5; tz=0
bpy.ops.transform.translate(value=(tx, ty, tz))

X=0.1; Y=0; Z=1.5
obj.rotation_euler = (X,Y,Z)

bpy.data.scenes['Scene'].render.filepath = str(fullpath)+'/z_Rendered Images/'+str(orientation)+'.png'
bpy.ops.render.render(write_still=True)




