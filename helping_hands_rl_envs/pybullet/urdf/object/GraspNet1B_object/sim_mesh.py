import os

#root = '../batch2'
#objlist = sorted(os.listdir(root))

for obj in range(0,89):
    objname = str(obj).zfill(3)

   # os.system('cp {}/close* ./'.format(os.path.join(root, objname), objname))
    #os.system('meshlab.meshlabserver -s process_raw_scan.mlx -i close.obj -o textured.obj -m vc vf vq vn vt fc ff fq fn wc wn wt')
    # os.system('/usr/bin/meshlab/distrib/meshlabserver -i {}/nontextured.ply -o {}/nontextured_simplified.ply -m vc vf vn fc ff fn -s sim_mesh.mlx'.format(objname,objname))
    os.system('meshlabserver -i {}/nontextured.ply -o {}/nontextured_simplified.ply -m vc vf vn fc ff fn -s sim_mesh.mlx'.format(objname,objname))
    #os.system('mv textured* {}/'.format(objname))
    #os.system('mv close.jpg {}/textured.jpg'.format(objname, objname))
    #os.system('rm close*'.format(objname))
    #f = open('{}/textured.obj.mtl'.format(objname))
    #lines = f.readlines()
    #f.close()
    #os.system('rm {}/textured.obj.mtl'.format(objname))
    #f = open('{}/textured.obj.mtl'.format(objname), 'w')
    #lines[-2] = lines[-2].replace('close.jpg', 'textured.jpg')
    #f.writelines(lines)
    #f.close()
    print(objname)
