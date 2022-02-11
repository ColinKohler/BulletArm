import os


for obj in range(0, 89):
# for obj in range(0, 1):
    objname = str(obj).zfill(3)
    # convex decomposition
    # os.system('~/bullet3/bin/test_vhacd_gmake_x64_release --input {}/textured.obj --output {}/convex.obj'
    #           .format(objname, objname))

    # cleaning big files
    # for item in ['nontextured.ply', 'nontextured_simplified.ply', 'texture_map.png',
    #              'textured.mtl', 'textured.obj', 'textured.sdf']:
    # for item in ['textured.jpg']:
    #     os.system('rm {}/{}'.format(objname, item))

    print(objname)
