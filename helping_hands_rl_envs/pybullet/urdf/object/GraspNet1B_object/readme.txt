COPY RIGHT: GRASPNET GROUP, MVIG, SJTU RESERVED

For the model files, nontextured.ply is the ply file for the obj but does map the texture file to face and vertices. 
The nontextured_simplified.ply is its simplified version, for easy loading in simulator like Bullet and V-REP.

You can also use the sim_mesh.py to further simplify the mesh by first enlarging the value of "Cell Size" (e.g., from 0.0005 to 0.001) in the sim_mesh.mlx and then running the sim_mesh.py. Meshlab is required to use this script.