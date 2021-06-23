[33mcommit 031faed8c6a8ecf792cfdccfb97d6ab46a2ee720[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Dec 11 17:47:04 2020 -0500

    remove multi_task_env

[33mcommit d797eb366105a5158abcf01131dff642849dc3cb[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Dec 9 17:54:05 2020 -0500

    add a parameter 'hard_reset_freq' that controls the frequency of hard reset; fix some import problems; fix function name of getStepsLeft in planners

[33mcommit 44042d21eab85f0e8ee9cfbfd0f7a70244437ae1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Dec 9 13:49:25 2020 -0500

    use 0.01 in adjustGripperCommand to provide firmer grasp

[33mcommit 47cf31ead2c0bd74083adc39b91376302559a53f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Dec 9 13:48:59 2020 -0500

    refactored ramp improvise house building 3

[33mcommit 097457e355fd46f1ada41928c77e4ebe0d076ac6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Dec 9 11:42:12 2020 -0500

    rename improvise_house_building_3, improvise_house_building_4 into improvise_house_building_discrete, improvise_house_building_random

[33mcommit 5a689a3b9d7b0d2da84c093870040ec47d7b4986[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Dec 9 11:08:43 2020 -0500

    refactored ramp_improvise_house_building_2 and deconstruct, changed the normal imh2 to match ramp_imh2's scale and zscale pattern

[33mcommit 90c40bc58e961f0ab5a3a171c79b90009e5cb9ff[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Dec 8 19:50:18 2020 -0500

    align the middle two cube orientations with brick/roof in house 4 deconstruct

[33mcommit 7a8aab18ddb2333d6dc64a59ea0887ec94a2fc57[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Dec 8 19:32:32 2020 -0500

    refactored ramp house 234

[33mcommit a346dae2f786aed2c45a3d87b8bd994f8166bf0b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Dec 8 15:15:46 2020 -0500

    refactored ramp_house_building_1 and ramp_house_building_1_deconstruct

[33mcommit 9313a14934aeffc9c2e018f61a9c827401de4e7e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Dec 3 21:41:13 2020 -0500

    refactored ramp_block_stacking_deconstruct, fix bug in deconstruct env reset where structure_objs is not set empty

[33mcommit e6d0868ad4266495f62aa79bdcf29d43c96a7d7d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Dec 2 11:16:56 2020 -0500

    refactoring, ramp_block_stacking works

[33mcommit 17a4eff040b0378de79a6819983fc252ad6fb0a7[m
Merge: 7519e5f 7f42d6a
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Dec 1 22:43:48 2020 -0500

    Merge branch 'dian_corl' into dian_merge_refactor
    
    # Conflicts:
    #	env_factory.py
    #	envs/base_env.py
    #	envs/numpy_envs/block_picking_env.py
    #	envs/pybullet_deconstruct_env.py
    #	envs/pybullet_envs/pybullet_env.py
    #	planners/base_planner.py
    #	planners/block_structure_base_planner.py
    #	planners/planner_factory.py
    #	simulators/constants.py
    #	simulators/pybullet/robots/kuka.py
    #	simulators/pybullet/robots/robot_base.py
    #	simulators/urdf/object/brick_small.urdf
    #	tests/unittests/test_bullet_block_picking.py
    #	tests/unittests/test_bullet_block_stacking.py
    #	tests/unittests/test_bullet_house_1_deconstruct.py
    #	tests/unittests/test_bullet_house_3.py
    #	tests/unittests/test_bullet_improvise_house_4_deconstruct.py
    #	tests/unittests/test_bullet_random_pick.py

[33mcommit 7519e5f4b1266892bd3acaaa010c928a549e4eb2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Dec 1 22:04:59 2020 -0500

    Revert "Changed the env_factory back to the old way w/the fn wrapper in the creation fn. Using lambda does some weird stuff with the env_configs getting copied."
    
    This reverts commit 78b71ce0

[33mcommit 08889267fe694d8bdea1f8867693845d34074149[m
Merge: 18377eb 78b71ce
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Dec 1 22:02:55 2020 -0500

    Merge remote-tracking branch 'origin/merge_refactor' into dian_merge_refactor
    
    # Conflicts:
    #	env_factory.py

[33mcommit 18377ebc2b55e825996ebac10d33eb7f9ae93c0e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Nov 4 09:31:11 2020 -0500

    test sr

[33mcommit 78b71ce050945edcc7efc6ae55dac21944fdda28[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Thu Oct 29 17:28:02 2020 -0400

    Changed the env_factory back to the old way w/the fn wrapper in the creation fn. Using lambda does some weird stuff with the env_configs getting copied.

[33mcommit 5498e69007a11f9ac3e74a23ebbe91dd380edcab[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Thu Oct 29 15:21:50 2020 -0400

    Added is_sim_valid call to runners, changed step behavior to default to non auto-reset

[33mcommit 21f1485c84031afc96e584b7db7674f7a90f26d8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Oct 29 11:13:55 2020 -0400

    fix bug in env_factory that makes all envs have the same seed

[33mcommit e0c35f1188372f1a1e3efa762f7f8aeb012b915a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Oct 29 11:13:19 2020 -0400

    change physic parameters in 'fast' to match pybullet default

[33mcommit fa5b2bda3c78d6b60844c6a9229a96b47356e987[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Oct 28 22:21:32 2020 -0400

    fix bug of reversed fast and slow parameter, add robot.position_gain in fast (0.02) and slow (0.01)

[33mcommit fd9df4fb252c3f070c4862406ec4dbc56ff50d73[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Wed Oct 28 00:13:26 2020 -0400

    Fixing small mistake.

[33mcommit b03ff305bbf746c8ec22af6189b678d165cf2d17[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Tue Oct 27 22:03:47 2020 -0400

    Removed unit tests backup, added size to objects (need to change this to height, width, length, etc), changed physics solver config settings to modes (fast mode is default at the moment)

[33mcommit 8922244a25a46b51f4955953adda48d76be070ae[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Oct 24 19:48:55 2020 -0400

    add improvise_house_building_2_deconstruct env; rename function generateStructureRandomShapeWithZScale

[33mcommit 32f0f805f82ce64ad2bdf8c372af88ecbc1a3776[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Oct 24 19:15:36 2020 -0400

    add house_building_2_deconstruct and house_building_2_deconstruct envs. Fix a bug in the planner when trying to place cube on top of brick. This also fixes the problem in brick_stacking planner

[33mcommit b1e327212386fb6513e3a2f3b3a84f573a0a8285[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Oct 24 17:04:58 2020 -0400

    refactor deconstruct envs

[33mcommit a19d4c1fce634530eac1ea5874b0f48ed7247b39[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Oct 23 20:16:25 2020 -0400

    only catch NoValidPositionException in reset loop

[33mcommit 69cc7f7753629cc1e9b650e33ffff6a7b465ed2f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Oct 23 20:07:31 2020 -0400

    make the return type in the tests match the tuple style

[33mcommit 00af41d237300ae8b1a0c6b5d3476791893948a1[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 23 17:53:08 2020 -0400

    Fixing more bugs

[33mcommit 04c467f032bf8861eaae89d25ac43d1681b927e5[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 23 13:39:44 2020 -0400

    Moved env creation fn wrapper to env_factory as its cleaner this way.

[33mcommit 11002b8f17d1f6e819db5652c431627adde0d0c6[m
Merge: f266fe2 2e37d8f
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 23 13:33:04 2020 -0400

    Fixing merge conflicts

[33mcommit 2e37d8f0b005c10722e937d97ddcd45d59be07f0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Oct 22 14:07:38 2020 -0400

    tested pybullet envs: house_building_2, house_building_3

[33mcommit b89901f528774ae2f5736c9dec0d8bbaa2287634[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Oct 22 12:46:21 2020 -0400

    add env_constants that contains the content of 'constants.py' under numpy_envs and pybullet_envs; fix the env factory, env runner, and other minor bugs; tested pybullet envs: block_stacking, block_picking, house_building_1, house_building_1_deconstruct

[33mcommit f266fe2e9dbef41a06db0c9f7da797a58f6efb1e[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 23 13:19:17 2020 -0400

    A whole bunch of refactoring

[33mcommit 0b010bda6303c3ef0f367f0aea9952d161db8512[m
Merge: f344466 77eecbd
Author: Colin Kohler <colink78@gmail.com>
Date:   Wed Oct 21 17:47:19 2020 -0400

    Changing things for multi task and fixes for a newer version of pybullet

[33mcommit f34446612079dd64ae4ae6f78c966a7f8e22957a[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Wed Oct 21 17:44:44 2020 -0400

    Added more default values to default dif, minor refacotoring of pybullet init.

[33mcommit 7f42d6a2c6133e6300c21b68ad8f547f70df7549[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Oct 21 10:21:18 2020 -0400

    add tilt_house_building_2_deconstruct

[33mcommit 76542d854a12fa297cdf4729e24c9cd104b3ef11[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 16 17:36:50 2020 -0400

    Added default dict for pybullet env config and cleaned up the default value checking

[33mcommit 61c35540174fa8c607a13b8e4b2e59a5efa253b6[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 16 14:07:13 2020 -0400

    Refactored the pybullet class definitions as they no longer need to be wrapped in a creation function.

[33mcommit 411f3c2ae9d82ff5314172f2e535b494dc7c0d52[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Fri Oct 16 13:40:18 2020 -0400

    Added pipfile for dependencies, removed pytorch from code (still need to do this for unittests)

[33mcommit 77eecbd8657cc0c12d6af4f910ad87b9b0d475fc[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Oct 2 15:34:08 2020 -0400

    Changes to get multi task working better. Had to split this out from the other refactor for the time being. Should be merged later.

[33mcommit fe27bc6e631c51b742b5afde529e5ed12795b1fd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Sep 30 09:26:54 2020 -0400

    match the ori of upper cubes in h4 deconstruct to reduce the chance of collision. Increase the size of ramp for 128 exps

[33mcommit ac6b3954015579c634a5985ce291f4cee03ef80f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Sep 24 11:54:06 2020 -0400

    fix a major bug in in hand projection

[33mcommit f6a52f74def0f7b8a98f168f1717c6532f4d1e4b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Sep 16 20:20:58 2020 -0400

    add 6d initialization for imh2 and imh6

[33mcommit 42ccf2d2c80538a927fc1945aecb55d4a726bd21[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Sep 16 16:20:21 2020 -0400

    Removed the simulator base env from the pybullet envs.

[33mcommit baf27423ada738ca344cea78632b832bcf9a268e[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Sep 16 14:00:56 2020 -0400

    Started the seperation of the numpy and pybullet envs.

[33mcommit 16a419d1a4f495542213b7be61b17661bc87b813[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Sep 16 13:42:24 2020 -0400

    Refactored the creation of env/planners.

[33mcommit bd82b6d65277d6398b9ea6a638280f9e5a016bbb[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Sep 15 12:18:37 2020 -0400

    Changed the pick primative so a pick action becomes a no-op if you are already holding a object

[33mcommit 0f21666bf48ebe5cbaf14ff8266c5c8b38feca14[m
Author: ColinKohler <colink78@gmail.com>
Date:   Sat Sep 12 14:12:13 2020 -0400

    Small change I forgot to make

[33mcommit aace0651220f67bb5842b8072e2e01a6c40d4d47[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Sep 11 17:18:07 2020 -0400

    Missed a comma.

[33mcommit 96149af4fb7cf5baeee38fcd4edde3d618cb4a34[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Sep 11 16:05:17 2020 -0400

    Started refactoring env_factory

[33mcommit b71f2a76cd662d341cbc61601854027ed6646596[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Sep 11 15:17:59 2020 -0400

    Made a couple of small changes to the pybullet object classes and the corresponding urdfs

[33mcommit dee84c4df7d8f09bee3d6e79c649537532c59d39[m
Merge: 42efd25 6e712fa
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Sep 11 14:49:00 2020 -0400

    Merge conflic fixin

[33mcommit 42efd25deafdf964dbe78ddf2e817ad4d3230cec[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Sep 11 14:48:25 2020 -0400

    More bug fixing

[33mcommit f21f93c96eb2eecc9158ea135e7402bc836f3b54[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Sep 11 14:47:37 2020 -0400

    Fixed picking issue when running fast mode causing invalid envs to happen often.

[33mcommit 6e712fa6797427dcf4dde57841adfb47611932fb[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Fri Sep 11 14:20:36 2020 -0400

    update brick and roof sim params

[33mcommit 5fe27fb0f40345ab65de4f6472c932bfad6c356c[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Fri Sep 11 13:50:15 2020 -0400

    _getHeightmap missing; I added it

[33mcommit 696edfabff9e29e2821a8014735f464aa91bb9b0[m
Merge: 0d4b6d4 e3179b1
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Thu Sep 10 14:45:30 2020 -0400

    added stacking grammar construct and deconstruct envs, modified rl_runner so that I can return metadata for each time step and modified pybullet_deconstruct_env to return handles of created objects

[33mcommit e3179b12a928dba3c6596ac04639a274d31cc952[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Thu Sep 10 14:26:22 2020 -0400

    Revert "merge master"
    
    This reverts commit a0ddd16f3ef64c728e0a59f02d301eb94ccdc00b, reversing
    changes made to 060da11312ddcd47c2f866232987775499dba403.

[33mcommit a0ddd16f3ef64c728e0a59f02d301eb94ccdc00b[m
Merge: 060da11 e2c4738
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Thu Sep 10 14:17:14 2020 -0400

    merge master

[33mcommit 00c636f0fc72a2533def41fbaad6d53598e745c4[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 7 17:24:35 2020 -0400

    Removed block cylinder stacking env, started to refactor env creation.

[33mcommit 0d4b6d4f58face96b5910041d36c43803590e76a[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 7 15:33:40 2020 -0400

    Fixed issues with my sensor refactor.

[33mcommit 5e0377a836bb8c5911318e107afaf56aa47b8217[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 7 15:27:18 2020 -0400

    Removed old robot files and moved sensor to seperate class.

[33mcommit f6d6527d84e9beaa9f4dc8a8fc73b5a4b037c4b4[m
Merge: acde1a2 ea1d1f3
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 7 12:50:00 2020 -0400

    Merge branch 'merge_refactor' of github.com:ColinKohler/helping_hands_rl_envs into merge_refactor

[33mcommit acde1a2032f9c750017d3d2f7d2efb4772a938fe[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 7 12:49:54 2020 -0400

    Small changes to play planner.

[33mcommit ea1d1f37ebeacd622caa29c1655338beba2c6466[m
Merge: e96cb56 3564f73
Author: Colin Kohler <colink78@gmail.com>
Date:   Thu Sep 3 17:53:13 2020 -0400

    Merge pull request #16 from ColinKohler/dian_z
    
    Dian's changes part 1

[33mcommit 3564f73bf586e6b651d3028818d373b640e908a5[m
Merge: b8158e4 e96cb56
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Sep 3 15:47:02 2020 -0400

    Merge branch 'merge_refactor' into dian_z
    
    # Conflicts:
    #	env_factory.py
    #	envs/base_env.py
    #	envs/block_stacking_env.py
    #	envs/pybullet_env.py
    #	simulators/pybullet/objects/cylinder.py
    #	simulators/pybullet/robots/kuka.py
    #	simulators/pybullet/robots/robot_base.py
    #	simulators/pybullet/robots/ur5_robotiq.py
    #	simulators/pybullet/robots/ur5_simple.py
    #	simulators/urdf/object/cylinder.urdf

[33mcommit e96cb56a9192ff31564f85d19ab4ed16998fc821[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Sep 2 16:18:12 2020 -0400

    This is one of those commits where you waited to long and you can not remember what stuff you did.

[33mcommit 060da11312ddcd47c2f866232987775499dba403[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Mon Aug 24 18:09:21 2020 -0400

    do not check roof upright

[33mcommit fae1a7ce4c7de20860cc8d9bc127425138298c6e[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Fri Aug 21 18:20:54 2020 -0400

    bugfix

[33mcommit 1a9621eb83983626e2ceb19af1e722c807e493dc[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Tue Aug 18 18:30:11 2020 -0400

    fix labels

[33mcommit 997930c089615d089e350b695ab38768c8295a4f[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Mon Aug 17 15:25:07 2020 -0400

    custom labels

[33mcommit a9d08fb701dd877346f23b08af63e012cde41b44[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Aug 14 12:48:33 2020 -0400

    change tilt envs to 6d

[33mcommit 7ba2c95f127c045b45e2f37b62ef989ed1bdee80[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Thu Aug 13 16:31:10 2020 -0400

    change object gen

[33mcommit 3112138f2648713da2f204a5d55975a9bf9133bb[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Tue Aug 11 14:54:27 2020 -0400

    comments

[33mcommit ca4932be07f90b4e636a74f4bb086468b52ced29[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Tue Aug 11 14:13:17 2020 -0400

    change resets

[33mcommit e9761ebd8983c610caf7b2e75b065e720bc694ae[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Fri Aug 7 17:24:33 2020 -0400

    control num objects generated

[33mcommit 3833d7467565693129bcd0118fc0abc3e424641e[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Thu Aug 6 15:14:44 2020 -0400

    fix workspace size

[33mcommit f111f37bdbbb250ca3ae9d35948f6b0c2137af9a[m
Merge: a93a388 7512278
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Tue Aug 4 18:37:02 2020 -0400

    Merge branch 'dian_ondrej' of github.com:ColinKohler/helping_hands_rl_envs into dian_ondrej

[33mcommit a93a388cd28df9506dc6aacce4902cf643b63737[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Tue Aug 4 18:36:59 2020 -0400

    fix roofs

[33mcommit 7512278f4c7e83f5ec3235b5a4c3be92026962f0[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Tue Aug 4 18:15:40 2020 -0400

    check goal

[33mcommit 6c2f672807cd71e1e046d928fb072aaba5a58e65[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 22 10:51:57 2020 -0400

    add min distance to tilt of 3cm in imh2 and imh6

[33mcommit 8b71377a34954cc92c14a8b4e05ea0cb8c1ccc31[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 22 10:12:59 2020 -0400

    add random brick that takes x_scale, y_scale and z_scale. Increase the size of objects in imh2 and imh6, use random brick in imh6

[33mcommit e566ca484e4e90f58b8dde3a6dc0d7bef347215f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 20 21:48:48 2020 -0400

    add imh2 and imh6

[33mcommit ac2e7830cf4ff65df6d106b5ca52d6849915b2be[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 20 13:46:42 2020 -0400

    minor change in imh5 parameters

[33mcommit c68647d60f84aab80dcbb0e767ce998c2b53a8d7[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 20 10:45:53 2020 -0400

    use cube instead of mesh for random object collision checking, switch back to previous physic parameters

[33mcommit bd7b77f80e91b2ae6f8e0d6f595bef70f9ff4e5e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Jul 19 18:38:52 2020 -0400

    add imh5 with colin's physic parameters

[33mcommit e432f5b486a65fb88982c547097ee07145a2d9a2[m
Author: Ondrej Biza <ondrej.biza@gmail.com>
Date:   Sat Jul 18 18:23:02 2020 -0400

    goal string

[33mcommit bf7e494fbe49d7d48c7ac53b05e65fbea88cce40[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jul 16 11:04:19 2020 -0400

    working on imh3

[33mcommit 15d9294a0d9ca993ab7624bfb2956874050c6292[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jul 14 17:36:30 2020 -0400

    Adding back in block scale range now.

[33mcommit 441ab2591c9af743f73754672dbff63ab7469cce[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jul 14 17:29:00 2020 -0400

    Trying to optimize some parameters for speed now.

[33mcommit 67fd381ba4fbc4aebad4f2f5cac6175e24214770[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jul 14 17:12:32 2020 -0400

    Removed extra wait time while generating objects that is no longer needed.

[33mcommit d6b295879d5e8fa0da7a5f1d58260afdcfd3c6c3[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jul 14 17:03:25 2020 -0400

    Lots of small changes to get more stable simulations, seems like this was mostly due to how the kuka gripper was opening/closing. Current fix is to sse the stabalize the block while the gripper opens and then allow dynamics to start effecting it again. Will need a better solution going forward however. Additional changes to stabalize the env may or may not be needed but it seems good to have them in so they are staying for the moment.

[33mcommit 05384bbee6ae76eb5d1305d94f23f89cce1bbc16[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jul 14 10:59:26 2020 -0400

    working on imh3

[33mcommit af561fc86f388cd0df395ca40b45c70b8ff2b554[m
Author: ColinKohler <colink78@gmail.com>
Date:   Sat Jul 11 01:30:27 2020 -0400

    Attempting to fix block stackinging physics inconsitentcies

[33mcommit 18439ec3c02ddc3d719b4714b5c3ab026a654a1e[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Jul 10 22:31:03 2020 -0400

    More attempts to stabalize the simulation

[33mcommit c528e371cc4923d413d6a089f37cba908b911d79[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jul 9 10:28:52 2020 -0400

    add stack deconstruct

[33mcommit 3f1800449a322e98e608191b0639d6c59b7f3f8b[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jul 7 18:36:59 2020 -0400

    Fixing issues with multi-task env construction due to shared memory and copy/deepcopy issues.

[33mcommit 97af41ce59398d4140e3212d0b68f5896678637d[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Jul 6 14:39:02 2020 -0400

    New cube urdf to help edge cases

[33mcommit 45b3ea1cbfd1a350f93d180f21071ec2ed4b427a[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Jul 3 16:14:00 2020 -0400

    Changes urdf block to be more realistic

[33mcommit e0307632e3a5fadc3b02c8bfe854f872f48ad2b7[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jul 2 16:38:02 2020 -0400

    1. randomize tilt boarder and tilt height; 2. initialize (and place in deconstruction planner) object either on tilt or off, eliminate situation where half object is on. 3. fix some triangle bug

[33mcommit ffaed2ed6956fc47231932fce9feaae68788fe03[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 1 12:20:57 2020 -0400

    lower kuka tip pos s.t. pick_offset=0

[33mcommit 51f14554ee3582f741a1f72a947703ad8c99b5ba[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Jul 1 02:26:16 2020 -0400

    Small changes to the pick/place offsets.

[33mcommit 31d1175bcb89e01dd843efae1cc01acec9ab47ce[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jun 30 12:30:44 2020 -0400

    Changed planner x/y pos noise to a explicit range in order to control the amount of noise better than the larger open interval it was before

[33mcommit 017e2005c6b81a55154d2cd48459bcb703c6ee16[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 30 11:56:37 2020 -0400

    try adjusting gripper command right after close to solve pick too deep problem

[33mcommit 7f3b67c3f1c85affe7702104643628a186334bc0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 30 11:55:48 2020 -0400

    switch tilt decon place planner back to using uniform bigger dist and padding

[33mcommit 10d1eb05e5a5acb8f1df97f3386b75df9421ffec[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Jun 29 16:06:41 2020 -0400

    Changed joint limits on kuka arm to -10,10, changed pixel action from rounding from floor to true rounding.

[33mcommit 10049f7976542e99f33d2035d27441f9cee1f4ea[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Jun 29 15:08:18 2020 -0400

    Switched to orth projection for depth sensor

[33mcommit f3dcaaef32d16e733af3cd5b608d49e1d34fbe0a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jun 29 12:27:23 2020 -0400

    add tilt H3 and decon; fix a reset problem that might cause infinite loop; initialize one cube on ground in H4 and H3

[33mcommit c6f1940a07af99a355156b374dc63dc0eae10502[m
Author: ColinKohler <colink78@gmail.com>
Date:   Sun Jun 28 23:26:52 2020 -0400

    Changes to depth sensor to fix offsets in image/real world

[33mcommit eba9809c20eef3185eb62b717cde0a70a8f0e8a8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Jun 28 17:11:38 2020 -0400

    always initialize one cube on the ground in H1

[33mcommit b8158e4ec791ea2046ac2a22b86e04138d66ffec[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Jun 27 12:21:48 2020 -0400

    fix init orientation when random_orientation=f

[33mcommit c069f3cab528ab4d0b79886f19a5519c5c946006[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Jun 27 12:08:02 2020 -0400

    depth bug fix 2

[33mcommit f355ff5efbb95c13e0d36702e2c747cef6eb5d35[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 18:52:03 2020 -0400

    depth bug fix

[33mcommit ff6ff9e0443aa44c05d5103149538702da2020a9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 20:48:29 2020 -0400

    add fixed ori h4 de

[33mcommit 3d9ac5ade6c483185ef2a2025dda29d668df3184[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jun 1 10:47:24 2020 -0400

    remove 'side_grasp' and 'side_place' in planners, change the object urdf s.t. they can be picked up at 0 orientation

[33mcommit 150312b045429b55ccb7a53c039f539cb5a70839[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 18:52:03 2020 -0400

    depth bug fix

[33mcommit dbf06c71f025301bf5b6a396723b8ad9be453f67[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 17:28:18 2020 -0400

    change camera matrix to move camera higher up

[33mcommit 5842b96dbce8fb919502d5137e4e44e12e476d2f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 17:28:18 2020 -0400

    change camera matrix to move camera higher up

[33mcommit f9802e3fd25487a8beac312303d19cda745cd42f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 17:20:37 2020 -0400

    rescale rz to [-pi, 0] in decode

[33mcommit 07afeb2e4c860cf90309ede3ccbaf6153ee8bef0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 12:27:32 2020 -0400

    half rot in decode action

[33mcommit 83ab53358a66c9e91dcc772715b3c6b585fa86fc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 26 12:27:05 2020 -0400

    modify H4 deconstruction s.t. 50% of the chance all blocks will be moved, 50% of the chance the last one is not moved.

[33mcommit d7134e2450e5fb2a54b35d83674a5c3d87e93c1c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 25 19:24:00 2020 -0400

    raise the center of triangle

[33mcommit ed8e8628495036d086c68a3aa1250a0ef8663ab4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 25 14:53:18 2020 -0400

    only switch z offset when z is not in action_sequence; modify H1 deconstruction s.t. 50% of the chance all blocks will be moved, 50% of the chance the last one is not moved. When placing the last block, always place it on tilt; change some deconstruction parameters

[33mcommit 903092ec481b8a3322e9082ce912eeaddbd81727[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jun 23 18:59:33 2020 -0400

    Seperated pick/place planning noise

[33mcommit ea0f75d448564adfc82e967d6919cae8bf7ddf1d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 23 13:13:55 2020 -0400

    implement random tilt rz (-pi/2 to pi/2)

[33mcommit 6934c6853aa92baefc568e54c9531e12fe115235[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Jun 14 01:13:27 2020 -0400

    add tilt_imh3 and tilt_h2

[33mcommit fc306235b38c0959254f2bce9c394a4f09ae103b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 5 21:40:40 2020 -0400

    improve parameters for tilt H1 and tilt H4

[33mcommit 598c4667c51b29588bbba2b8c3baab577281aa24[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 4 14:08:21 2020 -0400

    fix z in tilt deconstruct planner

[33mcommit 7ae78f466399dbc0113eb427c91a2abea0dbef9b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 3 19:46:11 2020 -0400

    fix wrong random planner action format

[33mcommit fef94cc2ee764730a425faefe820c2b0a3995cd2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 3 13:54:20 2020 -0400

    add tilt H4 and deconstruct

[33mcommit 028deb04d9232d0146c07202828aff111a5b9772[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Jun 3 01:46:58 2020 -0400

    more reseting

[33mcommit 1f1846d0adad658749ea45139ed8aa5037158a87[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Jun 3 00:39:54 2020 -0400

    Removed init/reset junk for now as it breaks on discovery

[33mcommit 07c1b9c271aec8af26884bc6871c02337de2958a[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jun 2 23:09:14 2020 -0400

    Init/reset changes.

[33mcommit d4aaa9fb618c5c353dda4be65fb3ddff801ca621[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Jun 2 17:26:15 2020 -0400

    Changes to get kuka arm working

[33mcommit 637f905f1b09cb80dc19a304984609893c451262[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jun 1 10:47:24 2020 -0400

    remove 'side_grasp' and 'side_place' in planners, change the object urdf s.t. they can be picked up at 0 orientation

[33mcommit 36a5d182159030f8e483fd585000f75ee59859fd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun May 31 19:06:43 2020 -0400

    bug fix

[33mcommit 31c6245bd980d7e2396b8aa03e6e656b97458ec8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun May 31 11:18:03 2020 -0400

    add tilt_house_building_1 and deconstruct

[33mcommit f4edc636edef2114f82220c9734e75097a642fcb[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 29 14:33:18 2020 -0400

    commit cylinder urdf

[33mcommit bf8162449a6f70c982e1306fbc4d8d73048aff2c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 29 14:31:29 2020 -0400

    add cylinder in pybullet, use cylinder instead of triangle in H5

[33mcommit 6f35a5af5e2a2a18be11b1e71667587f28cd3ded[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 29 12:24:41 2020 -0400

    add H5 that requires the agent to build multiple 'triangle on cube' structure

[33mcommit 7ace8461f815586f1547a9d6891f00f4f2d0fcf0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu May 28 16:13:07 2020 -0400

    add another slope

[33mcommit 0ad10d716148f2ff2f5f3a9257258a64998af439[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 25 15:41:57 2020 -0400

    include ground in proj

[33mcommit 4681a011b56f6130f0e2a6e671fd825f696d4627[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 22 16:11:07 2020 -0400

    finish tilt block stacking env

[33mcommit 047b10da2d671d6b254560ce74926206e84a7b31[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu May 21 19:10:13 2020 -0400

    working on tilt block stacking env

[33mcommit 131a50faf1acfd59f3484fc99bec52057c2c0212[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed May 20 19:33:23 2020 -0400

    fix bug in _checkStack where aligned blocks might be valid

[33mcommit 3c01e3f8aa650c30277d9ef669e11a9fe513457f[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed May 20 12:29:00 2020 -0400

    Working commit

[33mcommit 354ef3a3d270a3496f5dacc3677eda4828737c16[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 19 23:52:05 2020 -0400

    add block_placing_env

[33mcommit fa307e22291574eacb4ddfbd25fa6cabfe4e65e4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 19 14:48:59 2020 -0400

    return empty in hand img in picking envs

[33mcommit a46d69ecfa0610a7e8ded2c8026bf0cd31475075[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 18 23:02:21 2020 -0400

    fix _removeObject

[33mcommit 3abeb7ec55b3035f9954367ee83888f2cd531895[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 18 21:09:08 2020 -0400

    changes in test

[33mcommit 32c042905fddccec1fb9fd5c94cb9ef5b79fc3c9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 18 17:56:19 2020 -0400

    add no valid position exception handle in block stacking

[33mcommit c2b65dc416ec1f7c1de060920c09139a51078898[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 18 11:18:01 2020 -0400

    change rx_range in cube float picking to [-pi/8, pi/8]

[33mcommit 798ae6937fa63315642183946707b802e8aefb95[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun May 17 18:10:17 2020 -0400

    add kuka_float_pick to check T before picking up; fix bugs in cube_float_picking_env

[33mcommit 89048912843389d1dac404f5420149f4b523a110[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 15 00:33:13 2020 -0400

    add rx by changing rot=rz to rot=(rx, ry, rz) in env classes; add cube float picking and random float picking env

[33mcommit 890d1633a169834ea3c0e38168f0486a17a2b738[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed May 13 15:52:14 2020 -0400

    Added exception for invalid env/planner types in the mutli-task envs.

[33mcommit 42d81116dc5b03d8898f5e9abb37413a7473ae59[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed May 13 15:50:27 2020 -0400

    Working commit

[33mcommit 97243dd0defcb670392b04d56c125bbdb196f682[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed May 13 11:20:48 2020 -0400

    increase joint limits and ik iterations in kuka to solve the problem where kuka can't reach target; change pre_pos in pick/place to match the rotation

[33mcommit 9f3e3dde5836a74f7fe43c32ad41feead195aea8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu May 7 14:16:15 2020 -0400

    vectorize getInHandOccupancyGridProj

[33mcommit 6f207864275030be4a9cbf7795e45dc8bed54177[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed May 6 17:26:47 2020 -0400

    add in hand image occupancy grid projection

[33mcommit 4defd4180af8979176447983db8069fa02202c83[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 21 16:25:52 2020 -0400

    renaming

[33mcommit ade89093ad01d5fcd5f6d75ca872af36da871762[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 21 13:06:41 2020 -0400

    add H4 deconstruct

[33mcommit cad9511a669b1a8670842f51be2c71d524d8ee40[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 20 19:18:14 2020 -0400

    add H1 deconstruct

[33mcommit c5c0c602ea10d5d6da2b32d7aece35a23a3adcb8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 15 15:48:38 2020 -0400

    fix bug where deconstruct planner may pick non-structure obj

[33mcommit d797f008f799ee7479226abe151f9d14787d21ea[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 15 15:39:00 2020 -0400

    spelling

[33mcommit bcb13f0f7ef82ad0e82980fad7ea85e008fd4bb1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 15 15:37:31 2020 -0400

    fix problem where improvise envs randomize obj scale range

[33mcommit ecda814b3708b17496dbe119ef560d943ae73524[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 15 15:06:45 2020 -0400

    generalize deconstruction planner

[33mcommit 1394c103af77af12593c4e42e20261c09519c6b9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 15 15:03:22 2020 -0400

    remove three objects from random set; add random stacking env to test the stackability of random set; change _checkStack criterion in pybullet env

[33mcommit 9bd1d8dc19d1b0fc7f2a3c581ed2eb63d9d2d6ca[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 15 12:16:13 2020 -0400

    remove more 'hard' objects

[33mcommit 27b4697884d1da98725bc723d77a08785ca8917b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Apr 4 18:12:49 2020 -0400

    use easier obj set (objs that are more flat)

[33mcommit 6297a88ea04a9617de4bb030026b306568635bfc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 14 15:59:50 2020 -0400

    implement imh4 and de

[33mcommit e85c9941d07b67a62bcf4056825970a01cba68db[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 13 16:51:55 2020 -0400

    use createMultiBody to create random_object so that it's possible to change the z scale randomly

[33mcommit b1fb28bb10a0abe835905d52824cc05d212abda0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 9 15:27:14 2020 -0400

    working on imh4

[33mcommit 1c30bdfe7507e5e167c55c14bd794fc7d972852b[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Apr 10 14:44:53 2020 -0400

    Added getValue to multi task planner.

[33mcommit dfd7dcbd9779f7b35fd5e8b0e0d92fafc1eb4654[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Apr 10 14:18:41 2020 -0400

    Multi task + rot is doing weird things so removed it for now

[33mcommit 445ebe0745a3c1ae581c8ff88b45d6d0e6ccf649[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 9 19:09:12 2020 -0400

    adjust gripper command AFTER ee is moved to pre grasp. Testing on kuka

[33mcommit 93c7453423a3370d986441771cc31384bb0a91cb[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 9 15:26:01 2020 -0400

    update test

[33mcommit 07ee62cbde6d9c5dd1f1bf98a300ed3f86af1ebd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 8 16:29:37 2020 -0400

    add random pick env

[33mcommit a5e5303619da0d2f21c152d89e7ef38ccb6eaa06[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 8 16:27:37 2020 -0400

    Revert "use easier obj set (objs that are more flat)"
    
    This reverts commit 17f05887

[33mcommit c3223dc823c96b7312190e8a10694f7bb061e6d4[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Apr 7 15:05:05 2020 -0400

    Small termination checks and fixed weird bug w/multi env planner get object poses.

[33mcommit 56caab27ab29faceb9696a792bb40fe01fd5efb8[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Apr 6 17:42:58 2020 -0400

    Multi task env implemented. Needs refactor, block adjacent planner not optimal.

[33mcommit 17f05887d6166b1904d69a41efcfc5348e6c98ec[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Apr 4 18:12:49 2020 -0400

    use easier obj set (objs that are more flat)

[33mcommit b4b0ae91f4272856c7771cb2cc03a24271303974[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 31 17:11:49 2020 -0400

    fix bug where place orientation is always 0, and minor change

[33mcommit a7983884a8763690fe5a8511dcc2325d22b637e7[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 30 15:35:15 2020 -0400

    add base_objs list in deconstruct env; add random object outside the structure to meet with the original env

[33mcommit 812fe1deb90121ba9fdcedeb4b76b3e689a3fab5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 27 16:39:14 2020 -0400

    imh3 deconstruct

[33mcommit b83513008099432539eda43334c95aef183f10fc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Mar 26 09:54:32 2020 -0400

    add imh3deconstruct env

[33mcommit 4e67beaa26ec3e901a85778ad145aabc507f71ae[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Mar 26 11:24:42 2020 -0400

    reset each 1000 episodes

[33mcommit 749038538824eaccf470a15afbdbb2bb7ab99a0c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Mar 25 13:14:04 2020 -0400

    hard reset after 10k episodes

[33mcommit c80aa87ff84dcdb6758162f9d2ff88f91ff50338[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 24 11:11:36 2020 -0400

    decrease the step wait time of imh3 to 100

[33mcommit 3a544aa772244d279650615c199d44ccf18e2207[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 20 11:50:14 2020 -0400

    decrease the xy extensions of random objs

[33mcommit 1c4950f93426f77efb7145af8b25bca99d005a92[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Mar 19 22:08:50 2020 -0400

    decrease roof height threshold

[33mcommit 58b1514030102b5fb322ff947b3995beae6b52bd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Mar 19 16:30:01 2020 -0400

    fix bug that putting roof on 2 lower objects could meet success checking; decrease min safe z pos for better picking lower objs

[33mcommit 26c857eacab7f7645510cfaaa8f0e2f38620d3dd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Mar 18 15:09:50 2020 -0400

    change the z scale of random objs; add check_random_obj_valid flag

[33mcommit 43418d9de1e86fb789ebb8075f2820a3ea850bb2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Mar 18 12:27:30 2020 -0400

    remove some objects from RANDOM set

[33mcommit b28a2f1a8503702c3dd1d41e84f41c45d644e9ac[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 17 17:47:52 2020 -0400

    add improvise h3

[33mcommit 9aadde4e788e98015e0ae47e05c0267aa97b3872[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 17 17:46:29 2020 -0400

    fix a bug in in_hand img that might cause the in hand image to be cropped smaller than the required size

[33mcommit 4d3c90d2c633296b190f8fc0b794017ba72a6790[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 17 12:03:22 2020 -0400

    only use plane obj in random_obj

[33mcommit ee146b53593f8df58fa3958b9019dbb48ff5601c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 17 12:02:45 2020 -0400

    fix bug

[33mcommit 3c662f99cecf8bede3ba7adda4fac03da860f4a3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 17 12:01:51 2020 -0400

    add side_place in placeNearAnother; increase distance threshold in improvise_h2_planner

[33mcommit bf34e1333c0b3921bb9b12ef64907c927515a616[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 16 20:49:43 2020 -0400

    decrease the height difference threshold for checking on top

[33mcommit eefa0a77d42b6d64c7dc7243d39e02e3ace8630a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 16 11:47:57 2020 -0400

    add improvise_house_building_2_planner, fix getStepLeft in h1 when there is random objects, test envs with 2 random objs

[33mcommit 348a52217466475c2b522a8a9a8041ffb228681f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 13 20:12:36 2020 -0400

    minor changes

[33mcommit 089e7b6795e6c0f55d67750a5558901d6b06ce74[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 13 16:24:29 2020 -0400

    add 1000 random objects; add 'num_random_objects' in config to add those random objects in the scene. add improvise house building 2 that need the agent to put a roof on top of two random objects

[33mcommit e2c4738accdbcd267ba6fe82b73f6f0fc98e2872[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 21 16:06:59 2020 -0500

    Fiddling with parameters

[33mcommit 7d633eae602f47dffe3db34fcb0ad366eb5d6d7c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Feb 21 14:40:28 2020 -0500

    extract last_obj from last_action in pbenv, add getObs to get current obs using last_action

[33mcommit ba8b38d9cbaf835624daa258f7a702424535f2ca[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 19 11:31:00 2020 -0500

    change the criterion of _isObjectHeld in pybullet

[33mcommit 64d55cb2e3f3f382964112047341034f06341358[m
Merge: dce8db6 9b81e04
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Feb 18 13:52:12 2020 -0500

    Merge branch 'ck_old_camera' into pyramid_stacking_env

[33mcommit dce8db68ae65a13977756e43b3e098c865fadd07[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Feb 18 13:51:16 2020 -0500

    Added notes on forward model tree based planning.

[33mcommit e2300b63962708ecd8c50cb40b63622c8eb70c98[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 19:44:52 2020 -0500

    decrease kuka gripper open angle to avoid collisions

[33mcommit 6179fe90268ae1a72c969984cccc9023cb464c34[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 18:11:06 2020 -0500

    decrease max_out_id in _moveToCartesianPose to 10 to faster simulation

[33mcommit fe59a7a316996b0baf244e793b8d4497ee600303[m
Merge: 09e4265 f72fd87
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 15:47:25 2020 -0500

    Merge branch 'dian_ur5_robotiq_param' into dian_devel

[33mcommit f72fd87376748f81a62c1f4ac484a0a0a77bb2c2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 15:46:45 2020 -0500

    restore the changes in place_offset; decrease finger tip width to 0.02

[33mcommit 09e42656231d156133f0d3f7434de50adab4557e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 14:58:32 2020 -0500

    decrease pre offset to 0.1 to improve ik

[33mcommit 5acf09a0314b757ef985df5238e34c587218441c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 14:51:50 2020 -0500

    Revert "update parameters to improve performance of ur5_robotiq (mainly tested in 5 layer house1)"
    
    This reverts commit 168ffbee

[33mcommit 633dbb0f2bdd747114b27e23b618e395ac9666b6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 17 14:47:23 2020 -0500

    changes that improves the performance of ur5_robotiq

[33mcommit 168ffbeeace4106fc0daee9c15fc8eac3eef55ab[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Feb 16 18:00:00 2020 -0500

    update parameters to improve performance of ur5_robotiq (mainly tested in 5 layer house1)

[33mcommit 90e5fde740ece4245e029ae99f55107b0d39f4f7[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Feb 15 14:37:48 2020 -0500

    fix bug in save and load

[33mcommit 2e7de5f8d1d64f7d25bd7a7e1902bb14355c3c83[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 13 17:09:47 2020 -0500

    only initialize env (load urdf) once. reset() will not reload urdf anymore

[33mcommit a810775d3d0a171f339c50345997f2576f3302ec[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 13 14:09:18 2020 -0500

    add exception handling for loadFromFile

[33mcommit 9b81e04d0a9c40532b2918df05314965e9ebc6ac[m
Merge: ff17e67 e7f41af
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Feb 12 14:40:38 2020 -0500

    Merge branch 'master' into ck_old_camera

[33mcommit cf9430b758424c0e3738b6fca09e2fd173c1d496[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 12 14:10:11 2020 -0500

    set robotiq position at each iteration

[33mcommit 7be08478032455599bd3994ddbff66e6119c1417[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 19:56:00 2020 -0500

    minor changes

[33mcommit c4abe1b1af5e5c49fb3162be12d09edc02070a24[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 19:12:00 2020 -0500

    use a fake robotiq 85 for visual and use the simple gripper for collision

[33mcommit 8abaede221f5ca012d413c1926f87d50a206b320[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 18:04:54 2020 -0500

    use box to replace finger tip

[33mcommit 95333fc22e3af9af1803c086263d479b83c6a040[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 14:28:01 2020 -0500

    minor changes

[33mcommit 414d51f41c507198d5ded326b64589db41096066[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 14:26:38 2020 -0500

    select the most isolated block to align in H4 planner

[33mcommit 2817c1dd7295c5cb95efb55a3273b8fe7a230017[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 14:25:34 2020 -0500

    change max loop in placeNearAnother to 100

[33mcommit 2b3e9d7bc473020d7eed93a5c175853a20828c7e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 14:24:30 2020 -0500

    Revert "only try to get valid position one time to avoid looping."
    
    This reverts commit 191b6c48

[33mcommit c8999d649c9b71da8112ebae2e155ecff90fe8c4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 11 14:22:24 2020 -0500

    remove the outer loop in _getValidPositions. Callers can use NoValidPositionException to handle situations where there is no valid position. The outer loop significantly hurt the running time of planner

[33mcommit e7f41afcd439d8eb43d8cd2ff29802a6e3e92e05[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Feb 11 00:40:56 2020 -0500

    Fixing bugs introduced in planners while refactoring.

[33mcommit 001e9b74fa6c297673d502567d80f88767dad8bc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 10 18:48:04 2020 -0500

    add 'in_hand_size' and 'in_hand_mode' config in base_env

[33mcommit e86fd5e1f59fc86a3a742688b79b4a17ca7bcbbe[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 10 15:07:07 2020 -0500

    add test bullet house 4

[33mcommit ff17e67aacd1265e7896ca697445ef595c2504bc[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Feb 10 13:06:46 2020 -0500

    Changed camera heightmap back to transpose while debugging.

[33mcommit cf9df157d49c9bf1c103c66089050bc008f40ee3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 6 18:43:01 2020 -0500

    add house building 4

[33mcommit 191b6c48de858e63728e496d6ffc9e2592156353[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 6 18:42:52 2020 -0500

    only try to get valid position one time to avoid looping.

[33mcommit 8fc0bc6ff1ff63d5a5476512e08459eb34591e82[m
Merge: d6d33c7 a32e493
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Feb 7 16:31:32 2020 -0500

    Merge pull request #12 from ColinKohler/bug_fix
    
    fix bug in in_hand img

[33mcommit a32e493e78d4c394517c827e0d7531390db12038[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Feb 7 16:29:03 2020 -0500

    fix bug in in_hand img

[33mcommit d6d33c70e31b8d8609fea10e2bb2a05216c81fe3[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 7 14:54:29 2020 -0500

    Removed VREP from master, kept old code in vrep branch in case we need it later

[33mcommit 2ec580d617ff3ab969bee45191ac4494ab76720d[m
Merge: 9414bdc e5ba23a
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 7 14:33:16 2020 -0500

    Merge branch 'master' into cpk_refactor

[33mcommit 9414bdcba9ca5eb7e70622959bcb03e412be6a58[m
Merge: 8db8c08 2d4be15
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 7 14:32:53 2020 -0500

    Fixing merge conflicts

[33mcommit 8db8c0839769f102374bc26912eb7bec28f236e2[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 7 14:29:43 2020 -0500

    Adding steps left getter to data runner

[33mcommit 2d4be15e312625f9d836618c6b6aa7aea99fee62[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 6 16:00:02 2020 -0500

    minor changes in padding and min_dist in pybullet

[33mcommit df5cc766da78cab91c3cc6b9c3295c497f1e2342[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 6 15:26:17 2020 -0500

    fix bug in numpy env, tested numpy block stacking env

[33mcommit cedab6e8fa4c7b53466e64b496a67787420e24c5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 6 15:07:00 2020 -0500

    clean up env code, tested block stacking, brick stacking, house building 1-3 in pybullet

[33mcommit 75ad8cc31b4347800245a62fbbcb6a0d11ae7268[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 22:16:07 2020 -0500

    add 'workspace_check' in config to specify using 'box' or 'point' to check inside workspace. change some parameters for planner and object generation

[33mcommit a091cd292d7ac4fc62fdbf8d9e6ed01bdd079a43[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 20:57:46 2020 -0500

    refactor brick_stacking_planner

[33mcommit 2ae0c0a00bdd2b101303d06f8e353de727219206[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 16:54:01 2020 -0500

    refactor block_structure planners

[33mcommit 5ff2daf7ba945168be38241a785c0a110653dde3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 16:08:44 2020 -0500

    move getStepLeft to planner for house building 1-3, clean up env classes

[33mcommit 2003a8dd88a166da44576a88894aed11d11a4ce9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 15:19:19 2020 -0500

    finished integration of house building 3

[33mcommit 8f23380152b8025da7e9fcae5e01a5c66fc5294d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 13:07:42 2020 -0500

    integrating house building 3 planner

[33mcommit 4707f3d56bc86b28735be3622131ab96dcad245f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 5 12:00:39 2020 -0500

    fix the reversed xy axis problem in pybullet. (numpy not fixed yet)

[33mcommit 37f204048db3ea2e4a02836476ed1795f5456b1c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 4 22:01:00 2020 -0500

    integrate abstract_structure_planner into house building 1

[33mcommit 3ac08697d5715b1c1d5918ae9cd03d576ad0821c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 4 20:54:38 2020 -0500

    add abstract_structure_planner to help cleaner implementation of planners; integrate into house building 2

[33mcommit 01abe3add47f6c396494aa17527378bd44d72ac4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 4 16:08:22 2020 -0500

    integrate house2 planner

[33mcommit 81ee4c1c073257f7e95f5ec299eb9ced575ec127[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Feb 3 21:38:19 2020 -0500

    integrate house1 planner

[33mcommit 55dc72b693f26612ad769e8f726d7fee02bb5d39[m
Merge: 2e656e0 bf2a8c6
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jan 30 18:34:15 2020 -0500

    Merge remote-tracking branch 'origin/cpk_refactor' into dian_refactor
    
    # Conflicts:
    #	env_factory.py
    #	envs/pybullet_env.py

[33mcommit bf2a8c6e7c987a8c3d60725670f378ddd0d28628[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Jan 10 14:57:47 2020 -0500

    Added getValue method to data runner and planners

[33mcommit 4b9591656a44a10a634a78f49f7fa586dbb25869[m
Author: ColinKohler <colink78@gmail.com>
Date:   Thu Jan 2 14:37:27 2020 -0500

    Fixed brick stacking termination checking, added a way to break out of object generation if a good config is immposible, small bit of refactoring.

[33mcommit bd2d84ba61e70048149fa3041b15650be591784d[m
Author: ColinKohler <colink78@gmail.com>
Date:   Thu Dec 19 17:33:47 2019 -0500

    Added planner for brick stacking end and refactored things so that the env works.

[33mcommit 2ca21b5a1edd3a086045734c98cce7e43772c753[m
Author: ColinKohler <colink78@gmail.com>
Date:   Thu Dec 5 11:04:21 2019 -0500

    Moar refactoring

[33mcommit 2e656e0a8b9fa05683730a50aa229865891a9e54[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Nov 19 11:21:34 2019 -0500

    change default orientation

[33mcommit 23da377e5350f735e9d2fbcef41d5d9b0f2ba036[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Nov 15 15:29:53 2019 -0500

    Added hand obs to observation in rl runner.

[33mcommit 0766f9ec9d89f1f2019313d65e393c2b9a680561[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Nov 15 13:29:37 2019 -0500

    Fixed bug in data_runner

[33mcommit 1841374ba3bcc9ec1ff067666532dbed4cf5a39d[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Nov 15 12:58:38 2019 -0500

    Added ability to check if the last placed block fell over in the pybullet env. Added call to the data_runner

[33mcommit 8d467fce5419e8e109f19b5a12d9cdcbd6851ae1[m
Author: ColinKohler <colink78@gmail.com>
Date:   Thu Nov 14 17:42:14 2019 -0500

    Added noise to planners, added valid termination flag to data runner step return, some more general refactoring, I think there were other things here that I forgot as well...

[33mcommit e5ba23a0a03f1b8b4dbad45c88a19716af5a219e[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Thu Nov 14 16:18:39 2019 -0500

    Updated README

[33mcommit ef561a81346d096d91919d8566f1f27530d35533[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Nov 13 16:21:56 2019 -0500

    Added tests for numpy block picking/stacking and modified things so these tests work.

[33mcommit 83c16f51b00d0621656f74cc795977f81f034e81[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Nov 13 15:29:41 2019 -0500

    Removed planning from pybullet base env

[33mcommit 53ded95cdcb69331b60cdbd45e23a3d421a7883b[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Nov 13 15:28:18 2019 -0500

    Added block picking and stacking planners to the planners.

[33mcommit b572f47258e5f3c01c62268d4bc66a4b967829cc[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Nov 13 12:16:31 2019 -0500

    Started refactoring for data collection runner/ rl runner split

[33mcommit cfad1574ac43e9459b6b597aaaf4d8af1a329c77[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Nov 12 16:14:24 2019 -0500

    Refactoring

[33mcommit e1c8c74d9d0e7475c58a553d2619bc7f03df5397[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 30 16:42:49 2019 -0400

    Re-structured numpy objects/object generation.

[33mcommit 5f28a7ee4546a20ea355991e91aace77a9c656bc[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 30 14:49:02 2019 -0400

    General structure refactor, adding basic block construction building blocks

[33mcommit 83ae4363a2b22912dc2d4edebede79bbf33d7348[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Sep 23 21:24:23 2019 -0400

    add exception handle in house_building_2

[33mcommit f506df8a8ac607d4a932cf1c0a0ca37ca8ca19cc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Sep 23 12:15:06 2019 -0400

    add exception handle for failures in shape generation in house_building_1

[33mcommit 0a3dcff720c7588ed70172937ce9321f91bd11ef[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Sep 17 12:26:15 2019 -0400

    test via env runner

[33mcommit 54089ef99a31ce038c621706875f235cdadcbdc7[m
Merge: 1824b6a ec9faa0
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Sep 17 12:08:53 2019 -0400

    Merge branch 'house_building_3' into devel
    
    # Conflicts:
    #	env_factory.py

[33mcommit 1824b6a28ff724d9cbffb629eb5f619d261b9b1b[m
Merge: b3b8039 219e12d
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Sep 17 12:03:48 2019 -0400

    Merge pull request #9 from ColinKohler/master
    
    Merge Colin's master

[33mcommit b3b8039a58513412544509c1a4aac9fec87cd6fe[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Sep 16 17:17:29 2019 -0400

    save heightmap in save/load

[33mcommit ec9faa0ac6c9751fa2e3a053220e48dd17287f0e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Sep 16 21:09:22 2019 -0400

    increase threshold in _checkInBetween

[33mcommit 932d3417dfede8df05bfdeb53b2a559946a672f6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Sep 16 17:17:29 2019 -0400

    save heightmap in save/load

[33mcommit a12cb4c006b96635f715eb9ba3a42f4662df2bd1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Sep 16 17:17:13 2019 -0400

    add some edge case in both planner and step_left

[33mcommit 219e12d76dbe797b56e284d7990cb7bdec9d70bd[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 16 15:02:16 2019 -0400

    Adding cube urdf back to repo

[33mcommit f5118b0c1139405c30242d690e9a07fccc024940[m
Merge: bcf4301 c44aff2
Author: Colin Kohler <colink78@gmail.com>
Date:   Mon Sep 16 14:52:39 2019 -0400

    Merge pull request #11 from pointW/devel
    
    Add kuka arm and robotiq gripper

[33mcommit c44aff27f53dfb566808fd43727f8c993be0006f[m
Merge: 4878913 bcf4301
Author: Colin Kohler <colink78@gmail.com>
Date:   Mon Sep 16 14:51:16 2019 -0400

    Merge branch 'master' into devel

[33mcommit bcf43019fffb684374bf6fe3fc124436a7666c72[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Sep 16 14:43:53 2019 -0400

    various small changes

[33mcommit 787270b4b0976e6775cc1dd10350378252db9961[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Sep 15 20:50:55 2019 -0400

    check if brick/roof position is in between blocks; add some edge case step left in house building 3

[33mcommit 869a36799064fc65865c5ca1e256686b90c67e0b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Sep 15 18:43:02 2019 -0400

    add house building 3 to factory

[33mcommit 0e0c4136af86522c2fbc0be829222c3ef2b532b4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Sep 15 18:40:31 2019 -0400

    add NoValidPositionException in Pybullet to handle infinite loop problem in shape generation (now only using in house building 3); increase threshold in blockPosValidHouseBuilding2 to 1.3 to 2.2

[33mcommit bdd2bd888b1f40133eb50a9f6070a3feabeb5b03[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Sep 13 22:56:30 2019 -0400

    change block color to red
    
    (cherry picked from commit 30871000af5eb50e231f711adf124e879f58cac8)

[33mcommit 30871000af5eb50e231f711adf124e879f58cac8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Sep 13 22:56:30 2019 -0400

    change block color to red

[33mcommit 8e8257e17591c9e736cec182c549e54227cdc05b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Sep 13 18:15:17 2019 -0400

    toward implementing house building 3

[33mcommit 4878913950d622b3b6a8d37f2f680cae5a0b7772[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Sep 13 15:27:34 2019 -0400

    move config setting into pybullet and numpy class. task env only pass config into base env class

[33mcommit 2dc56304958a101afd20f51639289a989d8d433d[m
Merge: 751cc80 1cac7be
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Sep 13 15:25:55 2019 -0400

    Merge branch 'devel'

[33mcommit 1cac7be37878e71835d38b718ffa28330135b11f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Sep 5 21:06:45 2019 -0400

    fix bug in save/load

[33mcommit 23aff63e5861ea49aa1809ee762b434153df518c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Sep 5 15:42:35 2019 -0400

    fix bug in getEnvGitHash

[33mcommit b9c7bb1de29055e40681953db0174209ee7a6e92[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Sep 5 12:35:55 2019 -0400

    add random state to save/load; add getEnvGitHash

[33mcommit d2b242f695072cda9866370706066983e50c9f85[m
Merge: 7fa6c7c a130577
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Sep 3 16:44:53 2019 -0400

    Merge branch 'devel' into house_building_2
    
    # Conflicts:
    #	envs/pybullet_env.py
    #	pybullet_toolkit/robots/ur5_rg2.py

[33mcommit a130577b6bfd24461f52481cb70994dbeee40fa8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Aug 30 13:07:36 2019 -0400

    make stepAsync and stepWait public functions

[33mcommit d166e1f8df354636f0027d9d0ee0637c08a870f9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 29 20:50:24 2019 -0400

    increase min dist

[33mcommit e7285c60335a1b870293c1188f20f976e87907ba[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 29 17:51:48 2019 -0400

    add env checkpoint saving

[33mcommit f8c28030153e0fe09586d8b026adf644dd096def[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 27 17:38:41 2019 -0400

    set final gripper pos using the middle point; minor changes

[33mcommit 469a94996669541f44375cfd07621a78001b64da[m
Merge: 9bd7c25 2e3e0e2
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 27 12:14:44 2019 -0400

    Merge branch 'robotiq' into devel

[33mcommit 9bd7c25e14cf61f27efca747a2307864eeed8d39[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 26 21:04:49 2019 -0400

    change collision mesh in triangle

[33mcommit 791c0a6b2adbfb1416fa5c99ceb025fecac3d7fa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 26 19:47:07 2019 -0400

    Revert "fix bug running on discovery"
    
    This reverts commit ffd657f7

[33mcommit ffd657f754a735a24b9a76c5ce3e823fbde51c01[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 26 17:00:43 2019 -0400

    fix bug running on discovery

[33mcommit 8ab5c5ec8815b45fcb559d7e058f39ede74b2256[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Aug 25 14:49:06 2019 -0400

    improve gripper command

[33mcommit 4e538327d1ceb60dc7634b002c2f746330ea2a52[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Aug 23 12:13:50 2019 -0400

    improve gripper command

[33mcommit 7fa6c7c1522f042a71e553f4b9e72843f81a8405[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 22 18:08:58 2019 -0400

    toward implementing house building 2

[33mcommit 4fa51c092de39132c2ae1ae77c08e71e8a1f2e63[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 22 14:39:48 2019 -0400

    toward implementing house building 2

[33mcommit 1a49964da54d09c590a7b07cc31e72f9f3f33d87[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 22 13:57:32 2019 -0400

    toward implementing house building 2

[33mcommit 1c417153a90fe1673e6f438b1d2a0d07ca0e11aa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 21 21:39:57 2019 -0400

    toward implementing house building 2

[33mcommit 113a9104591924ae9f5855bfb5ecf22fb4683809[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 20 16:09:15 2019 -0400

    fix unstable grasp caused by small closing bias

[33mcommit e20bf3490168ffca2f12e6d2f8ef38e2f7328339[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 20 13:26:39 2019 -0400

    increase threshold in _checkObjUpright

[33mcommit 018be6d8a6ebde230681b7108162e1d45e422137[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 21:33:23 2019 -0400

    reduce min distance in object generation

[33mcommit c49542d47960b5b4566f5823f0470b6411067e64[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 20:04:48 2019 -0400

    abstract plan functions

[33mcommit 2c1e2a7f5ed38233e866686ac2134d94b6cb79d5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 16:59:29 2019 -0400

    add perfect_place in pybullet

[33mcommit 3c14f579258aacb01a2510e6454b167929b11e3f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 16:59:00 2019 -0400

    add lateral friction in gripper base. fix finger sliding problem after restoring

[33mcommit 7b4fe9ef8cb04bcd8585773e676aece93a28d982[m
Merge: 711dd77 6e77b0b
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 12:46:49 2019 -0400

    Merge branch 'house_building' into devel

[33mcommit 711dd774e6b7caa3559d71433f0bdfafd66fafd1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 12:28:19 2019 -0400

    improve stacking plan

[33mcommit 6e77b0b09d5c5ca29a07e07a4e94d85229cf9e77[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 19 12:13:35 2019 -0400

    fix bug in plan house building

[33mcommit 56f7943bdcf9110815d7a26e4be8f59f8e10463d[m
Merge: 72c5c3a a67bd29
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 14 17:04:20 2019 -0400

    Merge branch 'devel' into house_building
    
    # Conflicts:
    #	pybullet_toolkit/robots/robot_base.py

[33mcommit 72c5c3ab4a7e67a3238cf58f6b8e9f09a719f622[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 14 12:30:36 2019 -0400

    fix scale bug

[33mcommit 8e1ca8c702152b8f7bed535308584ca505ab1744[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 13 17:19:40 2019 -0400

    minor changes

[33mcommit 6d106b10a2b36ecbd5685a717b22a83b483a4319[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 13 13:58:39 2019 -0400

    move perfect grasp checking to pybullet_env from robot class

[33mcommit df892b9a8bd8ed8bcabc79663c8aac6adebf3bec[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 13 13:57:57 2019 -0400

    fix step left bug in house building env

[33mcommit 120eac295069bf6fed474711cb5bf591b4fad82d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 12 20:40:45 2019 -0400

    change pick rot checking in pybullet perfect grasp to pi/12 (15 degrees)

[33mcommit a67bd29cb9ab48661fd70cc7731363a97351eb65[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 12 20:40:45 2019 -0400

    change pick rot checking in pybullet perfect grasp to pi/12 (15 degrees)

[33mcommit 10f2df7b7d9de21b78b6b4fb5db33e111a28118f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 12 20:29:38 2019 -0400

    toward adding house building env

[33mcommit 9f1a84a732677b1156daa0591de1606b2fdcc685[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 12 18:12:01 2019 -0400

    change pick rot checking in numpy to pi/12 (15 degrees)

[33mcommit f1dbd97c199140363bda73296dea3a2593318155[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 12 17:58:19 2019 -0400

    toward adding house building env

[33mcommit 078f92036bfc12680ad140d9c2359c6e0faeab43[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Aug 9 17:40:54 2019 -0400

    add planner and step_left reward in numpy block picking

[33mcommit 150387361df0272b80ce06aabbc2927e7443a006[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 8 20:41:55 2019 -0400

    increase the difficulty of grasp checking and stack checking: decrease pick rot threshold to pi/15, which is the same as pybullet perfect grasp; decrease stack pos threshold to size/4

[33mcommit a2dc2a6538df548940252d46e8fb061e9ea8ec8d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 7 18:12:41 2019 -0400

    fix bug

[33mcommit 2e3e0e2d4b39a4c2660bb65eb5d62a1dc5edef65[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 7 17:40:31 2019 -0400

    play with parameters

[33mcommit adfbad404aa40705330790b39d2a949bd3c5f53e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 7 16:30:24 2019 -0400

    fix bad parameter

[33mcommit 62146d295547243b0e6b2a6399deb272a9f7a4d5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 7 15:08:37 2019 -0400

    integrate ur5_robotiq with robot base class, improve robotiq gripper

[33mcommit 91d3c85c7dfbd37deeda897e77e40891cfdd6867[m
Merge: 7a2f4b2 04288db
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Aug 7 12:09:13 2019 -0400

    Merge branch 'devel' into robotiq
    
    # Conflicts:
    #	envs/pybullet_env.py

[33mcommit 04288db440913e5efa44c92c741721600b1dd08b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 17:58:14 2019 -0400

    fix finger sliding problem in ur5

[33mcommit 0e453ce67d9938bec5fe124649204905480616a0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 17:03:42 2019 -0400

    fix bug that applies different forces for 2 fingers.

[33mcommit 40e93a16b0fca9a43d72829b38d7eae802fe8bdd[m
Merge: 35c0338 4a75190
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 15:34:42 2019 -0400

    Merge branch 'kuka' into devel
    
    # Conflicts:
    #	envs/pybullet_env.py

[33mcommit 4a75190c4e45441bddc339d3e1fe1a06d7c86a1c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 15:32:52 2019 -0400

    add arm base class robot_base. add 'robot' in env config to specify which robot to use

[33mcommit fd2b0c726d4834bdf0885f904357dd05cade0dd3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 14:19:53 2019 -0400

    change parameters

[33mcommit 35c0338cc4bc54354e001e1c226f08fdb0fef19a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 14:14:03 2019 -0400

    change parameters

[33mcommit b0271539ce7d79b64d958b4582f805610fe6d80b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 12:36:39 2019 -0400

    change parameters

[33mcommit 397ba8d9e42fca0286fe440fc5e954c92c87dd9e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Aug 6 10:59:00 2019 -0400

    change parameters

[33mcommit 264ec399ae33035d27c0a4f7687996d894e3d5db[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 5 18:01:31 2019 -0400

    finished kuka

[33mcommit 467162a972425d6eddf040a73fc17c647586dc89[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 5 16:28:57 2019 -0400

    toward adding kuka arm

[33mcommit e069ac936c41e157767592993189968c4663b4a3[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Aug 5 15:53:25 2019 -0400

    Fixing merge conflicts

[33mcommit 2fb9f6284c77fe7bdda434adb16b5eed2039dc05[m
Merge: 23b85d5 aea1b9e
Author: Colin Kohler <colink78@gmail.com>
Date:   Mon Aug 5 15:48:26 2019 -0400

    Merge pull request #10 from pointW/devel
    
    Devel

[33mcommit 196c82f6e95f2cec15f7b20b35413ca9ea0312e2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Aug 5 14:10:49 2019 -0400

    toward adding kuka arm

[33mcommit aea1b9e418fad26ea5b66d4776d5f5cd98c74d3a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Aug 4 14:36:49 2019 -0400

    fix infinite loop in object creation

[33mcommit 7a2f4b28793eeb01df708d955bcdaa5e66c5c879[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Aug 2 15:05:38 2019 -0400

    grasping not stable

[33mcommit 1f1e1890efe096f8cdc9303874b9b727a4cd1344[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 1 22:14:08 2019 -0400

    toward adding robotiq85 gripper

[33mcommit 5cd92be24c37f2496520f504145b9a4165b347c2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Aug 1 11:55:46 2019 -0400

    add stack planner in pybullet

[33mcommit a126134b8abdcd4bc343e184753be7e83af5a1da[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 31 21:43:23 2019 -0400

    change action rotation in pybullet into counter-clockwise

[33mcommit f76bd9d0e6991472ac92d9b3225f07f22584aea6[m
Merge: 4771299 a41dbc9
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 31 21:11:06 2019 -0400

    Merge branch 'x_optimal_estimate' into sl

[33mcommit a41dbc9ea342f3a66ac82aaa42068ce0612c8d54[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 31 21:10:30 2019 -0400

    fix function naming inconstancy for using sl target in pybullet

[33mcommit 7a138034092349229baeacd5d2be47c694c712f3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 31 20:32:10 2019 -0400

    fix low height problem

[33mcommit 94d936c94af7cd42d4c429d8d364cdf79fbc9b19[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jul 26 15:42:23 2019 -0400

    add scale parameter in numpy, allowing test in smaller size

[33mcommit f3b44895b7f3738227f571107a4dbba160ab5164[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jul 24 21:38:18 2019 -0400

    fix bug; fix pybullet offset

[33mcommit abb4a289ae659745771150582ea7b009a32df10c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jul 23 11:49:00 2019 -0400

    fix bug in getStepLeft. now return 100 when sim is not valid

[33mcommit 7f001f954c7f3102b5d8213b0a03ad1d84fad769[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jul 23 11:48:29 2019 -0400

    add encodeAction function that enables planner to return action base on action_sequence

[33mcommit 4771299ca53866b9bc73a201919f7eb35aaa723a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 22 20:52:45 2019 -0400

    decrease size of local region for selecting safe z

[33mcommit d0bd61183e3e976cc9a2fb0e8cd2201dffe7c59a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 22 20:52:45 2019 -0400

    decrease size of local region for selecting safe z

[33mcommit ec8ab4e2d4111a9edc7ab1d3309b876cfcaaba18[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 22 20:52:12 2019 -0400

    fix optimal step left computation by trying a planned action at X

[33mcommit a02291cd02cf21c8c4020e8ed384ba8d7e61f0c8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jul 22 12:34:35 2019 -0400

    add reward type step_left_optimal that will return (step left, optimal step left in X, if X is possible)

[33mcommit 6c633ea1667e15f0fa81eebeaa2b959d0760a491[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Jul 21 18:30:56 2019 -0400

    toward implementing functionality that estimate the best outcome in particular X

[33mcommit 12baa5c866d1f65dc9c7cf54eca4926aa470c544[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 28 15:29:02 2019 -0400

    add block stacking planner

[33mcommit 2f0086966d39a09c5277ec256898b31c08fed643[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 27 20:51:48 2019 -0400

    add 'step_left' reward type in numpy block stacking for sl

[33mcommit c7a4f3a1f131c76361d9aa33c94ce8aaa038816d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 20 13:48:15 2019 -0400

    fix block initial position problem

[33mcommit a469adcedcc93bfd2dcfc9c42d2fae3efe517f98[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 19 17:04:49 2019 -0400

    change min_distance and padding in pybullet block generation

[33mcommit b6077f6f5d7e72be111755a9e3c69a2531701506[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 18 19:05:13 2019 -0400

    check perfect grasp before closing gripper

[33mcommit ae98ef98c82aea550fa12f8af77ef999285fbd75[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 18 16:13:59 2019 -0400

    define projection matrix base on workspace

[33mcommit f42a4209724b8dc7821beb3834d0e8f63bbfbe6e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 18 13:06:02 2019 -0400

    change perfect grasp threshold

[33mcommit 2780b66f9a9ee6fd2ecb384eb743539dc3a3d4a6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Jun 15 16:54:29 2019 -0400

    stop tracking test files

[33mcommit ca8cb2abcfd8a3ff89f09f0ca4888691e3ab4f50[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Jun 15 16:46:33 2019 -0400

    increase getPickedObj z threshold

[33mcommit 281543f56df6e8ee569d76ba9a392f7fc6951fac[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sat Jun 15 16:45:59 2019 -0400

    restore changes in reset function.

[33mcommit ed7453b02a903463f046105e75544270a8029859[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 13 13:38:05 2019 -0400

    increase joint control gain

[33mcommit 6ba388e74f5ef65783d2c52e04b66fd4cddecbcd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 13 13:35:20 2019 -0400

    change reset policy: only reset whole simulator after certain amount of reset() calls, otherwise just restore simulator state

[33mcommit 1e3d27deca7238d549fac691234e7d33875160e0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 12 20:24:13 2019 -0400

    increase min distance; when getting number of top objects, return -1 when holding obj

[33mcommit a497354a61e790fcf18d3c87c9c9b5bb7e7bf74c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Jun 9 18:45:29 2019 -0400

    fix missing attribute rot in cylinder

[33mcommit 299508cb1e27eb481f809ed7a6dcb5e9bb635865[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Jun 9 18:35:26 2019 -0400

    fix argument bug in numpy cylinder

[33mcommit 843181d0344f46fcc8eb51789387f8bdcd6abadd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Jun 7 16:41:31 2019 -0400

    add perfect_grasp option for pybullet

[33mcommit 78b581cc5fe5f763d388608ebed9a0be0ad48c0f[m
Merge: 4d1a0e3 546dc7e
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 6 17:36:02 2019 -0400

    Merge branch 'stack_w_rot' into devel

[33mcommit 4d1a0e306c4528ec9634cea24c5d3bc493b81178[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 6 14:47:53 2019 -0400

    when simulate grasp, apply dynamics from pre to grasp

[33mcommit 546dc7e4508d13baf5ff6f9530560c106226d2f9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Jun 6 14:46:48 2019 -0400

    minor change

[33mcommit fb6fde553b5e708f441888e71725733a24106808[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 5 18:22:02 2019 -0400

    add pick_rot and place_rot as parameters (currently only in block stacking)

[33mcommit 65219e96253946a88bd7d4d496cf295c7ce62f19[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 5 16:01:32 2019 -0400

    change stacking env: when picking, ignore pick rot, change block.rot to - pick rot. when placing, place rot += block.rot and check place rot

[33mcommit 3d8fdd48e3e91448f62381fd2dc6feafce8ee1d1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 5 12:30:52 2019 -0400

    check rotation in stacking numpy

[33mcommit 37653f89a1a8c677736607f69db5e446aa52a897[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Jun 5 12:25:21 2019 -0400

    increase threshold in isSimValid

[33mcommit a5b3b8277993cd4f81651a2355543aed39ebbeef[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Jun 4 18:33:23 2019 -0400

    when simulate grasp, apply dynamics in placing

[33mcommit 23b85d527957b6c025fdcc49fb900860c946258d[m
Merge: b84da3f 70c05d6
Author: Colin Kohler <colink78@gmail.com>
Date:   Tue Jun 4 14:49:51 2019 -0400

    Merge pull request #9 from pointW/curriculum
    
    Curriculum

[33mcommit 70c05d657c5c2276ecbc61b13325374736a4f3cb[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jun 3 15:37:57 2019 -0400

    use height to check stack in pybullet

[33mcommit 71a3178456fcacb6cf89029efa66c54273881e80[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Jun 3 12:57:26 2019 -0400

    add setPosCandidate in env factory

[33mcommit 559f8be6b42db9f8544b84fb44a8deaf65b119a4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 31 16:04:06 2019 -0400

    decrease threshold

[33mcommit d453c297d161e66422228b2b7667f50c8c5e1806[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 28 16:41:08 2019 -0400

    change wait timestep back to 100

[33mcommit 7ab658b139dde6d361db495904378e36a9e3a61e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 28 15:47:23 2019 -0400

    fix bug in no simulate grasp env placing is not waiting correctly. change not valid distance to 2cm. increase wait timestep

[33mcommit 7fe377c9750ffdb3e5c0787e7ad6f3f96183129b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 24 17:24:32 2019 -0400

    remove the randomness in object position. check object position near pos candidate in isSimValid

[33mcommit 578b388d37cf9b02356262b4796a6afa99dca761[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 24 11:11:33 2019 -0400

    add function for setting pos candidate

[33mcommit 53641d66496f44a676c615712c84d050ad5d5cbb[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed May 22 16:14:29 2019 -0400

    Revert "add randomness in block position"
    
    This reverts commit 4b442b9f

[33mcommit 4b442b9f8274fb1405f6d45126a6b6b3d0f56e16[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed May 22 15:19:28 2019 -0400

    add randomness in block position

[33mcommit 556fcee7242818a91950065e2f85e7e60de2297f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 21 17:12:27 2019 -0400

    add pos_candidate for curriculum learning. block will only be created at those candidate positions

[33mcommit c391148f87d4465e8564aa153d77b86f1cc95d9e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 10 15:02:23 2019 -0400

    add option to not simulate grasp, just attach instead

[33mcommit 1215cda6faeccf1bebaa662fad97b1426f2bb8d5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 7 16:33:21 2019 -0400

    fix bug

[33mcommit 41fca5af10b23160bd942f49d25db8a5f5069c47[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue May 7 13:44:01 2019 -0400

    add option to set constant seed

[33mcommit 126bec923a9a161ff65457c84b16f5700b56c433[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 6 16:20:19 2019 -0400

    add dense reward in block cylinder env

[33mcommit 20e760f1bea67eeec15bd0c92dc69ccd5afd0fd1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon May 6 15:09:37 2019 -0400

    bug fix

[33mcommit 25d32e8777063dbb3e208ef556fe34c6fdc2fa11[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri May 3 17:21:22 2019 -0400

    add block cylinder stacking env

[33mcommit 07b4eff88a6869c7e1c61e402a78ac2ae49fbf6a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu May 2 15:07:52 2019 -0400

    add brick stacking env in pybullet

[33mcommit 24bf19cd69e76ef6fa69305351b01e3e9793f290[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu May 2 15:07:34 2019 -0400

    always use dynamics when move from grasp pos to pre grasp pos, then check if obj is grasped. change getPickedObj metric

[33mcommit c6809937033a8d5338668ca2e45612991fc81a89[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 25 16:46:05 2019 -0400

    implement teleporting arm. teleport block using the relative position of block and end effector

[33mcommit 406ff5d83c708e7a458c9f7665fdcc41566535f6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 25 10:57:11 2019 -0400

    add dense reward option in block stacking

[33mcommit b84da3f3ec9c3d35f9eff4d96ff1e19f00dc0fc4[m
Merge: bd6c5da c64fd6f
Author: Colin Kohler <colink78@gmail.com>
Date:   Wed Apr 24 16:31:15 2019 -0400

    Merge pull request #8 from pointW/devel
    
    Add block stacking environment

[33mcommit c64fd6f488f848ad7cbc5b1eca6c5742e66a83fc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 24 13:09:19 2019 -0400

    keep 'take action' in base class, move everything afterwards (getting obs, reward, checking termination, etc.) in step() to subclass

[33mcommit 77b29f3382b2edff4e67834abf754480782e7011[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 24 12:47:48 2019 -0400

    use class variable gripper_joint_indices instead of hand coding

[33mcommit 9f4af2df4ac8b4fdafb7626772423014185248fe[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 23 15:50:36 2019 -0400

    only move to home position in action step. moving to resp pose is not necessary since home position is not blocking camera

[33mcommit 82b0ede8fe5022a1652c02ef128a44fc0983a9dd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 23 13:44:04 2019 -0400

    add class variable gripper_joint_limit, use setJointMotorControlArray in gripper commands.
    remove .item() in _getSpecificAction since the input action should always be numpy array

[33mcommit db31bb20f0479b7c9e43df62b81df22f1bf04470[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 22 17:58:14 2019 -0400

    move arm to home position before and after action

[33mcommit caaa53a8471468b2820d1748e4b84bdaa7586bc2[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 22 12:15:30 2019 -0400

    remove unused comments

[33mcommit da302cfd9b7a4fd6cd9f7f4614040ef2ef4bb461[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 22 11:47:44 2019 -0400

    revert changes in test

[33mcommit 0a6f466316f9288dda01076b0dcfd6a2939dc7e7[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Apr 21 17:29:23 2019 -0400

    open gripper at the start of each picking

[33mcommit 3adeb92fde8a53d6caea93115bc77cfa6369a8f6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Apr 21 16:53:58 2019 -0400

    1. fix bug in view_matrix. It should correspond to workspace
    2. send position command after setting joint poses in fast mode to overwrite previous command

[33mcommit 7d58b1256188f4af24db18fe2563de79bc1cbbc5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 19 16:54:08 2019 -0400

    1. check gripper status after transition
    2. check if simulator is valid by checking position of each block

[33mcommit 773a2332c7ec964e49dc7e418f0f29c9576711ec[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 18 17:40:52 2019 -0400

    restore changes in reset to fix the memory issue

[33mcommit da56c8eb339a1a387a1838f87a5689ade84fc0be[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 18 17:11:53 2019 -0400

    change parameters to make pybullet more stable. reset pybullet without reload urdf

[33mcommit 4c63db3a9bc9a0963d9995b999ed41a4e0c60f37[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 18 15:01:41 2019 -0400

    add height constraint in stacking in numpy

[33mcommit 0afc2aaf5bde2a357e6f86247bee00826fcc95fa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 18 14:34:10 2019 -0400

    add damping in finger joints, which seems helping with finger sliding issue

[33mcommit 5a1fa3c6d0d839be41ee2f720fdcdcb3f1abe485[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 18 14:33:22 2019 -0400

    change gripper command into position control. add max iteration in open gripper.
    position control command will not be overwritten by other position control commands, so it will not be necessary to send gripper command after each arm joint commands.

[33mcommit cd1a52514657a4b9be2761a538f6c2cf140887a3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 18 10:45:58 2019 -0400

    fix bug in (more than 2 blocks) stacking

[33mcommit fcc8ac25c15e2c8bdf69643fffd69f3e0c1adf20[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 17 12:42:13 2019 -0400

    add pybullet stacking env

[33mcommit 9d1cbdfd0cd7a0f7bcc821c2e004639e051e4cd5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 17 11:30:28 2019 -0400

    add inertial data to dummy link to suppress bullet warning

[33mcommit e3875618b8b5eb90a0ebbdcfbe1539e75a14a3fe[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 17 10:24:43 2019 -0400

    add padding in place action

[33mcommit f0e12798ba82ec1000f711385d4461975debca62[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 16 18:12:08 2019 -0400

    increase stacking tolerance

[33mcommit 40d350d39d1ceffb1baae17b41182901074973c4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 16 17:21:43 2019 -0400

    minor fix

[33mcommit df277d6d5efa6cf063623bbb7f2313103ba61ea9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 15 15:50:43 2019 -0400

    fix bug in save/restore in pybullet

[33mcommit f09844c9945809e4e57332b799784c1a76d79cdd[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Apr 14 13:58:59 2019 -0400

    minor change in test

[33mcommit 751cc800346324d356f3b646102ac01481e76a74[m
Merge: 59c5dcc bd6c5da
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Sun Apr 14 13:56:15 2019 -0400

    Merge pull request #8 from ColinKohler/master
    
    merge from colin's master

[33mcommit 2b8682c79192de0d09d1d8cccb6a5fc9177676fa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 12 16:22:20 2019 -0400

    fix rotation judgement in grasping AGAIN; increase minimal cube distance

[33mcommit 0dd24080f0002c3824bb905654d378652901cb76[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 12 13:06:22 2019 -0400

    fix rotation judgement in grasping

[33mcommit bf0266f1d5d8370a7fdbf4bc1dd7fa70bf173fc8[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 11 20:44:20 2019 -0400

    fix bug

[33mcommit c5f742ccc5a4157d7ef45f1238d84ef14b52b03f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 11 10:35:44 2019 -0400

    fix commit

[33mcommit bbc5efa28bc4ec8f0949e3d99ebd1deae7ccf0ad[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 11 10:28:29 2019 -0400

    fix bug

[33mcommit 56480e8fa07ea3a8ecf959cea575d9115b4b376b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 11 10:22:41 2019 -0400

    fix bug in save state

[33mcommit bd6c5dac6e391ebc19e322b18bdb971bcaa3dcfc[m
Merge: e455505 d78a4d9
Author: Colin Kohler <colink78@gmail.com>
Date:   Wed Apr 10 16:28:26 2019 -0400

    Merge pull request #6 from pointW/pybullet_devel
    
    Add multiple object option in both numpy and pybullet

[33mcommit ecf2c8076fe399664decf2f890f58e42a10e6dcf[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 10 15:28:03 2019 -0400

    modify the logic of block stacking:
    remove the bottom block pointer within Cube since it caused copy issue;
    now keep an attribute 'chunk_before', which is the heightmap chunk before stacking. when removing, restore the chunk.
    after placing, fix the 'on_top' by comparing heitmap value at cube mask and cube pos

[33mcommit b5ba23fd46d423f11897759840c0469fbfd64b60[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 10 12:18:50 2019 -0400

    bug fix

[33mcommit a25149f48b2fdbd673a7beb4e483fbada5b9e2d3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 9 18:24:29 2019 -0400

    bug fix

[33mcommit 169038ad01d70fb7b370bc318b74644794222f09[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 9 17:14:28 2019 -0400

    add save and restore in numpy

[33mcommit ceaf6dc37d1447cb47d3be4bb078241a83b6d7f4[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 9 15:05:08 2019 -0400

    add block stacking env in numpy

[33mcommit d78a4d97184ccefe7dba2b81fb34333c66bcfccf[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 17:55:44 2019 -0400

    minor change

[33mcommit de792f9154e675e490cd863d78fefe2646f843d6[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 17:42:07 2019 -0400

    change pybullet env to fit multiple object domain

[33mcommit e45550538ecbcaa5f28bd6c4d7dff079a65c7389[m
Merge: 7cfa0fc e937bca
Author: Colin Kohler <colink78@gmail.com>
Date:   Mon Apr 8 17:30:37 2019 -0400

    Merge pull request #5 from pointW/pybullet_devel
    
    Add save and restore; fix fast mode; fix bugs

[33mcommit 370222a52660725a6cae3296309a89ae31ae0928[m
Merge: e937bca 6149d8a
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 17:13:38 2019 -0400

    Merge branch 'numpy_devel' into pybullet_devel
    
    # Conflicts:
    #	envs/block_picking_env.py
    #	envs/numpy_env.py
    #	numpy_toolkit/object_generation.py
    #	tests/test_numpy_env.py

[33mcommit 6149d8af782f60ba823869aa64194c7e77a6d9ee[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 16:54:58 2019 -0400

    delete vcs.xml

[33mcommit e937bcad3b0cf45e6641bc184e6c73d9f9f35c8f[m
Merge: a11871f a090aa0
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 16:43:57 2019 -0400

    Merge remote-tracking branch 'origin/pybullet_devel' into pybullet_devel
    
    # Conflicts:
    #	env_runner.py

[33mcommit a11871fe73c4cf0d7cba38437c41cbeaaca1dcf6[m
Merge: 0d680a9 bda2d53
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 16:39:00 2019 -0400

    merge pybullet_test

[33mcommit a090aa05d5e24a85021bf03e520ea58406e04c9a[m
Merge: 0d680a9 7cfa0fc
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 15:20:25 2019 -0400

    Merge branch 'master' into pybullet_devel

[33mcommit 0d680a961204221dfec39bb19447625824a5c74b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 15:06:38 2019 -0400

    restore changes in reset. not necessary to increase reset speed

[33mcommit 5b9a4dff7481074c91fd9219809bd3bb6926b7f3[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 14:31:48 2019 -0400

    set joint velocity to 0 before set joint poses

[33mcommit c6e002328f46bef613e70b936a15366e0c43beaa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 14:31:01 2019 -0400

    add max inner loop iteration to prevent loop forever

[33mcommit e75c2a6c4857a9e9624a213f986704ced3167665[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 13:54:14 2019 -0400

    add random_orientation config option in block picking

[33mcommit db65cb34d191ae244662c625e954c4b95951b45d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 13:51:02 2019 -0400

    switch observation xy axis in pybullet to match other envs

[33mcommit f9d75c1b78ed9e0c09dba73832f21cc02cb48f89[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 8 10:40:12 2019 -0400

    increase collision box size

[33mcommit d5836ec57cb79e6c3f53b90a4957047bd752ec24[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 5 18:59:32 2019 -0400

    fix bug in picking in fast mode

[33mcommit 9bf9e4d48a7ba445d2d0d92a09806a7e2a9fa168[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 5 17:48:42 2019 -0400

    add option to not automatic reset env when done

[33mcommit 9616cf41fdc080a48a6d429be68e86158a071826[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 5 17:15:41 2019 -0400

    speed up reset with restoreState

[33mcommit c2676b3efb29de8a64dd4b67f5eceeca0148824d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 5 16:33:33 2019 -0400

    add save and restore in pybullet

[33mcommit 62282f886eea4786222ed0a860f6b8a76bbe0c9c[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 5 15:34:11 2019 -0400

    upsample before rotate in numpy image to increase performance in small size domain

[33mcommit 59c5dcc7e10ea1f9fb1ade59957734f667be055d[m
Merge: a1c9603 7cfa0fc
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Apr 5 10:44:23 2019 -0400

    Merge pull request #6 from ColinKohler/master
    
    Merge from colin's master

[33mcommit 167ebd42551604057f056cb31722533cf71793cc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 19:04:37 2019 -0400

    fix a severe bug in sparse reward

[33mcommit 7cfa0fc027e9dfd66153af48dd177257dc3e8b9f[m
Merge: 396b901 bda2d53
Author: Colin Kohler <colink78@gmail.com>
Date:   Thu Apr 4 18:00:55 2019 -0400

    Merge pull request #4 from pointW/pybullet_test
    
    Test pybullet picking

[33mcommit bda2d53b6ff9494e81ff84bac4aea17954c761f5[m
Merge: 0716892 a599ae8
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 17:40:50 2019 -0400

    Merge remote-tracking branch 'remotes/origin/pybullet_devel' into pybullet_test

[33mcommit 396b901780cdf64da7f3cc73c2b07e003e86432e[m
Merge: a1e180e a599ae8
Author: Colin Kohler <colink78@gmail.com>
Date:   Thu Apr 4 17:19:12 2019 -0400

    Merge pull request #3 from pointW/pybullet_devel
    
    Fix inaccurate IK and gripper loose during transit

[33mcommit a599ae86ebdd558df0f10c72547ac4f7901ad625[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 17:18:12 2019 -0400

    remove motor_names and motor_indices

[33mcommit 407d246bb606077b59d02fe26f02c231a04b1e1f[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 16:58:58 2019 -0400

    remove commented code. change gripper force back to 100

[33mcommit ca4dd9f6d1725ae37e39e71a6b98579a26341ed0[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 16:56:18 2019 -0400

    set min_distance and padding base on heightmap size

[33mcommit 0716892f82a2820377f194080fe14ce8fb5f6612[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 12:24:12 2019 -0400

    test pybullet picking

[33mcommit 2154e9d6959829a678f4829ba836a91c9d2fe734[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 11:35:02 2019 -0400

    add arm_joint_names and arm_joint_indices to for 6 arm motors. restore motor_names and motor_indices for all motors including gripper

[33mcommit 1d7cc30df6d85f8867589336a2f0045f93725f27[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 11:22:33 2019 -0400

    small change

[33mcommit dcf87e2f566b2175a26c008211a2aed8ea5f1597[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Apr 4 11:16:52 2019 -0400

    remove gripper position command. change the metric for judging gripper is closed or not with finger distance and number of iterations

[33mcommit b036781786d6be5efd1c1e9277e786d7f368fdcc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 3 18:00:49 2019 -0400

    resend close command after moving arm if gripper is supposed to be closed

[33mcommit 4c9889662d03ed4216cff81c02e86dfae47c0844[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 3 17:58:16 2019 -0400

    change move function to iteratively solve ik and send position command

[33mcommit c332730823b0a5632c2347374f76dac852347080[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 3 17:53:58 2019 -0400

    fix bug that pre_rot is different from rot in picking. change timestep into default 1/240. increase block size

[33mcommit 30a662028682e85f16d66f108b9e8cc9a746ebd5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Apr 3 11:03:41 2019 -0400

    update gitignore

[33mcommit a1c9603b3a1a1a9f9d41d6cc294334acec94e17b[m
Merge: 8b844f6 a1e180e
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 2 17:28:54 2019 -0400

    Merge pull request #5 from ColinKohler/master
    
    merge from colin

[33mcommit a1e180e6ccafe4ea8062583da3fd845a109fc080[m
Merge: c8791e4 625027c
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Apr 2 17:23:21 2019 -0400

    merge conflict fix

[33mcommit c8791e4df0f4eebf23629fc7809af2be1d3ce475[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Apr 2 17:22:45 2019 -0400

    Small changes

[33mcommit 625027cfda7189453710cbfaa3d55995afa8d244[m
Merge: 1b75cf4 8b844f6
Author: Colin Kohler <colink78@gmail.com>
Date:   Tue Apr 2 17:18:59 2019 -0400

    Merge pull request #2 from pointW/master
    
    Fix grasp checking doesn't work when z is not part of the action in numpy.

[33mcommit ebb795d190f5e5b63a5ba4ef38ab2db224c213a1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Apr 2 16:41:40 2019 -0400

    add multiple object option in numpy

[33mcommit 6aa51779a1ef28a69006e5beeb11f972f5bf21ee[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 1 16:56:33 2019 -0400

    working on adding multiple object option

[33mcommit 8b844f65496401a0d0ab9f29423713636698da6a[m
Merge: 5a37cff 1b75cf4
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Apr 1 15:29:10 2019 -0400

    Merge branch 'master' into master

[33mcommit 272464a51e18eafda2e43d4fc9ab27e2c696d833[m
Author: ColinKohler <colink78@gmail.com>
Date:   Mon Apr 1 14:50:43 2019 -0400

    More work towards simple gripper working

[33mcommit 5a37cfffda4d4e6578a349a186c1126701837e2e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 26 15:36:50 2019 -0400

    check grasp with block size

[33mcommit 1b75cf4275ae7ccd42555be0eb136d279de8e6ad[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Mar 26 15:04:57 2019 -0400

    pybullet gripper can now pick objects but placing is doing weird stuff

[33mcommit 1c3aff1071a6b533b3f94c49f3614a2252682b4a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 26 12:42:24 2019 -0400

    remove test changes

[33mcommit a9eebdf6030257e04adbef11ec6b871f29f752c2[m
Merge: b31c3c6 0fe149b
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 26 12:37:42 2019 -0400

    Merge remote-tracking branch 'origin/master'

[33mcommit b31c3c6a147c84e4b369f0b950039b50d7985f01[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Mar 26 12:36:48 2019 -0400

    fix isGraspValid doesn't work when z is not part of the action in numpy.

[33mcommit 4facd13ff4b303b04e8d770bc5887782898d1c5c[m
Merge: 89f907e 0fe149b
Author: Colin Kohler <colink78@gmail.com>
Date:   Tue Mar 26 10:43:40 2019 -0400

    Merge pull request #1 from pointW/master
    
    Add orientation and config options

[33mcommit 0fe149b47a217750999cd3524ab341d883383bf2[m
Merge: fc8ef53 89f907e
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 25 17:54:17 2019 -0400

    Merge branch 'master' into master

[33mcommit fc8ef5303c5a932e7b1dd3a9627a62e6e9a4eb08[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 25 15:11:28 2019 -0400

    fix bug that re-add objects in heightmap

[33mcommit 1a2118998f230aac2789b44980c9dd3afa25251a[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 25 15:09:17 2019 -0400

    add _isObjectHeld in vrep/pybullet and unify _checkTermination in block picking.

[33mcommit 89f907ee22e9585da385324253aff462662a2071[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Mar 22 17:39:52 2019 -0400

    Swaped gripper with simple gripper on ur5/

[33mcommit 476589ff205304ec16420de294b50ad6473bd18c[m
Merge: 07539b5 a54866e
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 22 14:31:55 2019 -0400

    Merge pull request #4 from pointW/devel
    
    Devel

[33mcommit a54866e6cd93ea5c207d9844ea52d1743433491d[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 22 14:09:06 2019 -0400

    Fix bug in reversed x y axis in obs. Fix isGraspValid

[33mcommit b7612063b7fd593e5c1fdbee1c3fdb1809e27aa9[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 18 17:37:12 2019 -0400

    add random rotation and action sequence in numpy env

[33mcommit 602e81a93fb47083ae5fcd0986dd3d503135fc6d[m
Merge: 8098075 bb839f6
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Mon Mar 18 15:41:51 2019 -0400

    Merge branch 'master' of https://github.com/ColinKohler/helping_hands_rl_envs into ColinKohler-master
    
    # Conflicts:
    #	envs/block_picking_env.py
    #	tests/test_base_env.py

[33mcommit 8098075c8c65c278e62d336dba23d9f7dcea7ad5[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Mar 6 14:50:46 2019 -0500

    speed up vrep by only restart simulator when not valid

[33mcommit a6d97cd981dc203130315ded49a5aa6fa9f0977f[m
Merge: c6cdbc0 c0876c5
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Mar 6 09:53:17 2019 -0500

    Merge remote-tracking branch 'remotes/origin/height_devel' into devel
    
    # Conflicts:
    #	test_base_env.py

[33mcommit c6cdbc0b2d7c2e89781c40118744c2a6291dc756[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Mar 6 09:51:53 2019 -0500

    change test

[33mcommit bb839f68899994c8f27944e2c3e2bc6465873d66[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Mar 1 13:52:38 2019 -0500

    Added numpy to simulators, simple block picking task works for numpy envs

[33mcommit c0876c545010c5ba9a352c9bb780eec1168e21ec[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Fri Mar 1 12:04:32 2019 -0500

    add config['action_sequence'] to specify action format. add _getRestPoseMatrix to solve bug caused by different rest pose format in different simulator

[33mcommit ae1f1f08ba500dfce1bdb6cb01373e96d99d40fa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 28 18:39:15 2019 -0500

    add simulation file

[33mcommit d40227502cae87c3a5fd6ea112ab3528ef18f638[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 28 18:37:39 2019 -0500

    changes for own use

[33mcommit 07539b5d369abcf14c66c78c1a3cbc6c1c733d05[m
Merge: fadab82 e831003
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 28 18:37:05 2019 -0500

    Merge pull request #1 from pointW/pick_rot_devel
    
    Pick rot devel

[33mcommit e831003d697d593bd8973e9840c24628b7d8a4dc[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 28 13:47:27 2019 -0500

    add port option in vrep configs; change termination height in block picking

[33mcommit d2e080ce70f4a2bffbf0aca8038e1050f4637daa[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Thu Feb 28 13:30:53 2019 -0500

    fix bug in rotation: now using target own frame to calculate rotation at each step, instead of world frame
    set euler flag to 'rxyz'.

[33mcommit 53fd681b2601ba23ba10e47cee1c71d5982f964e[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Wed Feb 27 17:55:34 2019 -0500

    add orientation flag in _generateShapes

[33mcommit 4af01ac013893539384caa93a1f2bf4cf22035d1[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 26 18:59:58 2019 -0500

    add rotation in pick

[33mcommit 5907eb1b4d593f4bc0ad826f9ab0c758d1381824[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 26 18:47:07 2019 -0500

    fix bug in _checkTermination

[33mcommit 56fc600bce7e01119edb3aceed49d1f5ff07db6b[m
Author: Dian Wang <wangdian1007@gmail.com>
Date:   Tue Feb 26 17:26:38 2019 -0500

    add _getObjectPosition in vrep env, change home_pose into rest_pose

[33mcommit fadab829aeee8c089c8a20c9547b4e1168acb84b[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 22 13:03:20 2019 -0500

    Started adding numpy base env. More work on pybullet envs as well.

[33mcommit 5197ff0226a866fd5c100e67f1898579159dcf54[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Feb 20 12:07:24 2019 -0500

    Working on getting the block picking env working with a rl agent.

[33mcommit 42a1b94937ac97d85e8984f4c6a443cd1abab174[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Feb 19 17:54:31 2019 -0500

    Changed moveTo to use realistic UR5 forces and added a way to break out of the loop if the desired position cannot be acheived.

[33mcommit f1f669bec0c88baeeefbc0ddd07436e0872cf6ff[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Feb 19 12:27:24 2019 -0500

    Fairly ghetto at the moment but the block picking env seems to be working when not run in fast mode.

[33mcommit 930a7438d59d22c7603d9fc4bae994cdda716fe2[m
Author: ColinKohler <colink78@gmail.com>
Date:   Fri Feb 15 14:09:11 2019 -0500

    More work towards pybullet envs. Block picking is close to working.

[33mcommit f1a976ae67a40c324b9b59044d241afddd9ba062[m
Author: ColinKohler <colink78@gmail.com>
Date:   Wed Feb 13 18:05:38 2019 -0500

    Added vrep and pybullet toolkits to repo. Modified multiprocessing junk to work with the new env inits. Added a base env for the vrep and pybullet envs. More work on getting pybullet env working; currently basic pick functionality works but its very rough.

[33mcommit 645b8da08e3b226098bfb5680fd491a65b8b3027[m
Author: ColinKohler <colink78@gmail.com>
Date:   Tue Feb 12 14:44:58 2019 -0500

    Initial commit.

[33mcommit ca3be469a65435841c3dc2cfb3dbaa770627d47d[m
Author: Colin Kohler <colink78@gmail.com>
Date:   Mon Feb 11 11:14:36 2019 -0500

    Initial commit
