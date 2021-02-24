import itertools
from helping_hands_rl_envs.simulators import constants


class CheckGoal:

    ONE_BLOCK = "1b"
    TWO_BLOCKS = "2b"
    BRICK = "1l"
    TRIANGLE = "1r"
    ROOF = "2r"
    CHARS = [ONE_BLOCK, TWO_BLOCKS, BRICK, TRIANGLE, ROOF]

    class Stack:

        def __init__(self, levels):

            self.levels = levels

        def check(self):

            #print("find candidates")
            for level in self.levels:
                level.find_candidates()
                #print(level.candidates)

            #print("eliminate candidatees")
            for level in reversed(self.levels):
                level.eliminate_top_down()
                #(level.candidates)

            return len(self.levels[0].candidates) > 0

    class Level:

        def __init__(self, level_type, env, height, below_level=None):

            self.type = level_type
            self.env = env
            self.height = height
            self.below_level = below_level

        def eliminate_top_down(self):

            if self.below_level is None:
                return

            joint_candidates = [[], []]

            for c1 in self.candidates:

                for c2 in self.below_level.candidates:

                    if self.type != CheckGoal.TWO_BLOCKS:
                        if self.below_level.type != CheckGoal.TWO_BLOCKS:
                            # two singleton objects on top of each other
                            if self.env._checkOnTop(c2, c1):
                                joint_candidates[0].append(c1)
                                joint_candidates[1].append(c2)
                        else:
                            if self.type in [CheckGoal.ONE_BLOCK, CheckGoal.TRIANGLE]:
                                # width 1 object on top of one or both of two cubes that are next to each other
                                for c3 in c2:
                                    if self.env._checkOnTop(c3, c1):
                                        joint_candidates[0].append(c1)
                                        joint_candidates[1].append(c2)
                                        break
                            else:
                                # width 2 object on top of both cubes
                                flag = True
                                for c3 in c2:
                                    if not self.env._checkOnTop(c3, c1):
                                        flag = False
                                if not self.env._checkInBetween(c1, c2[0], c2[1]):
                                    flag = False

                                if flag:
                                    joint_candidates[0].append(c1)
                                    joint_candidates[1].append(c2)
                    else:
                        if self.below_level.type != CheckGoal.TWO_BLOCKS:
                            # two blocks on top of one a brick
                            assert self.below_level.type == CheckGoal.BRICK
                            flag = True
                            for c3 in c1:
                                if not self.env._checkOnTop(c2, c3):
                                    flag = False
                            if not self.env._checkInBetween(c2, c1[0], c1[1]):
                                flag = False

                            if flag:
                                joint_candidates[0].append(c1)
                                joint_candidates[1].append(c2)
                        else:
                            # two blocks on top of two blocks:
                            if (self.env._checkOnTop(c2[0], c1[0]) and self.env._checkOnTop(c2[1], c1[1])) or \
                                    (self.env._checkOnTop(c2[1], c1[0]) and self.env._checkOnTop(c2[0], c1[1])):
                                joint_candidates[0].append(c1)
                                joint_candidates[1].append(c2)

            self.candidates = joint_candidates[0]
            self.below_level.candidates = joint_candidates[1]

        def find_candidates(self):

            candidates = []
            objects = self.env.objects

            for obj in objects:

                height = obj.getZPosition()
                level = height // 0.03

                if level != self.height:
                    continue

                if self.type in [CheckGoal.ONE_BLOCK, CheckGoal.TWO_BLOCKS]:
                    if self.env.object_types[obj] == constants.CUBE:
                        candidates.append(obj)
                elif self.type == CheckGoal.BRICK:
                    if self.env.object_types[obj] == constants.BRICK:
                        candidates.append(obj)
                elif self.type == CheckGoal.TRIANGLE:
                    if self.env.object_types[obj] == constants.TRIANGLE:
                        candidates.append(obj)
                elif self.type == CheckGoal.ROOF:
                    if self.env.object_types[obj] == constants.ROOF:
                        candidates.append(obj)

            if self.type == CheckGoal.TWO_BLOCKS:

                if len(candidates) >= 2:

                    self.candidates = []

                    for c in itertools.combinations(list(range(len(candidates))), 2):

                        obj1 = candidates[c[0]]
                        obj2 = candidates[c[1]]

                        if self.next_to(obj1, obj2):
                            self.candidates.append((obj1, obj2))

                else:

                    self.candidates = []

            else:

                self.candidates = candidates

        def next_to(self, cube1, cube2):

            x1, y1 = cube1.getXYPosition()
            x2, y2 = cube2.getXYPosition()

            diff_x = abs(x1 - x2)
            diff_y = abs(y1 - y2)

            return diff_x <= 0.015 and diff_y <= 0.075

    def __init__(self, goal, env):

        self.goal = goal
        self.env = env

        self.parse_goal_()

    def check(self):

        #import time
        #time.sleep(100)

        blocks, bricks, triangles, roofs = self.get_objects_()
        if not self.check_counts_(blocks, bricks, triangles, roofs):
            return False

        return self.levels.check()

    def get_objects_(self):

        blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
        bricks = list(filter(lambda x: self.env.object_types[x] == constants.BRICK, self.env.objects))
        triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
        roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

        return blocks, bricks, triangles, roofs

    def check_counts_(self, blocks, bricks, triangles, roofs):

        return len(blocks) >= self.num_blocks and len(bricks) >= self.num_bricks and \
               len(triangles) >= self.num_triangles and len(roofs) >= self.num_roofs

    def parse_goal_(self):

        self.num_blocks = 0
        self.num_bricks = 0
        self.num_triangles = 0
        self.num_roofs = 0

        self.levels = []

        assert len(self.goal) % 2 == 0

        for i in range(len(self.goal) // 2):
            chars = self.goal[i * 2: (i + 1) * 2]
            assert chars in self.CHARS

            if chars == self.ONE_BLOCK:
                self.num_blocks += 1
            elif chars == self.TWO_BLOCKS:
                self.num_blocks += 2
            elif chars == self.BRICK:
                self.num_bricks += 1
            elif chars == self.TRIANGLE:
                self.num_triangles += 1
            elif chars == self.ROOF:
                self.num_roofs += 1

            if i == 0:
                level = self.Level(chars, self.env, i, None)
            else:
                level = self.Level(chars, self.env, i, self.levels[-1])

            self.levels.append(level)

        self.levels = self.Stack(self.levels)

        self.height = len(self.levels.levels)
