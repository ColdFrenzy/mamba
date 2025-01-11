from envs.multigrid import *

class ComplexEnv(MultiGridEnv):
    """
    Multi-agent reinforcement learning environment with multiple room. The agents must
    navigate in a maze composed of multiple interconnected rooms.
    The doors leading to the goal can be opened only two agents are standing simultaneously
    on two switches, located in different rooms. If both switched are pressed, 
    the door leading to the goal will be opened and another agents can reach the goal.
    All agents will receive a reward if one of them reaches the goal.
    """

    def __init__(
        self,
        size=None,
        view_size=3,
        width=20,
        height=10,
        goal_pst = [],
        goal_index = [],
        num_switches= [],
        switches_pos = [],
        toggled_switches_index = 1,
        untoggled_switches_index = 4,
        correct_switch_combinations = {},
        agents_index = [],
        actions_set=SmallActions,
        zero_sum = False,

    ):
        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.switches_pos = switches_pos
        self.num_switches = len(num_switches)
        self.untoggled_switches_index = untoggled_switches_index
        self.toggled_switches_index = toggled_switches_index
        self.zero_sum = zero_sum

        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        self.door_pos = []
        room_w = width // 20
        room_h = height // 2
            
        
        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(self.world, xR, yT, room_h)
                    # pos = (xR, self._rand_int(yT + 1, yB))
                    # self.door_pos.append(pos)
                    # self.grid.set(*pos, Door(self.world,color="brown", is_locked=True))

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(self.world, xL, yB, room_w)
                    # pos = (self._rand_int(xL + 1, xR), yB)
                    # self.door_pos.append(pos)
                    # self.grid.set(*pos, Door(self.world, color="brown",is_locked=True))

        for i in range(len(self.goal_pst)):
            self.place_obj(ObjectGoal(self.world,self.goal_index[i], 'goal'), top=self.goal_pst[i], size=[1,1])

        for sw_pos in self.switches_pos:
            self.grid.set(*sw_pos, GroundSwitch(self.world,self.untoggled_switches_index, self.toggled_switches_index))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards,reward=1):
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward


    def step(self, actions):
        obs, rewards, terminated, truncated, info = MultiGridEnv.step(self, actions)
        return obs, rewards, terminated, truncated, info


class ComplexEnvDefault(ComplexEnv):
    def __init__(self):
        super().__init__(size=None,
        height=10,
        width=20,
        goal_pst = [[5,5]],
        goal_index = [1],
        switches_pos = [[4,4]],
        toggled_switches_index = 1,
        untoggled_switches_index = 4,
        num_switches = [1],
        correct_switch_combinations = {},
        agents_index = [0,0],
        zero_sum=True)


if __name__ == "__main__":
    env = ComplexEnvDefault()
    _ = env.reset()

    nb_agents = len(env.agents)


    while True:
        env.render(mode='human', highlight=True)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, terminated, truncated, _ = env.step(ac)

        if terminated or truncated:
            break
