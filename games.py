#General Imports
from copy import copy
import functools
from typing import Optional
import numpy as np

# Gymnasium Imports
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

#PettingZoo and Supersuit Imports
from pettingzoo import ParallelEnv

#My Custom Functions
from functions import plot_results, plot_multi
class GridWorldEnv(gym.Env):

    # Initializes the environment with specific attributes including size, observation_space, action_space, and any other variables
    # defining the agent, environment, or reward structure.
    def __init__(self, size: int = 5):
        # The size of the square grid
        self._size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    # A common design pattern is to include a _get_obs method for translating state into an observation. However, this helper method
    # is not mandatory, and you might want to compute observations directly in env.reset and env.step, which may be preferable if
    # you want to compute them differently in each method call.
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    # A similar pattern, _get_info can be used to return auxiliary information. In this Env, we would like to calculate and return
    # Manhattan distance from the agent to the target square.
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    # Reset is called to initiate a new episode for an environment and has two parameters, seed and options. Seed initializes the
    # random number generator to allow us to consistently generate the same environment when there are random variables involved.
    # Options is a dict containing any additional parameters we might want to specify during the reset.\

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self._size, size=2, dtype=int)
        # Sample random target locations until they do not coincide with the agent's starting location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self.agent_location):
            self._target_location = self.np_random.integers(
                0, self._size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to a direction on the map, using our helper dictionary
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self._size - 1
        )

        # We use `np.clip` to make sure we don't leave the grid bound
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


class SimpleColorGame(gym.Env):
    # Initializes the Env, including observation space and action space. This one initializes the Observation space as a grid
    # of boxes with colors assigned to them, and the action space as the movement of the agent along the grid.
    def __init__(self, size=2, step_limit=200):
        # The size of one side of the square grid. It will be NxN squares in area, where N is self._size
        self._size = size
        self._num_colors = size**2

        # This is a time limit on the number of steps the agent is allowed to take in the game. This is necessary to
        # prevent the game from running forever if the agent's policy prevents it from moving or reaching the target.
        self._step_limit = step_limit
        # Integer to keep track of the number of steps taken in a particular iteration of the game
        self._step_count = 0

        # The agent location is stored inside of a local variable.
        self._agent_location = np.array([-1, -1], dtype=np.int32)

        # The colors of the boxes are also stored in a local variable. These colors are randomized on start-up. For this
        # version of the game, I will substitute integer values for colors.
        self._square_colors = np.arange(self._num_colors).reshape(size, size)

        # The target color will be a random number between 1 and 4. This number will be initialized during the reset() method.
        self._target_color = np.random.randint(0, self._num_colors)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "square colors": gym.spaces.Box(
                    0, self._num_colors - 1, shape=(size, size), dtype=int
                ),
                "target color": gym.spaces.Discrete(self._num_colors),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    # Helper method used to get the observation from the state, useful in reset and step methods. This version returns
    # the properties of agent location, square colors, and the target color.
    def _get_obs(self):
        return {
            "agent location": self._agent_location,
            "square colors": self._square_colors,
            "target color": self._target_color,
        }

    # Helper method used to get auxiliary information from the state. Currently returns nothing.
    def _get_info(self):
        info = {"info": None}
        return info

    # Helper method for calculating the reward from the state. This will be useful as I can override it in child classes.
    def _get_reward(self):
        reward = (
            1
            if (self._square_colors[tuple(self._agent_location)] == self._target_color)
            else 0
        )
        return reward

    # Reset the environment to an initial configuration. The initial state may involve some randomness, so the seed argument
    # is used to guarantee an identical initial state whenever reset() is called with that seed. Options is a dict containing
    # any additional parameters we might want to specify during the reset.
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        # Firstly, we will call this method to seed self.np_random with the seed argument if given.
        super().reset(seed=seed)

        # Reset the step count to 0 for the new iteration of the game
        self._step_count = 0

        # Now randomly generate a starting location for the agent using self.np_random. We generate an array of size two
        # representing the agent's starting coordinates.
        self._agent_location = self.np_random.integers(0, self._size, size=2)

        # Generate a random permutation of the square colors, and reshape them into a sizeXsize grid.
        self._square_colors = self.np_random.permutation(self._num_colors).reshape(
            self._size, self._size
        )

        # Now we generate the target color, which is a random integer from 0 to self._num_colors inclusive.
        self._target_color = self.np_random.integers(0, self._num_colors)

        # Now we can return the observation and auxiliary info
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Takes an action as input and updates the state of the Env according to that Action. Step then returns an observation
    # containing the new Env state, as well as some other additional variables and info.
    def step(self, action):
        # First, iterate the step count by one
        self._step_count += 1

        # Next, we convert our action to a direction.
        direction = self._action_to_direction[action]

        # Then we add the direction coordinates to the agend coordinates to get the new agent location. We must clip the
        # agent location at the Box boundary, so the agent's coordinates are within 0 and self._size-1.
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self._size - 1
        )

        # Now we terminate the game and give the agent a reward if the square it's standing on is the target color.
        terminated = (
            self._square_colors[tuple(self._agent_location)] == self._target_color
        )

        # We also truncate the game if self._step_count > self._step_limit.
        truncated = self._step_count > self._step_limit

        # Reward is 1 if we are on the target color square, otherwise 0
        reward = self._get_reward()

        # Finally, use the helper functions to generate Obs and Info.
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
#This will be the v1 of the game, with a stop function included.
class StopOnColorGame(gym.Env):
    # Initializes the Env, including observation space and action space. This one initializes the Observation space as a grid
    # of boxes with colors assigned to them, and the action space as the movement of the agent along the grid.
    def __init__(self, size=2, step_limit=200):
        self._size = size
        self._num_colors = size**2
        self._step_limit = step_limit
        self._step_count = 0
        self._stopped = False;

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._square_colors = np.arange(self._num_colors).reshape(size, size)
        self._target_color = np.random.randint(0, self._num_colors)
        
        self.observation_space = gym.spaces.Dict(
            {
                "agent location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "square colors": gym.spaces.Box(
                    0, self._num_colors - 1, shape=(size, size), dtype=int
                ),
                "target color": gym.spaces.Discrete(self._num_colors),
            }
        )
        self.action_space = gym.spaces.Discrete(5)

        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
            4: np.array([0,0]), #stop
        }

    # Helper method used to get the observation from the state, useful in reset and step methods. This version returns
    # the properties of agent location, square colors, and the target color.
    def _get_obs(self):
        return {
            "agent location": self._agent_location,
            "square colors": self._square_colors,
            "target color": self._target_color,
        }

    # Helper method used to get auxiliary information from the state. Currently returns nothing.
    def _get_info(self):
        info = {"info": None}
        return info

    # Helper method for calculating the reward from the state. This will be useful as I can override it in child classes.
    def _get_reward(self):
        reward = 1 if (self._stopped == True and self._square_colors[tuple(self._agent_location)] == self._target_color) else 0
        return reward

    def _get_terminated(self):
        return self._stopped

    def _get_truncated(self):
        return self._step_count > self._step_limit;
    # Reset the environment to an initial configuration. The initial state may involve some randomness, so the seed argument
    # is used to guarantee an identical initial state whenever reset() is called with that seed. Options is a dict containing
    # any additional parameters we might want to specify during the reset.
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        # Firstly, we will call this method to seed self.np_random with the seed argument if given.
        super().reset(seed=seed)

        # Reset the step count to 0 for the new iteration of the game
        self._step_count = 0

        # Reset the _stopped flag
        self._stopped = False

        # Now randomly generate a starting location for the agent using self.np_random. We generate an array of size two
        # representing the agent's starting coordinates.
        self._agent_location = self.np_random.integers(0, self._size, size=2)

        # Generate a random permutation of the square colors, and reshape them into a sizeXsize grid.
        self._square_colors = self.np_random.permutation(self._num_colors).reshape(
            self._size, self._size
        )

        # Now we generate the target color, which is a random integer from 0 to self._num_colors inclusive.
        self._target_color = self.np_random.integers(0, self._num_colors)

        # Now we can return the observation and auxiliary info
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Takes an action as input and updates the state of the Env according to that Action. Step then returns an observation
    # containing the new Env state, as well as some other additional variables and info.
    def step(self, action):
        # First, iterate the step count by one
        self._step_count += 1

        # Next, we convert our action to a direction.
        direction = self._action_to_direction[action]

        #Finally, we check for the STOP action.
        if action == 4:
            self._stopped = True

        # Then we add the direction coordinates to the agend coordinates to get the new agent location. We must clip the
        # agent location at the Box boundary, so the agent's coordinates are within 0 and self._size-1.
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self._size - 1
        )
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


# I'll make this class extend the SimpleColorGame class. We just want to modify the reward function to subtract 0.1 for each step taken.
class TimedColorGame(SimpleColorGame):
    # Override the _get_reward() method to subtract 1/self._step_limit from the normal reward. This creates an incentive to reach the target
    # square quickly.
    def _get_reward(self):
        reward = (
            (1.0 - self._step_count / self._step_limit)
            if (self._square_colors[tuple(self._agent_location)] == self._target_color)
            else 0
        )
        return reward

class EmbodiedCommunicationGame(ParallelEnv):
    """
    My first naive implementation of the ECG in Gymnasium's PettingZoo.
    """
    metadata = { "render_modes": [],
        "name":"EmbodiedCommunicationGame-v0"}
    
    def __init__(self, step_limit = 100):
        """
        Variables to instantiate:
        -2 Grids 
        -2 Agent locations
        -Color List
        -Step Count
        -Step Limit
        -Possible Agents (agents)?
        We will not initialize the agent locations or grid color patterns here, that will occur in self.reset().
        We instantiate the variables here to keep better track of them.
        """

        #public variables
        self.step_limit = step_limit
        self.possible_agents = ["agent1", "agent2"]
        self.render_mode = None
        self.spec = None
        
        #private variables
        self._step_count = 0

        #Grids, agent coords, and agents committed. To make step() easier, I'm going to put each of these into it's own
        # dict, with keys corresponding to agent names.
        self._color_grids = {a : np.zeros((2,2)) for a in self.possible_agents}

        self._agent_coords = {a : np.array([-1,-1], dtype=np.int32) for a in self.possible_agents}

        #Track whether agents are committed.
        self._agents_committed = {a : False for a in self.possible_agents}
        
        #Dictionary to map action space to a direction on the Gridworld.
        self._action_to_direction = {
            0: np.array([0,-1]), #Down
            1: np.array([1,0]), #Right
            2: np.array([0,1]), #Up
            3: np.array([-1,0]), #Left
            4: np.array([0,0]), #Commit, for redundancy.
        }
        
        #Color list
        self._colors = np.arange(8)

        self._observation_spaces = { a : {
            "myCoords" : self._agent_coords[a],
            "allCoords" : [self._agent_coords[a] for a in self.possible_agents],
            "myColors" : self._color_grids[a],
        } for a in self.possible_agents }
            
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        initialize the Env for play.
        Variables to initialize:
        -Agent locations
        -Color arrangements for each 2x2 grid
        -step-count
        -Agent committed flag
        """
        #call super().reset() with seed to set self.np_random with seed.
        self._seed(seed)

        #Reset flags
        self._step_count = 0

        #Instantiate self.agents to be a copy (by value) of self.possible_agents.
        self.agents = copy(self.possible_agents)

        #Iteratively generate agent data, including grid color, agent location, and agent committed.
        for a in self.agents:
            self._color_grids[a] = self.np_random.choice(self._colors, size = (2,2), replace = False)
            self._agent_coords[a] = self.np_random.integers(0,2, size=(2,))
            self._agents_committed[a] = False
        
        #now to return obs and infos
        obs = self._get_obs()
        infos = self._get_infos()
        # print("obs: ",obs,"infos: ",infos, sep="\n\n")
        
        return obs, infos

    def step(self, actions):
        """
        update the Env according to the actions of each agent.
        returns:
        obs{}, rewards{}, terminateds{}, truncateds{}, infos{}
        """
        
        for (agent, action) in actions.items():
            
            #update the agent committed status first.
            if action == 4:
                self._agents_committed[agent] = True
                
            #Update agent's coordinates only if agent not committed. Checking action != 4 for redundancy.
            if (self._agents_committed[agent] == False and action != 4):
                direction = self._action_to_direction[action]
                self._agent_coords[agent] = np.clip(self._agent_coords[agent] + direction, 0, 1)

        self._step_count += 1
        
        #Now calculate the return values.
        obs = self._get_obs()
        rewards, terminateds, truncateds = self._get_rewards_terminateds_truncateds()
        infos = self._get_infos()

        if all(terminateds.values()) or all(truncateds.values()):
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)

    def _get_obs(self):
        #I tried getting fancy and generating this with a loop, but it was causing problems, so for now I'm just going
        # to make it by hand.
        a1 = self.agents[0]
        a2 = self.agents[1]
        obs = {
            a1 : {
                "myCoords" : self._agent_coords[a1],
                "theirCoords" : self._agent_coords[a2],
                "myColors" : self._color_grids[a1],
            },
            a2 : {
                "myCoords" : self._agent_coords[a2],
                "theirCoords" : self._agent_coords[a1],
                "myColors" : self._color_grids[a2],
            },
        }
                     
        return obs

    def _get_rewards_terminateds_truncateds(self):
        """
        this function will return the rewards, terminateds, and truncateds state.
        There is specific logic relating rewards to truncation and termination.
        """   
        #we're basically building up a long boolean check to see if the agents have both committed to squares of the
        # same color. For that reason, I'm going to shorten some of these variable names to make the boolean statement
        # more readable.
        a1 = self.agents[0]
        a2 = self.agents[1]
        grid1 = self._color_grids[a1]
        grid2 = self._color_grids[a2]
        a1_coords = self._agent_coords[a1]
        a2_coords = self._agent_coords[a2]

        #Common/base case for terminated/reward is Terminated = False and Reward = 0. I want to also punish agents for
        # not committing to speed up the process of committing at some point.
        rewards = {a: (-1 if self._step_count > self.step_limit else 0) for a in self.agents}
        terminateds = {a: False for a in self.agents}
        truncateds = {a : self._step_count > self.step_limit for a in self.agents}

        #If both of the agents have committed, Terminated = True.
        if (self._agents_committed[a1] == True and self._agents_committed[a2] == True):
            terminateds = {a : True for a in self.agents}
            
            #If the color of a1's square matches the color of a2's square, reward = 1. Noteably, we only reward
            # the agents for having matching square colors if both agents have chosen to commit to their squares.
            if (grid1[tuple(a1_coords)] == grid2[tuple(a2_coords)]):
                rewards = {a : 1 for a in self.agents}
        
        return rewards, terminateds, truncateds

    #TODO: Learn more about what I can use info for. this might be a way to give the agents some form of memory.
    def _get_infos(self):
        return {a : {} for a in self.agents}

    #Farama recommends using memo-ized functions for returning action_space and observation_space rather than public
    # or private class variables.
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent):
        return gym.spaces.Dict({
            "myCoords": gym.spaces.Box(0,2, shape = (2,), dtype=int),
            "theirCoords": gym.spaces.Box(0,2, shape = (2,), dtype=int),
            "myColors": gym.spaces.Box(0,8, shape = (2,2), dtype=int),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self,agent):
        return Discrete(5)

class SimpleEmbodiedCommunicationGame(ParallelEnv):
    """
    My second attempt at creating emergent communication in the RL models. This game differs from the original in 2 ways:
    1. The grids are smaller with less possible colors
    2. The agents are fed the previous agents' previous 10 locations in addition to their current location. This allows the models to interpret
    behavior over time as communiction.
    """
    metadata = { "render_modes": [],
        "name":"SimpleEmbodiedCommunicationGame-v0"}
    
    def __init__(self, step_limit = 100):
        """
        Variables to instantiate:
        -2 2x1 Grids 
        -2 Agent locations (int)
        -Color List (3 colors)
        -Step Count
        -Step Limit
        -Possible Agents (agents)?
        -Lagging Memory (dict{a: np.array(1,10)})
        We will not initialize the agent locations or grid color patterns here, that will occur in self.reset().
        We instantiate the variables here to keep better track of them.
        """

        #public variables
        self.step_limit = step_limit
        self.possible_agents = ["agent1", "agent2"]
        self.render_mode = None
        self.spec = None
        
        #private variables
        self._step_count = 0

        #Grids, agent coords, and agents committed. To make step() easier, I'm going to put each of these into it's own
        # dict, with keys corresponding to agent names.
        self._color_grids = {a : np.zeros((2,)) for a in self.possible_agents}

        self._agent_coords = {a : -1 for a in self.possible_agents}

        #Track whether agents are committed.
        self._agents_committed = {a : False for a in self.possible_agents}
        
        #Lagging Memory for agent locations
        self._lagging_memory = {a : np.zeros((10,)) for a in self.possible_agents}
        
        #Dictionary to map action space to a direction on the Gridworld.
        self._action_to_direction = {
            0: -1, #Left,
            1: 1, #Right,
            2: 0, #Chill,
            3: 0, #Commit
        }
        
        #Color list
        self._colors = [[0,1],[0,2],[1,2]]
            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        initialize the Env for play.
        Variables to initialize:
        -Agent locations
        -Color arrangements for each 2x2 grid
        -step-count
        -Agent committed flag
        """
        #call super().reset() with seed to set self.np_random with seed.
        self._seed(seed)

        #Reset flags
        self._step_count = 0

        #Instantiate self.agents to be a copy (by value) of self.possible_agents.
        self.agents = copy(self.possible_agents)
        
        #generate color grids by first generating the grid selections for each agent
        grid_choices = self.np_random.choice(3, size=(2,),replace=False)

        #Iteratively generate agent data, including grid color, agent location, and agent committed.
        for idx, a in enumerate(self.agents):
            self._color_grids[a] = self.np_random.permutation(self._colors[grid_choices[idx]])
            self._agent_coords[a] = self.np_random.integers(2)
            self._agents_committed[a] = False
            self._lagging_memory[a].fill(-1)
        
        #now to return obs and infos
        obs = self._get_obs()
        infos = self._get_infos()
        # print("obs: ",obs,"infos: ",infos, sep="\n\n")
        
        return obs, infos

    def step(self, actions):
        """
        update the Env according to the actions of each agent.
        returns:
        obs{}, rewards{}, terminateds{}, truncateds{}, infos{}
        """
        for (agent, action) in actions.items():
            #roll the lagging memory
            self._lagging_memory[agent] = np.roll(self._lagging_memory[agent],1)
            
            #update the agent committed status first.
            if action == 3:
                self._agents_committed[agent] = True
                
            #Update agent's coordinates only if agent not committed. Checking action != 4 for redundancy.
            if (self._agents_committed[agent] == False and action != 3):
                direction = self._action_to_direction[action]
                self._agent_coords[agent] = np.clip(self._agent_coords[agent] + direction, 0, 1)
                self._lagging_memory[agent][0] = self._agent_coords[agent]

        self._step_count += 1
        
        #Now calculate the return values.
        obs = self._get_obs()
        rewards, terminateds, truncateds = self._get_rewards_terminateds_truncateds()
        infos = self._get_infos()

        if all(terminateds.values()) or all(truncateds.values()):
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)

    def _get_obs(self):
        #I tried getting fancy and generating this with a loop, but it was causing problems, so for now I'm just going
        # to make it by hand.
        a1 = self.agents[0]
        a2 = self.agents[1]
        obs = {
            a1 : {
                "myCoords" : self._agent_coords[a1],
                "theirCoords" : self._lagging_memory[a2],
                "myColors" : self._color_grids[a1],
            },
            a2 : {
                "myCoords" : self._agent_coords[a2],
                "theirCoords" : self._lagging_memory[a1],
                "myColors" : self._color_grids[a2],
            },
        }
                     
        return obs

    def _get_rewards_terminateds_truncateds(self):
        """
        this function will return the rewards, terminateds, and truncateds state.
        There is specific logic relating rewards to truncation and termination.
        """   
        #we're basically building up a long boolean check to see if the agents have both committed to squares of the
        # same color. For that reason, I'm going to shorten some of these variable names to make the boolean statement
        # more readable.
        a1 = self.agents[0]
        a2 = self.agents[1]
        grid1 = self._color_grids[a1]
        grid2 = self._color_grids[a2]
        a1_coords = self._agent_coords[a1]
        a2_coords = self._agent_coords[a2]

        #Common/base case for terminated/reward is Terminated = False and Reward = 0. I want to also punish agents for
        # not committing to speed up the process of committing at some point.
        rewards = {a: (-1 if self._step_count > self.step_limit else 0) for a in self.agents}
        terminateds = {a: False for a in self.agents}
        truncateds = {a : self._step_count > self.step_limit for a in self.agents}

        #If both of the agents have committed, Terminated = True.
        if (self._agents_committed[a1] == True and self._agents_committed[a2] == True):
            terminateds = {a : True for a in self.agents}
            
            #If the color of a1's square matches the color of a2's square, reward = 1. Noteably, we only reward
            # the agents for having matching square colors if both agents have chosen to commit to their squares.
            if (grid1[a1_coords] == grid2[a2_coords]):
                rewards = {a : 1 for a in self.agents}
        
        return rewards, terminateds, truncateds

    #TODO: Learn more about what I can use info for. this might be a way to give the agents some form of memory.
    def _get_infos(self):
        return {a : {} for a in self.agents}

    #Farama recommends using memo-ized functions for returning action_space and observation_space rather than public
    # or private class variables.
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent):
        return gym.spaces.Dict({
            "myCoords": gym.spaces.Discrete(2),
            "theirCoords": gym.spaces.Box(0,2, shape = (10,), dtype=np.float64),
            "myColors": gym.spaces.Box(0,3, shape = (2,), dtype=np.float64),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self,agent):
        return Discrete(4)