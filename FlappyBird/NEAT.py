# pip3 install gym (openAI gym)
# pip3 install neat-python

# apt install xvfb ffmpeg xorg-dev libsdl2-dev cmake
# pip3 install gym[box2d]
import neat, gym

'''
env = gym.make("CartPole-v1")
observation = env.reset()
print(observation)
print(env.action_space)
done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(env.action_space.sample())
    
    env.render()
'''
    
# env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0f')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()