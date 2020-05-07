# Self Driving Car

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage

# Importing the TD3 object from our TD3_DNN.py
from TD3DNN import TD3
from TD3DNN import ReplayBuffer

# Importing torch packages
import torch
import torch.nn.functional as F

import os
import time

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Define TD3 variables
orientation = 0  # +ve & -ve for model stability
max_action = 5  # 5 degree rotation
crop_dim = 60
state_dim = 28  # state dimension is 60x60 image having car at the center with one channel for grayscale
action_dim = 1
latent_dim = 16

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(state_dim, action_dim, max_action, latent_dim)  # TD3(5,3,0.9)

scores = []  # Not using
core_img = CoreImage("./images/MASK1.png")
mask = cv2.imread('./images/mask.png', 0)

# Initializa Replay buffers with 1e6 random transitions
# Replay Buffer defined in TD3_DNN file
replay_buffer = ReplayBuffer()

# Initializing the map
first_update = True

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))

    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img) / 255
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = float(rotation)
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10,
                                  int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 10:int(self.sensor2_x) + 10,
                                  int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 10:int(self.sensor3_x) + 10,
                                  int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.
        if self.sensor1_x > longueur - 10 or self.sensor1_x < 10 or self.sensor1_y > largeur - 10 or self.sensor1_y < 10:
            self.signal1 = 10.
        if self.sensor2_x > longueur - 10 or self.sensor2_x < 10 or self.sensor2_y > largeur - 10 or self.sensor2_y < 10:
            self.signal2 = 10.
        if self.sensor3_x > longueur - 10 or self.sensor3_x < 10 or self.sensor3_y > largeur - 10 or self.sensor3_y < 10:
            self.signal3 = 10.


# function to extract image and rotate
# img - MASK image represented as numpy array with 0's & 1's
# angle - Angle car makes with x axis of environment
# center - center of enviroment
# crop_size - size of the cropped Image
def get_corp_and_rotate_mask_imag(img, angle, center, crop_size, fill_with=255):
    # Since sand image is 90 degree flip of actual map. so we add 90 with angle.
    angle = angle + 90
    center[0] -= 0
    # MASK Image: Sand is in while background and roads in block line.
    img = np.pad(img, crop_size, 'constant', constant_values=fill_with)
    # Crop an image of crop_size 1.6 times of crop_size.
    init_size = 1.6 * crop_size
    center[0] += crop_size
    center[1] += crop_size
    # Crop an Image of crop_size
    half_init_size = (init_size / 2)
    cropped = img[int(center[0] - half_init_size): int(center[0] + half_init_size),
              int(center[1] - half_init_size): int(center[1] + half_init_size)]

    # Rotate cropped image by 90 degree
    rotated = ndimage.rotate(cropped, angle, reshape=False, cval=255.0)
    y, x = rotated.shape
    half_crop_size = (crop_size / 2)
    final = rotated[int(y / 2 - half_crop_size):int(y / 2 + half_crop_size),
            int(x / 2 - half_crop_size):int(x / 2 + half_crop_size)]
    final = torch.from_numpy(np.array(final)).float().div(255)
    final = final.unsqueeze(0).unsqueeze(0)
    final = F.interpolate(final, size=(state_dim, state_dim))
    return final.squeeze(0)


###################### GAME CLASS #######################
# Creating the game class

class Game(Widget):
    car = ObjectProperty(None)

    # intialize TD3 variables with default values
    total_timesteps = 0
    episode_num = 0
    done = True
    max_timesteps = 100000
    # initalize state with zero
    state = torch.zeros([1, state_dim, state_dim])
    episode_reward = 0
    episode_timesteps = 0
    sand_counter = 0
    sand_penalty = 0
    living_penalty = 0
    lp_counter = 0

    expl_noise_vals = np.linspace(max_action, int(max_action / 100), num=int(max_timesteps / 4000), endpoint=True,
                                  retstep=False, dtype=None, axis=0)

    ###
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        # global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global last_action
        global last_distance_travelled
        global orientation

        # Inital values for Training TD3  Network

        start_timesteps = 10000  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
        batch_size = 30  # crop_size of the batch
        discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
        tau = 0.005  # Target network update rate
        policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
        noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
        policy_freq = 2  # Number of iterations to wait before the policy network (Actor model) is updated
        expl_noise = 0

        sand_time = []

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        # We start the main loop over 500,000 timesteps
        if self.total_timesteps < self.max_timesteps:
            # If the episode is done
            if self.done:
                # If we are not at the very beginning, we start the training process of the model
                if self.total_timesteps != 0:
                    distance_travelled = np.sqrt((self.car.x - 715) ** 2 + (self.car.y - 360) ** 2)
                    distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
                    s_reward = round(
                        float(self.episode_reward * self.sand_penalty / (self.sand_penalty + self.living_penalty)), 2)
                    l_reward = round(
                        float(self.episode_reward * self.living_penalty / (self.sand_penalty + self.living_penalty)), 2)
                    print("Steps:", self.total_timesteps, "Episode:", self.episode_num, "Reward:", self.episode_reward,
                          "EpiSteps:", self.episode_timesteps, "DistCovered:", round(float(distance_travelled), 2),
                          "DistLeft:", round(float(distance), 2), "Sand_P:", s_reward, "Living_P:", l_reward)
                # Train TD3 model when start_timesteps > total_timesteps(10k)
                if self.total_timesteps > start_timesteps:
                    print("TD3 Training: ", self.episode_timesteps)
                    start_time = time.time()
                    brain.train(replay_buffer, self.episode_timesteps, batch_size, discount, tau, policy_noise,
                                noise_clip, policy_freq)
                    end_time = time.time()
                    print("TD3 Train time: {} mins".format((end_time - start_time) / 60))

                # update car position to center for every episode
                self.car.x = 715
                self.car.y = 360
                self.car.velocity = Vector(6, 0)
                xx = goal_x - self.car.x
                yy = goal_y - self.car.y
                orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
                orientation = [orientation, -orientation]

                # initialise 1st state after done, move it towards orientation
                self.car.angle = 0
                self.state = get_corp_and_rotate_mask_imag(mask, self.car.angle, [self.car.x, self.car.y], crop_dim)

                self.done = False
                last_action = [0]
                last_distance_travelled = 0
                # Set rewards and episode timesteps to zero
                self.episode_reward = 0
                self.episode_timesteps = 0
                self.episode_num += 1
                self.sand_counter = 0
                self.lp_Counter = 0
                self.living_penalty = 0
                self.sand_penalty = 0

            # Before 10000 timesteps, we play random actions based on uniform distn
            if self.total_timesteps < start_timesteps:
                action = [np.random.uniform(-max_action, max_action)]

            else:  # After 10000 timesteps, we switch to the model
                action = brain.select_action(self.state, np.array(orientation))
                expl_noise = self.expl_noise_vals[int(self.total_timesteps / 4000)]
                action = (action + np.random.normal(0, expl_noise)).clip(-max_action, max_action)

                if round(abs(float(action[0]) - float(last_action[0]))) > 0.5:
                    action[0] = (action[0] + last_action[0]) / 2

            self.car.move(action[0])
            new_state = get_corp_and_rotate_mask_imag(mask, self.car.angle, [self.car.x, self.car.y], crop_dim)

            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            new_orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
            new_orientation = [new_orientation, -new_orientation]
            distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

            sand_time = []

            # evaluating reward and done
            if sand[int(self.car.x), int(self.car.y)] > 0:  # and self.total_timesteps < start_timesteps:
                self.car.velocity = Vector(1, 0).rotate(self.car.angle)  # 0.5
                self.sand_counter += 1
                reward = -0.5
                self.done = False
                self.sand_penalty += 0.5


            else:  # otherwise
                self.car.velocity = Vector(3, 0).rotate(self.car.angle)  # 1
                self.sand_counter = 0
                reward = -0.5  # living penalty
                self.living_penalty += 0.5

                if distance < last_distance:
                    reward += 1
                    self.living_penalty -= 1

            if (self.car.x < 5) or (self.car.x > self.width - 5) or (self.car.y < 5) or (
                    self.car.y > self.height - 5):  # crude way to handle model failing near boundaries
                self.done = True
                reward -= 0.5
                self.living_penalty += 0.5

            if distance < 10:
                reward += float(100 * (1 / distance))  # 0.2
                self.living_penalty -= float(100 * (1 / distance))
                if swap == 1:
                    goal_x = 1420  # 260
                    goal_y = 622  # 40
                    swap = 0
                    self.done = False
                    print("Goal: Reached at x={},y={}".format(goal_x, goal_y))
                else:
                    goal_x = 9  # 421
                    goal_y = 85  # 1073
                    swap = 1
                    self.done = True
                    print("Goal: Reached at x={},y={}".format(goal_x, goal_y))
            last_distance = distance

            # additional rewards and punishments:

            # End episode if car moves >200 times in sand.
            if self.sand_counter >= 200:
                reward -= 0.2
                self.sand_penalty += 0.2
                self.done = True
                print("Episode done:{}".format(self.sand_penalty))

            # End episode if car moves >50% times in sand.
            if self.sand_counter > 200 and self.episode_timesteps % self.sand_counter >= 50:
                reward -= 0.2
                self.sand_penalty += 0.2
                self.done = True

            # punish roundabout circles
            if abs(float(action[0]) - float(last_action[0])) / max_action < 0.2:
                reward -= 0.5
                self.living_penalty += 0.5


            distance_travelled = np.sqrt((self.car.x - 715) ** 2 + (self.car.y - 360) ** 2)
            if distance_travelled < last_distance_travelled:
                self.lp_counter += 1

            if self.lp_counter == 20:
                reward -= 0.5
                self.living_penalty += 0.5

            # We increase the total reward
            self.episode_reward += reward

            # end episode after some timesteps
            if self.episode_timesteps >= 1000:
                reward -= -1
                self.done = True

            sand_time.append(self.sand_counter)  # not used

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((self.state, new_state, orientation, new_orientation, action, reward, self.done))
            # print(self.state, new_state, action, reward, self.done)
            self.state = new_state
            self.orientation = new_orientation
            self.episode_timesteps += 1
            self.total_timesteps += 1
            last_action = action
            last_distance_travelled = distance_travelled


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8") * 255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 1

            last_x = x
            last_y = y


# Adding the API Buttons (clear, save and load)

class CarApp(App):
    directory = os.getcwd()
    file = "td3_model"

    def build(self):
        parent = Game()
        parent.serve_car()
        # Clock.max_iteration = 5
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear', size=(50, 50))
        savebtn = Button(text='save', pos=(parent.width, 0), size=(50, 50))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0), size=(50, 50))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving TD3 brain models at dir={} file={}".format(self.directory, self.file))
        brain.save(self.directory, self.file)
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("Loading last saved TD3 brain models from dir={} file={}".format(self.directory, self.file))
        brain.load(self.directory, self.file)


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
