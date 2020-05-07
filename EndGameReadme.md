##  ENDGAME



##        **Title: self-driving Car using TD3(Twin Delayed Deep Deterministic Policy Gradient Algorithm)**



**Objective:**

Objective of this work is to train a self-driving Car using TD3(Twin Delayed Deep Deterministic Policy Gradient Algorithm**) algorithm** in city map environment(car/map/sand) and make it to initialize in any location in map, travels to a point say Goal A and from there to travels to another point say goal B and return to a same initial position. 

Here Goal is a point in a city map and A & B can be any two points. 



### Problem Statement:

####  **THE ENDGAME PROJECT**

1. Refer to the code shared in Assignment 7. 

2. Now use the same environment (car/map/sand/etc), and move that to TD3

3. Record your best attempt (it will be evaluated on how good your car's driving was in this video)

4. Upload Video on YouTube, upload your code to GitHub along with Google-Colab code for A7Car+TD3. 

5. Copy the Video and Code link to this assignment. 

6. Done!

 

#### **EXPECTATION**

1. You are indeed using TD3

2. You are NOT using SENSORS

3. You are using Conv Network (very very simple one) on your sand (so you can use MNIST type too) to get the sensory data. 

4. You have understood the problem and actually have tried to solve this problem. 





**City Map Environment:**

**City Map:** image of a city map of size 1429x660 image(citymap.png) extracted using google map.

**Mask:** This is city map Image in which roads were represented in block line and rest of the image was represented as white image called Sand. (mask.png)

**Mask1**: 90-degree rotated mask image. kivy process this Mask1 Image.

**Car:** car object image of size 100x45 (car.png)

 **Car Initialization:** Car is always initialized at x=715, y=360 position in Map. Whenever the car completes its current episode it gets initialized from x=715, y=360. 

**Goal:** Goal is the destination location for the car supposed to reach. 

Car always start from its initial location to reach goal A, once it reaches goal A it starts reaching to goal B and then initial position.

**Goal1:** A location (x:1420, y:620) in MASK (90degree rotated map) image

**Goal2:** A location (x:9, y:85) in MASK (90degree rotated map) image

 

### Implementation:

 In Phase2Assignment7, I used DQN algorithm to train a self-driving car in city map environment, there sensors values in the cars were used for moving car from initial position to Goal A or Goal B. 

**Crop & Rotate:**

Approach I have applied was crop a sand image in gray scale and rotate it with the angle the car current position makes with X axis of the environment.

Sand image is 90-degree rotated image. So, we rotate sand image by 90 degree before it is interpolated.

Sand image was cropped by 60x60 and later it is scaled down to 28x28 image using Tensor interpolate method.

**TD3 Implementation:**

**State dimension:** It is 28x28 gray scale image that is cropped & rotate from sand image of size 60x60 .

**Max Action:** Chosen 5 degree a car can rotate.

**Orientation:** Used positive and negative orientation for model stability. 

**Action Dimension:** 1 

**Crop dimension:** cropped 60x60 gray scale sand image, rotated it by 90 degree  and down sized it to 28x28.

**Replay Buffer:**

TD3 uses experience replay where experience tuples are added to replay buffer and are randomly sampled from the replay buffer.

Experience tuple:  (state dim, next_state, orientation, next_orientation, action dim, reward, done)

Size : 1e6

**Actor & Critic CNN :**

TD3 concurrently leans Q-function and policy. It uses actor and critic approach where actor jobs is to specify the action based on current state of the environment and critic function is to specify the error to criticize the action made by actor.

TD3 has 6 networks actor, target actor, two critic network and two target critic network. 

 Network we used for this work is CNN (customized) that used to train MNIST data set and achieved target accuracy as 99.3% less than 15k parameters.



**Actor Network:**
Actor(
  (convolution_actor_module): ModuleList(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
    (4): ReLU()
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): AdaptiveAvgPool2d(output_size=(1, 1))
    (10): Flatten()
  )
  (linear): ModuleList(
    (0): Linear(in_features=18, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=8, bias=True)
    (3): ReLU()
    (4): Linear(in_features=8, out_features=1, bias=True)
  )
)

#### **Critic Network**



Critic(
  (convolution_critic1_module): ModuleList(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
    (4): ReLU()
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): AdaptiveAvgPool2d(output_size=(1, 1))
    (10): Flatten()
  )
  (linear_critic1_module): ModuleList(
    (0): Linear(in_features=19, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=8, bias=True)
    (3): ReLU()
    (4): Linear(in_features=8, out_features=1, bias=True)
  )
  (convolution_critic2_module): ModuleList(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
    (4): ReLU()
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (8): ReLU()
    (9): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): AdaptiveAvgPool2d(output_size=(1, 1))
    (11): Flatten()
  )
  (linear_critic2_module): ModuleList(
    (0): Linear(in_features=19, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=8, bias=True)
    (3): ReLU()
    (4): Linear(in_features=8, out_features=1, bias=True)
  )
)

**Episode Done:**

- Episode is Done, When the car reaches to Environment boundary.
- Episode is Done, when episode time stamp is 1000 start time stamp is > total time stamp(10000)
- Episode is Done,  when  car moves 50% of times in sand then it is penalized reward -= 0.2 and self.sand_penalty += 0.2
- Episode is Done, In  single episode car moves 200 times in sand then episode is marked done. 



**Penalty and Rewards**

There are two penalty used in this approach called sand penalty and living penalty.

- if the car moves on sand, its sand penalty is  0.5 (sand_penalty) and rewarded -0.5

- if the car moves on road, its living penalty incremented by 1 if this moves towards goal otherwise living penalty is 0.5 and rewarded -0.5

- if distance between car position and goal is very less below formula is used for final reward 	reward -= float(100 * (1/distance))  and living_penalty -= float(100 * (1/distance))

  

  ##### **Goals:**

if the distance between car position and goal A is less than 10, then it is rewarded and living penalty has been increased.

Next Goal for Car has been swapped from Goal A to Goal B.



#### Output:



#### Future Improvements:

- **Train TD3 in GPU:** I have implemented this work in win10 and trained TD3 in CPU, during training time Kivy environment was not responding, it could have been much better if this is implemented in GPU.

- **Training Improvements:** During TD3 Training in main loops, following improvements can be experimented in future for better training.

  - Reward & penalization policy can be tweaked further when car moves on sand in general.
  - Can make a car to change angles, when its last couple of moves in sand.

- **Random Initialization:** Car can be initialized from random location rather than center.

- **Episode Done:** 

  - When car hits environment boundary, we mark it as done here we can consider changing car angle and make it to complete at least 1000 steps per episode for better learning.

    

#### Conclusion:

Self Driving car has been trained using TD3 algorithm in city map environment that starts from center of the map and can reach to goal A successfully.







