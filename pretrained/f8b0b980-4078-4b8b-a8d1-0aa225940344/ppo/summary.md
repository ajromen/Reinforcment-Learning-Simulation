# Reinforcement Learning Simulation Summary


---

## Simulation Information

- **Method :** PPO
- **Date Time:** 08-02-2026 20:28:09
- **Device:** CUDA
- **Physics Timestamp:**  1/60s
- **Physics Substeps:** 30
- **Total Simulation Time:** 01h:04m:38s
- **Number of Steps per Episode:** 540
- **Number of Episodes:** 890
  

---

## Creature Information

- **Creature ID:** f8b0b980-4078-4b8b-a8d1-0aa225940344
- **Joints:** 7
- **Bones:** 6
- **Muscles:** 5
- **Joint Degrees Min:** 5
- **Joint Degrees Max:** 20
- **Scale:** 15
  
![Creature image](assets/creature.png)


---

## Method Description

Proximal Policy Optimization (PPO) is an on-policy reinforcement learning algorithm that optimizes a clipped surrogate objective to ensure stable policy updates. 
        In this simulation, PPO is used to control muscle activations directly, balancing forward locomotion with energy efficiency through activation penalties.

  
## Network Configuration

- **Method:** Proximal Policy Optimization (PPO)
- **Inputs:** 52
- **Outputs:** 5
### Network Architecture

- **Actor** 
	- Layer Widths: `[52, 30, 30, 30, 5]`
	- Learning Rate: `3e-04`
	- Activation: `Tanh`
	- Optimizer: `Adam`
	- Number of Parameters: `3610`

- **Critic** 
	- Layer Widths: `[52, 30, 30, 30, 1]`
	- Learning Rate: `1e-04`
	- Activation: `Leaky ReLU`
	- Optimizer: `Adam`
	- Number of Parameters: `3481`

### Hyperparameters

- Batch Size: `12`
- Discount Factor: `0.99`
- Clip Epsilon: `0.2`
- K Epochs: `10`
- Entropy Coefficient: `0.01`

  

---

## Results

### Distances

From the distances graph we can see when the final distance matches the maximum distance.
             From this graph we can conclude in which episodes the possibility of creature going further was limited by time and not by fitness.

![Max distance graph](assets/distances.png)

This graph show how many times per episode is maximum distance equal to final distance. 
            If number is growing we can consider that model is improving.

![Max reached per batch](assets/distances_bar.png)

### Times per episode

Here we can see spikes in time when parameter update is being called and also times when the episode is terminated prematurely.(Letting the simulation run visually will be visible because of the longer time)

![Last distance graph](assets/time.png)

### Activation per episode

Number increases as the muscle is activated more strongly. Per episode average is displayed.

![Last distance graph](assets/activation.png)



### Rewards per episode

Main goal of any method maximize rewards. Per episode average is displayed.

![Last distance graph](assets/rewards.png)

  

---

### Best vs First Episode

|  | **First Episode**  | **Best Episode**  |
| --- | --- | --- |
| **Episode Index**  | 0 | 847 |
| **Max Distance**  | 1.70m | 9.42m |
| **Last Distance**  | 1.60m | 9.42m |
| **Average Activation**  | 2.79 | 3.26 |
| **Average Rewards**  | 0.17 | 13.11 |
| **Time**  | 00m:01s | 00m:08s |

### Graph Comparison

Activation per neuron and rewards per step.

![](assets//episode_comparison.png)

  

---

## Notes

- This report was generated automatically after simulation completion.
- All conclusions should be made visually by the reader.
- For more information go to the [github repository](https://github.com/ajromen/Reinforcment-Learning-Simulation).
