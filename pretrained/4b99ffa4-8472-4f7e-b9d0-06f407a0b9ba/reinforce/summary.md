# Reinforcement Learning Simulation Summary


---

## Simulation Information

- **Method :** REINFORCE
- **Date Time:** 23-02-2026 19:05:40
- **Device:** CUDA
- **Physics Timestamp:**  1/60s
- **Physics Substeps:** 30
- **Total Simulation Time:** 00h:43m:22s
- **Number of Steps per Episode:** 540
- **Number of Episodes:** 3165
  

---

## Creature Information

- **Creature ID:** 4b99ffa4-8472-4f7e-b9d0-06f407a0b9ba
- **Joints:** 7
- **Bones:** 6
- **Muscles:** 5
- **Joint Degrees Min:** 5
- **Joint Degrees Max:** 20
- **Scale:** 15
  
![Creature image](assets/creature.png)


---

## Method Description

REINFORCE is a policy gradient algorithm that
updates agent policy parameters by increasing the probability of actions that resulted in higher cumulative rewards,
directly maximizing expected returns without needing a value function.

  
## Network Configuration

- **Method:** Vanilla Policy Gradient (REINFORCE)
- **Inputs:** 52
- **Outputs:** 5
### Network Architecture

- **Actor** 
	- Layer Widths: `[52, 30, 30, 30, 5]`
	- Learning Rate: `1e-03`
	- Activation: `Leaky ReLU`
	- Optimizer: `Adam`
	- Number of Parameters: `3610`

### Hyperparameters

- Batch Size: `12`
- Discount Factor: `0.99`

  

---

## Results

### Distances

From the distances graph we can see when the final distance matches the maximum distance.
             From this graph we can conclude in which episodes the possibility of creature going further was limited by time and not by fitness.

![Max distance graph](assets/distances.png)

This graph show how many times per episode is maximum distance equal to final distance. 
            If number is growing we can consider that model is improving.

![Max reached per batch](assets/distances_bar.png)

### Loss per backprop

Main metric for determining success/fitness. Tells us how much agent is improving over time.(Loss is calculated only during backprop)

![Loss graph](assets/loss.png)

### Time per episode

Here we can see spikes in time when parameter update is being called and also times when the episode is terminated prematurely.(Letting the simulation run visually will be visible because of the longer time)

![Time per episode graph](assets/time.png)

### Rewards per episode

Main goal of any method maximize rewards. Per episode average is displayed.

![Rewards per episode graph](assets/rewards.png)

### Activation per episode

Number increases as the muscle is activated more strongly. Per episode average is displayed.

![Activation per episode graph](assets/activation.png)



  

---

### Best vs First Episode

|  | **First Episode**  | **Best Episode**  |
| --- | --- | --- |
| **Episode Index**  | 0 | 3090 |
| **Max Distance**  | 0.93m | 20.75m |
| **Last Distance**  | 0.62m | 20.75m |
| **Average Activation**  | 2.77 | 3.28 |
| **Average Rewards**  | -1.62 | 35.15 |
| **Time**  | 00m:01s | <1s |

### Graph Comparison

Activation per neuron and rewards per step.

![](assets//episode_comparison.png)

  

---

## Notes

- This report was generated automatically after simulation completion.
- All conclusions should be made visually by the reader.
- For more information go to the [github repository](https://github.com/ajromen/Reinforcment-Learning-Simulation).
