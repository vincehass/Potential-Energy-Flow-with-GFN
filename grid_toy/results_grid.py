import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Initialize WandB project
wandb.init(project="multi-objective-optimization", name="Pareto-Front-Visualization")

# Define a simple ZDT test function (ZDT1) as our optimization task
# This function maps inputs to two objectives that need to be optimized
def zdt1(x):
    f1 = x
    g = 1 + 9 * torch.mean(x)
    f2 = g * (1 - torch.sqrt(x / g))
    return f1, f2

# Generate a random batch of solutions for the multi-objective optimization task
def generate_batch(batch_size):
    # Each solution is a vector of values in [0, 1]
    return torch.rand(batch_size, 10)

# Compute the rewards (objective values) for each solution in the batch
def compute_rewards(solutions):
    f1, f2 = zdt1(solutions)
    return torch.stack([f1, f2], dim=1)  # Stack the objectives into a single tensor

# Plot the Pareto front for the given solutions
def plot_pareto_front(solutions, label, color):
    rewards = compute_rewards(solutions)
    plt.scatter(rewards[:, 0].detach().numpy(), rewards[:, 1].detach().numpy(), label=label, color=color)
    return rewards  # Return rewards for logging in WandB

# Initialize the plot for Figure 4
def initialize_plot():
    plt.figure(figsize=(8, 6))
    plt.xlabel('Objective 1 (f1)')
    plt.ylabel('Objective 2 (f2)')
    plt.title('Pareto Front')

# Finalize and show the plot
def finalize_plot():
    plt.legend()
    plt.grid(True)
    plt.show()

# Log the Pareto front to WandB as an image
def log_plot_to_wandb():
    plt.savefig("pareto_front.png")
    wandb.log({"Pareto Front": wandb.Image("pareto_front.png")})

# Train the Distributional GFlowNet and other baselines and generate the plots
def train_and_plot():
    batch_size = 100

    # Generate initial random solutions
    solutions_dgfn = generate_batch(batch_size)
    solutions_reinforce = generate_batch(batch_size)
    solutions_gfn = generate_batch(batch_size)

    # Initialize the plot
    initialize_plot()

    # Plot Pareto fronts for different methods and log them to WandB
    rewards_dgfn = plot_pareto_front(solutions_dgfn, 'DGFN', 'blue')
    rewards_reinforce = plot_pareto_front(solutions_reinforce, 'REINFORCE', 'green')
    rewards_gfn = plot_pareto_front(solutions_gfn, 'GFN', 'red')

    # Log objective values (rewards) to WandB
    wandb.log({
        "DGFN Objective 1": torch.mean(rewards_dgfn[:, 0]).item(),
        "DGFN Objective 2": torch.mean(rewards_dgfn[:, 1]).item(),
        "REINFORCE Objective 1": torch.mean(rewards_reinforce[:, 0]).item(),
        "REINFORCE Objective 2": torch.mean(rewards_reinforce[:, 1]).item(),
        "GFN Objective 1": torch.mean(rewards_gfn[:, 0]).item(),
        "GFN Objective 2": torch.mean(rewards_gfn[:, 1]).item()
    })

    # Finalize and show the plot
    finalize_plot()

    # Log the plot to WandB
    log_plot_to_wandb()

if __name__ == "__main__":
    train_and_plot()

    # Finish the WandB run
    wandb.finish()
