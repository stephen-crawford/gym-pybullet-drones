import torch
import torch.nn as nn


# Define a simple neural network as you specified
class SimpleNN(nn.Module):
    def __init__(self, n0, n1, n2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n0, n1)
        self.fc2 = nn.Linear(n1, n2)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)  # ReLU activation
        output = self.fc2(a1)
        return output

def calculate_downwash_force(drone1_x, drone1_y, drone1_z, drone1_roll, drone1_pitch, drone1_yaw, drone1_thrust,
                drone2_x, drone2_y, drone2_z, drone2_roll, drone2_pitch, drone2_yaw, drone2_thrust):
    # Initialize the model architecture again

    # Define the network parameters
    features = ['drone1_x', 'drone1_y', 'drone1_z', 'drone1_roll', 'drone1_pitch', 'drone1_yaw', 'drone1_thrust',
                'drone2_x', 'drone2_y', 'drone2_z', 'drone2_roll', 'drone2_pitch', 'drone2_yaw', 'drone2_thrust']

    n0 = 14  # Input size: number of features
    n1 = 3  # Hidden layer size (you can change this)
    n2 = 1  # Output size (downwash force)

    model = SimpleNN(n0, n1, n2)

    # Load the state_dict
    model.load_state_dict(torch.load('/home/stephencrawford/gym-pybullet-drones/gym_pybullet_drones/envs/phsyics/force_trained_shallow_model.pth'))

    # Put the model in evaluation mode (important for inference)
    model.eval()
    print("Model set to eval mode")
    # Assume new_input is a NumPy array or a list representing the new input
    # Example: new_input = [1.0, 2.0, 3.0, ..., n_features]
    new_input = [drone1_x, drone1_y, drone1_z, drone1_roll, drone1_pitch, drone1_yaw, drone1_thrust, drone2_x, drone2_y, drone2_z, drone2_roll, drone2_pitch, drone2_yaw, drone2_thrust]  # Example new input with the correct number of features

    # Convert the new input to a PyTorch tensor
    new_input_tensor = torch.tensor(new_input, dtype=torch.float32).view(1,
                                                                         -1)  # .view(1, -1) reshapes it to be [1, n_features]

    # Ensure the input is on the correct device (e.g., CPU or GPU)
    new_input_tensor = new_input_tensor.to(torch.device('cpu'))  # Change 'cpu' to 'cuda' if using GPU

    # Perform a forward pass to get predictions
    with torch.no_grad():  # Disable gradient calculation during inference
        prediction = model(new_input_tensor)  # Forward pass: model output

    # The output 'prediction' is a tensor; you can convert it to a NumPy array or print it
    print("Prediction:", prediction.numpy())  # Convert tensor to NumPy array for easier reading

    # If you have a multidimensional output, you might want to extract just the predicted value

    print("Predicted value:", prediction)

    return prediction
