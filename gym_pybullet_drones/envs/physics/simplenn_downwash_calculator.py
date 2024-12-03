import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, n0, n1, n2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n0, n1)
        self.fc2 = nn.Linear(n1, n2)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        output = self.fc2(a1)
        return output

def calculate_downwash_force(drone1_x, drone1_y, drone1_z, drone1_roll, drone1_pitch, drone1_yaw, drone1_thrust,
                drone2_x, drone2_y, drone2_z, drone2_roll, drone2_pitch, drone2_yaw, drone2_thrust):

    feature_count = 14
    hidden_layer_neurons = 3
    predictor = 1

    model = SimpleNN(feature_count, hidden_layer_neurons, predictor)

    # Load the state_dict
    model.load_state_dict(torch.load('/home/stephencrawford/gym-pybullet-drones/gym_pybullet_drones/envs/physics/force_trained_shallow_model.pth'))

    model.eval()

    new_input = [drone1_x, drone1_y, drone1_z, drone1_roll, drone1_pitch, drone1_yaw, drone1_thrust, drone2_x, drone2_y, drone2_z, drone2_roll, drone2_pitch, drone2_yaw, drone2_thrust]

    new_input_tensor = torch.tensor(new_input, dtype=torch.float32).view(1, -1)  # .view(1, -1) reshapes it to be [1, n_features]

    new_input_tensor = new_input_tensor.to(torch.device('cpu'))

    # Perform a forward pass to get predictions
    with torch.no_grad():  # Disable gradient calculation during inference
        prediction = model(new_input_tensor)

    return prediction
