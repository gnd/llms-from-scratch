import torch

class NeuralNetwork(torch.nn.Module):

	def __init__(self, num_inputs, num_outputs):
		super().__init__()

		self.layers = torch.nn.Sequential(
			# 1st hidden layer
			torch.nn.Linear(num_inputs, 30),
			torch.nn.ReLU(),
			
			# 2nd hidden layer
			torch.nn.Linear(30, 20),
			torch.nn.ReLU(),

			# output layer
			torch.nn.Linear(20, num_outputs),
		)

	# required function for forward pass
	def forward(self, x):
		# we can also call layers manually instead of using self.layers(x) by using sequential
		logits = self.layers(x)
		return logits


#torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].bias)
print(model.layers[0].weight.shape)

# simple forward pass
X = torch.rand((1, 50))
# using softmax for class categorizations using probabilities
out = torch.softmax(model(X), dim=1)
print(out)

# can also use with torch.no_grad():
#	out = model(X), when model already trained