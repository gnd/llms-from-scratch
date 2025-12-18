import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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


class ToyDataset(Dataset):
	def __init__(self, X, y):
		self.features = X
		self.labels = y

	def __getitem__(self, index):
		x = self.features[index]
		y = self.labels[index]
		return x, y

	def __len__(self):
		return self.labels.shape[0]


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

X_train = torch.tensor([
[-1.2, 3.1],
[-0.9, 2.9],
[-0.5, 2.6],
[2.3, -1.1],
[2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
[-0.8, 2.8],
[2.6, -1.6],
])

y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)	

train_loader = DataLoader(
	dataset=train_ds,
	batch_size=2,
	shuffle=True,
	num_workers=0
)

test_loader = DataLoader(
	dataset=test_ds,
	batch_size=2,
	shuffle=False,
	num_workers=0
)

for idx, (inputs, targets) in enumerate(train_loader):
	print(f"Batch {idx+1}:", inputs, targets)

print(train_loader)