import torch
import torch.nn.functional as F
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
	num_workers=0,
	drop_last=True
)

test_loader = DataLoader(
	dataset=test_ds,
	batch_size=1,
	shuffle=False,
	num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNetwork(2, 2)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
num_epochs =10
for epoch in range(num_epochs):
	model.train()

	for batch_idx, (features, labels) in enumerate(train_loader):
		features, labels = features.to(device), labels.to(device)
		logits = model(features)
		loss = F.cross_entropy(logits, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		### LOGGING
		print(	f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
				f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
				f" | Train Loss: {loss:.2f}")

	model.eval()
	# Insert optional model evaluation code

# Save model
torch.save(model.state_dict(), "model.pth")

# Classify
with torch.no_grad():
	for batch_idx, (features, labels) in enumerate(test_loader):
		res = torch.softmax(model(features), dim=1)
		print(f"Result: {res} Label: {labels}")