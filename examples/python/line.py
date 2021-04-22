import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 1),
            #nn.Linear(8, 8),
            nn.Linear(1, 1)
        )

    def forward(self, x):
        return self.model(x)

def line(x, m=5, b=3):
    return (m * x) + b

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3)


print(f'Using device: {device}')


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    model_in = torch.randn([100, 1])
    model_out = model(model_in)
    true_out = line(model_in)

    loss = loss_fn(true_out, model_out)

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch} --- loss: {loss.item()}')

model.eval()

test_in = torch.Tensor([
    [-2], [-1], [0], [1], [2]
])

test_out = model(test_in)
real_out = [line(x) for x in test_in]

print('Test Results:')
for x, real, test in zip(test_in, real_out, test_out):
    print(f'\tX: {x.item()}, real: {real.item()}, test: {test.item()}')

