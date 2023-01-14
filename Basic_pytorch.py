import torch
print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([[1,2,3], [4,5,6]],
                         dtype = torch.float32,
                         device = device,
                         requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)


# Other initializations
x = torch.empty(size = (3,3))

x = torch.zeros((3,3))
print(x)

x = torch.rand((3,3))
print(x)

x = torch.eye(3,3)
print(x)

# Operations

x = torch.tensor([1,2,3])
y = torch.tensor([2,3,4])
z = torch.empty(3)
torch.add(x,y,out=z)
print(z)

z = x+y
print(z)


# Indexing

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)

print(x[:,0].shape)

print(x[2,:10])

x[0,0] = 100

x = torch.arange(10)
i = [2,5,7]
print(x[i])

x = torch.rand((3,5))
r = torch.tensor([1,0])
c = torch.tensor([4,0])
print(x[r,c])
