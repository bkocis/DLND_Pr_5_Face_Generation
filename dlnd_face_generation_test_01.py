


#######

# to run inside the DLND workspace
#######


# !unzip processed_celeba_small.zip

data_dir = 'processed_celeba_small/'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim


# Define function hyperparameters

batch_size = 128
img_size = 32

# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100

lr, b1, b2 = 0.0002, 0.5, 0.999

# set number of epochs 
n_epochs = 10


# call training function
losses = train(D, G, n_epochs=n_epochs)

def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    # TODO: Implement function and return a dataloader
    
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform = transform)
    print('Number of images: ', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                            num_workers = 0, shuffle = True)
    
    return dataloader

def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max-min) - max
    return x


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
#    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#        init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'weight'):
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)


# define the loss
def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    loss = torch.mean((D_out - 0.9)**2)
    return loss

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    loss = torch.mean((D_out)**2)
    return loss


# Define the model
# 1- Discriminator

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        
        #self.conv1 = conv(3, self.conv_dim, kernel_size = 4, batch_norm = False) 
        self.conv1 = nn.Sequential(nn.Conv2d(3, self.conv_dim, 4,2,1, bias = False))

        #self.conv2 = conv(self.conv_dim, self.conv_dim*2, kernel_size = 4)    
        self.conv2 = nn.Sequential(nn.Conv2d(self.conv_dim, self.conv_dim*2, 4,2,1, bias = False), nn.BatchNorm2d(self.conv_dim*2))
        
        #self.conv3 = conv(self.conv_dim*2, self.conv_dim*4, kernel_size = 4)
        self.conv3 = nn.Sequential(nn.Conv2d(self.conv_dim*2, self.conv_dim*4, 4,2,1, bias = False), nn.BatchNorm2d(self.conv_dim*4))
        
        self.fc = nn.Linear(self.conv_dim*4 * 4 * 4, 1)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        #                   input image   :         (128, 3, 32, 32)
        x = F.leaky_relu(self.conv1(x))           # (128, 32, 16, 16)
        x = F.leaky_relu(self.conv2(x))           # (128, 64, 8, 8)
        x = F.leaky_relu(self.conv3(x))           # (128, 128, 4, 4)
        
        x = x.view(-1, self.conv_dim*4 * 4 * 4)   # (128, 1, 128*4*4)
        x = self.fc(x) # (128, 1, 1)
        return x

# 2 - Generator
class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        
        self.z_size = z_size
        self.conv_dim = conv_dim
    
    
        self.fc = nn.Linear(self.z_size, 4 * 4 * self.conv_dim * 4) 
        
        # complete init funtion
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.conv_dim*4, self.conv_dim*2, 4, 2, 1, bias = False), nn.BatchNorm2d(self.conv_dim*2))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(self.conv_dim*2, self.conv_dim ,4, 2, 1, bias = False), nn.BatchNorm2d(self.conv_dim))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(self.conv_dim, 3, 4, 2, 1, bias = False))
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        
        x = self.fc(x)
        
        x = x.view(-1, self.conv_dim*4, 4, 4)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        
        return x

# Build the Network
def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G


# training function
def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================
            
            
            # 1. Train the discriminator on real and fake images
            
            d_optimizer.zero_grad()
            
            d_loss_real = real_loss(D(real_images.cuda()))
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            
            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            d_loss_fake = fake_loss(D(fake_images))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            
            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            
            g_loss = real_loss(D(fake_images))
            g_loss.backward()
            g_optimizer.step()
            
            
            
            
            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    # finally return losses
    return losses


# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')



# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)


D, G = build_network(d_conv_dim, g_conv_dim, z_size)

d_optimizer = optim.Adam(D.parameters(), lr, [b1, b2])
g_optimizer = optim.Adam(G.parameters(), lr, [b1, b2])

# call training function
losses = train(D, G, n_epochs=n_epochs)





################
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))

with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

_ = view_samples(-1, samples)
