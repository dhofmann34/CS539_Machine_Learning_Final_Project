import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        # encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        # decoder        
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x_lat = self.pool(x)

        # decoder
        x = F.relu(self.t_conv1(x_lat))
        x = torch.sigmoid(self.t_conv2(x))
              
        return x, x_lat

    # only runs decoder to return original image
    def run_decoder(self, latent):
        x = F.relu(self.t_conv1(latent))
        x = torch.sigmoid(self.t_conv2(x))
        return x


def train(args, train_loader):
    model_path = "./model.pt"  # where to save trained model

    # initialize AutoEncoder
    image_shape = 3
    model = ConvAutoencoder()
    print(model)


    # main training loop
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    # move the model onto GPU
    model.cuda()
    model.train()

    # mean-squared error loss, we may switch to cross entropy
    mse = nn.MSELoss()

    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = args.epochs

    for epoch in tqdm(range(n_epochs)):
        loss = 0
        lat = 0

        # batch
        for idx, img in enumerate(train_loader):
            # put data onto GPU
            img = img.cuda()

            optimizer.zero_grad()
            
            # feed image
            outputs, lat = model(img)
            
            # loss
            train_loss = mse(outputs, img)
            
            # compute gradients
            train_loss.backward()
            
            # parameter update
            optimizer.step()
            
            # add loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        if epoch %5 ==0:
            print(f"epoch : {epoch}, loss = {loss}")
            #torch.save(model.state_dict(), model_path)
        
    # save model so we do not have to rerun each time
    torch.save(model.state_dict(), model_path)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, n_epochs, loss))