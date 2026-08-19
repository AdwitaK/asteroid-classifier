import torch
from torch import nn
from dataset import AsteroidDataset
from preprocess import preprocess, convert_to_tensors
from torch.utils.data import DataLoader
from model import AsteroidClassifier

# Get preprocessed data
data = preprocess("../data/nasa.csv")
x_train, y_train, x_val, y_val, x_test, y_test = convert_to_tensors(data)

# Put data in asteroid dataset class
train_data = AsteroidDataset(x_train, y_train)
val_data = AsteroidDataset(x_val, y_val)
test_data = AsteroidDataset(x_test, y_test)

# Make batches
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

model = AsteroidClassifier(x_train.shape[1])

#Account for imbalance in datases
hazardous = y_train.sum()
pos_weight = torch.tensor([(len(y_train) - hazardous )/ hazardous], dtype = torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):

    ## Train
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        output = model(x_batch)
        output = output.squeeze()

        loss = criterion(output, y_batch)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss/len(train_loader)

    ## Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
        
            output = model(x_batch)
            output = output.squeeze()
    
            loss = criterion(output, y_batch)

            val_loss += loss.item()
        val_loss /= len(val_loader)


    print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Training Loss: {average_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}"
        )