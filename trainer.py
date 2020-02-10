def train_triplet(train_loader,model,criterion,optimizer,print_freq=10):
    model.train()
    for (input1, input2) in train_loader:
        output = model(input1,input2)

        loss = criterion(output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    
