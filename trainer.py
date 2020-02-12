from toolbox import metrics

def train_triplet(train_loader,model,criterion,optimizer,epoch,print_freq=10):
    model.train()
    for i, (input1, input2) in enumerate(train_loader):
        output = model(input1,input2)

        loss = criterion(output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], top1=meters['acc1']))



    
