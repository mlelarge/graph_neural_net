from toolbox import metrics
import time

def train_triplet(train_loader,model,criterion,optimizer,
                logger,device,epoch,eval_score=None,print_freq=100):
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    for i, (input1, input2) in enumerate(train_loader):
        batch_size = input1.shape[0]
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input1 = input1.to(device)
        input2 = input2.to(device)
        output = model(input1,input2)

        loss = criterion(output)
        meters['loss'].update(loss.data.item(), n=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

    
        if i % print_freq == 0:
            if eval_score is not None:
                #print(np_out.shape)
                acc_max, n, bs = eval_score(np_out)
                #print(acc_max, n, bs)
                meters['acc_max'].update(acc_max,n*bs)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_Max {acc_max.avg:.3f} ({acc_max.val:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], acc_max=meters['acc_max']))

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


def val_triplet(val_loader,model,criterion,
                logger,device,epoch,eval_score=None,print_freq=10):
    model.eval()
    meters = logger.reset_meters('val')

    for i, (input1, input2) in enumerate(val_loader):
        input1 = input1.to(device)
        input2 = input2.to(device)
        output = model(input1,input2)

        loss = criterion(output)
        meters['loss'].update(loss.data.item(), n=1)
    
        if eval_score is not None:
            np_out = output.cpu().detach().numpy()
                #print(np_out.shape)
            acc, n, bs = eval_score(np_out)
                #print(acc_max, n, bs)
            meters['acc_la'].update(acc,n*bs)
        if i % print_freq == 0:
            print('Validation set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=meters['loss'], acc=meters['acc_la']))

    logger.log_meters('val', n=epoch)
    return acc
