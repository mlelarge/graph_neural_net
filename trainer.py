import time
import torch

def train_triplet(train_loader,model,optimizer,
                helper,device,epoch,eval_score=False,print_freq=100):
    model.train()
    helper.reset_meters('train')
    helper.reset_meters('hyperparams')
    learning_rate = optimizer.param_groups[0]['lr']
    helper.update_value_meter('hyperparams', 'learning_rate', learning_rate)
    end = time.time()
    batch_size = train_loader.batch_size

    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        helper.update_meter('train', 'data_time', time.time() - end, n=batch_size)

        data = data.to(device)
        target_deviced = target.to(device)
        output = model(data)#,input2)
        rawscores = output.squeeze(-1)
        raw_scores = torch.softmax(rawscores,-1)

        loss = helper.criterion(raw_scores,target_deviced)
        helper.update_meter('train', 'loss', loss.data.item(), n=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        helper.update_meter('train', 'batch_time', time.time() - end, n=batch_size)
        end = time.time()

    
        if i % print_freq == 0:
            if eval_score:
                #print(np_out.shape)
                values = helper.eval_function(raw_scores,target_deviced)
                #print(acc_max, n, bs)
                helper.update_eval('train', values)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '{helper_str}'.format(
                   epoch, i, len(train_loader), batch_time=helper.get_meter('train', 'batch_time'),
                   data_time=helper.get_meter('train', 'data_time'), lr=learning_rate,
                   loss=helper.get_meter('train', 'loss'), helper_str=helper.get_eval_str('train')))
    
        helper.log_meters('train', n=epoch)
    helper.log_meters('hyperparams', n=epoch)


def val_triplet(val_loader,model,helper,device,epoch,eval_score=False,print_freq=10,val_test='val'):
    model.eval()
    helper.reset_meters(val_test)
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):

            data = data.to(device)
            target_deviced = target.to(device)
            output = model(data)
            rawscores = output.squeeze(-1)
            raw_scores = torch.softmax(rawscores,-1)
            
            loss = helper.criterion(raw_scores,target_deviced)
            helper.update_meter(val_test, 'loss', loss.data.item(), n=1)
    
            if eval_score:
                values = helper.eval_function(raw_scores,target_deviced)
                helper.update_eval(val_test,values)
            if i % print_freq == 0:
                los = helper.get_meter(val_test, 'loss')
                if val_test == 'val':
                    print('Validation set, epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{helper_str}'.format(
                        epoch, i, len(val_loader), loss=helper.get_meter(val_test, 'loss'),
                        helper_str = helper.get_eval_str(val_test)))
                else:
                    print('Test set, epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{helper_str}'.format(
                        epoch, i, len(val_loader), loss=helper.get_meter(val_test, 'loss'),
                        helper_str = helper.get_eval_str(val_test)))

        helper.log_meters(val_test, n=epoch)
    relevant_metric = helper.get_relevant_metric(val_test)
    return relevant_metric.avg, los.avg



def train_simple(train_loader,model,optimizer,
                helper,device,epoch,eval_score=False,print_freq=100):
    model.train()
    model.to(device)
    learning_rate = optimizer.param_groups[0]['lr']
    end = time.time()
    batch_size = train_loader.batch_size

    for i, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target_deviced = target.to(device)
        output = model(data)#,input2)
        raw_scores = output.squeeze(-1)

        loss = helper.criterion(raw_scores,target_deviced)
        helper.update_meter('train', 'loss', loss.data.item(), n=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
        if i % print_freq == 0:
            if eval_score:
                #print(np_out.shape)
                values = helper.eval_function(raw_scores,target_deviced)
                #print(acc_max, n, bs)
                helper.update_eval('train', values)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {lr:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '{helper_str}'.format(
                   epoch, i, len(train_loader),
                   lr=learning_rate,
                   loss=helper.get_meter('train', 'loss'), helper_str=helper.get_eval_str('train')))
    optimizer.zero_grad()   
    helper.log_meters('train', n=epoch)
    helper.log_meters('hyperparams', n=epoch)

def cycle_simple(n_epochs, train_loader, val_loader, model, optimizer, scheduler, helper, device):
    best_state_dict = model.state_dict()
    prev_loss = -1
    for current_epoch in range(n_epochs):
        train_simple(train_loader,model,optimizer,helper,device,epoch=current_epoch,eval_score=True,print_freq=100)
        
        _, loss = val_triplet(val_loader,model,helper,device,epoch=current_epoch,eval_score=True)
        if prev_loss == -1 or prev_loss > loss:
            prev_loss = loss
            best_state_dict = model.state_dict()
        scheduler.step(loss)

        cur_lr = optimizer.param_groups[0]['lr']
        if helper.stop_condition(cur_lr):
            print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
            break
    model.load_state_dict(best_state_dict)