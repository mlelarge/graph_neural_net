from toolbox import metrics
import time
import torch

def train_triplet(train_loader,model,criterion,optimizer,
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
        raw_scores = output.squeeze(-1)

        loss = criterion(raw_scores,target_deviced)
        #loss = criterion(proba,K[:,:,:,1])
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
    optimizer.zero_grad()   
    helper.log_meters('train', n=epoch)
    helper.log_meters('hyperparams', n=epoch)


def val_triplet(val_loader,model,criterion,
                helper,device,epoch,eval_score=False,print_freq=10,val_test='val'):
    model.eval()
    helper.reset_meters(val_test)
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):

            data = data.to(device)
            target_deviced = target.to(device)
            output = model(data)
            raw_scores = output.squeeze(-1)
        
            loss = criterion(raw_scores,target_deviced)
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


def train_tsp(train_loader,model,criterion,optimizer,
                logger,device,epoch,eval_score=None,print_freq=100):
    model.train()
    logger.reset_meters('train')
    logger.reset_meters('hyperparams')
    learning_rate = optimizer.param_groups[0]['lr']
    logger.update_value_meter('hyperparams', 'learning_rate', learning_rate)
    end = time.time()

    for i, (input, target, mask) in enumerate(train_loader):
        batch_size = input.shape[0]
        # measure data loading time
        logger.update_meter('train', 'data_time', time.time() - end, n=batch_size)

        input = input.to(device)
        mask = mask.to(device)
        target = target.to(device)
        target = target.type(torch.float32)
        
        raw_scores = model(input).squeeze(-1)
        #raw_scores = torch.matmul(output,torch.transpose(output, 1, 2))
        loss = criterion(raw_scores,mask,target)
        logger.update_meter('train', 'loss', loss.data.item(), n=1)
        #optimizer.zero_grad()
        loss.backward()
        #optimizer.step()
        # measure elapsed time
        logger.update_meter('train', 'batch_time', time.time() - end, n=batch_size)
        end = time.time()   
        if i % print_freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            if eval_score is not None:
                #print(np_out.shape)
                prec, rec, f1 = eval_score(raw_scores*mask,target,device)
                #print(acc_max, n, bs)
                logger.update_meter('train', 'f1', f1)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.2e}\t'
                  'Loss {loss.avg:.4f} ({loss.val:.4f})\t'
                  'F1 {f1.avg:.3f} ({f1.val:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=logger.get_meter('train', 'batch_time'),
                   data_time=logger.get_meter('train', 'data_time'), lr=learning_rate,
                   loss=logger.get_meter('train', 'loss'), f1=logger.get_meter('train', 'f1')))

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


def val_tsp(val_loader,model,criterion,
                logger,device,epoch,eval_score=None,print_freq=10,val_test='val'):
    model.eval()
    logger.reset_meters(val_test)

    for i, (input, target, mask) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        target = target.type(torch.float32)
        mask = mask.to(device)

        with torch.no_grad():
            raw_scores = model(input).squeeze(-1)
        #raw_scores = torch.matmul(output,torch.transpose(output, 1, 2))
        loss = criterion(raw_scores,mask,target)
        logger.update_meter('val', 'loss', loss.data.item(), n=1)
    
        if eval_score is not None:
            prec, rec, f1 = eval_score(raw_scores*mask,target,device)
            logger.update_meter(val_test, 'f1', f1)
        if i % print_freq == 0:
            current_f1 = logger.get_meter(val_test, 'f1')
            los = logger.get_meter(val_test, 'loss')
            if val_test == 'val':
                print('Validation set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.avg:.4f} ({loss.val:.4f})\t'
                    'F1 {f1.avg:.3f} ({f1.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=logger.get_meter(val_test, 'loss'),
                    f1=current_f1))
            else:
                print('Test set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.avg:.4f} ({loss.val:.4f})\t'
                    'F1 {acc.avg:.3f} ({acc.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=logger.get_meter(val_test, 'loss'),
                    f1=current_f1))

    logger.log_meters(val_test, n=epoch)
    return current_f1.avg, los.avg
