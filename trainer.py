from toolbox import metrics
import time
import torch

def train_triplet(train_loader,model,criterion,optimizer,
                logger,device,epoch,clique_size,eval_score=None,print_freq=100):
    model.train()
    logger.reset_meters('train')
    logger.reset_meters('hyperparams')
    learning_rate = optimizer.param_groups[0]['lr']
    logger.update_value_meter('hyperparams', 'learning_rate', learning_rate)
    end = time.time()

    for i, (input1, input2) in enumerate(train_loader):
        batch_size = input1.shape[0]
        # measure data loading time
        logger.update_meter('train', 'data_time', time.time() - end, n=batch_size)

        input1 = input1.to(device)
        K = input2.to(device)
        output = model(input1)#,input2)

        rawscores = output.squeeze(-1)
        proba = torch.softmax(rawscores,-1)
        loss = torch.mean(criterion(proba,K[:,:,:,1])*input1[:,:,:,1])
        #loss = criterion(proba,K[:,:,:,1])
        logger.update_meter('train', 'loss', loss.data.item(), n=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        logger.update_meter('train', 'batch_time', time.time() - end, n=batch_size)
        end = time.time()

    
        if i % print_freq == 0:
            if eval_score is not None:
                #print(np_out.shape)
                acc, total_n_vertices = eval_score(proba*input1[:,:,:,1],clique_size)
                #print(acc_max, n, bs)
                logger.update_meter('train', 'acc', acc, n=total_n_vertices)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=logger.get_meter('train', 'batch_time'),
                   data_time=logger.get_meter('train', 'data_time'), lr=learning_rate,
                   loss=logger.get_meter('train', 'loss'), acc=logger.get_meter('train', 'acc')))

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


def val_triplet(val_loader,model,criterion,
                logger,device,epoch,clique_size,eval_score=None,print_freq=10,val_test='val'):
    model.eval()
    logger.reset_meters(val_test)

    for i, (input1, input2) in enumerate(val_loader):
        input1 = input1.to(device)
        K = input2.to(device)
        output = model(input1)
        rawscores = output.squeeze(-1)
        proba = torch.softmax(rawscores,-1)
        
        loss = torch.mean(criterion(proba,K[:,:,:,1])*input1[:,:,:,1])
        logger.update_meter(val_test, 'loss', loss.data.item(), n=1)
    
        if eval_score is not None:
            acc, total_n_vertices = eval_score(proba*input1[:,:,:,1],clique_size)
            logger.update_meter(val_test, 'acc', acc, n=total_n_vertices)
        if i % print_freq == 0:
            accu = logger.get_meter(val_test, 'acc')
            los = logger.get_meter(val_test, 'loss')
            if val_test == 'val':
                print('Validation set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=logger.get_meter(val_test, 'loss'),
                    acc=accu))
            else:
                print('Test set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=logger.get_meter(val_test, 'loss'),
                    acc=accu))

    logger.log_meters(val_test, n=epoch)
    return accu.avg, los.avg


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