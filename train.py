import time
import Optim
import sys
import os
# sys.path.append("D:/Research/Triplet Loss/TripletLoss/TEGNN/Baseline_models/")
base_path = os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))
sys.path.append(base_path)

import TripletNet
from Baseline_models.ml_eval import *
from Baseline_models import CNN, X_CNN, delta_X_CNN, X_LSTNet, delta_X_LSTNet, X_RNN, delta_X_RNN, TENet
import datetime
# import X_LSTNet

np.seterr(divide='ignore',invalid='ignore')

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    global train_count, epoch_start_time
    starttime = datetime.datetime.now()
    with torch.no_grad():
        for X, Y in data.get_batches(X, Y, batch_size, True):
            if X.shape[0] != args.batch_size:
                break
            output = model(X)

            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))

            scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)
            del scale, X, Y
            torch.cuda.empty_cache()

            train_count=train_count+1

            if train_count%50==0:
            	endtime = datetime.datetime.now()	
            	# print("count",train_count,(endtime - starttime))#.seconds)

    rmse = math.sqrt(total_loss / n_samples)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    mae = total_loss_l1 / n_samples

    return rmse, rse, mae, rae, correlation

def triplet_evaluate(data, X, Y, delta_X, delta_Y, Y_1, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    global train_count, epoch_start_time
    starttime = datetime.datetime.now()
    with torch.no_grad():
        for X, Y, delta_X, delta_Y, Y_1 in data.triplet_get_batches(X, Y, delta_X, delta_Y, Y_1, batch_size, False):
        # for X, Y, delta_X, delta_Y, Y_1 in data.triplet_get_batches(X, Y, delta_X, delta_Y, Y_1, batch_size, True):
            if X.shape[0] != args.batch_size:
                break
            output, delta_output, output_1 = model(X, delta_X)

            if predict is None:
                predict = output_1
                test = Y_1
            else:
                predict = torch.cat((predict, output_1))
                test = torch.cat((test, Y_1))

            scale = data.scale.expand(output_1.size(0), data.m)
            total_loss += evaluateL2(output_1 * scale, Y_1 * scale).item()
            total_loss_l1 += evaluateL1(output_1 * scale, Y_1 * scale).item()
            n_samples += (output_1.size(0) * data.m)
            del scale, X, Y, delta_X, delta_Y, Y_1
            torch.cuda.empty_cache()

            train_count=train_count+1

            if train_count%50==0:
            	endtime = datetime.datetime.now()	
            	# print("count",train_count,(endtime - starttime))#.seconds)

    rmse = math.sqrt(total_loss / n_samples)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    mae = total_loss_l1 / n_samples

    return rmse, rse, mae, rae, correlation

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    global train_count, epoch_start_time
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad()
        output = model(X)
        # print("scale",data.scale,data.scale.shape)
        scale = data.scale.expand(output.size(0), data.m)
        # print("scale",scale,scale.shape)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.m)
        torch.cuda.empty_cache()
        train_count=train_count+1
        if train_count%500==0:
        	endtime = datetime.datetime.now()	
        	# print("count",train_count,(endtime - starttime).seconds)

    return total_loss / n_samples

train_count = 0


def triplet_train(data, X, Y, delta_X, delta_Y, Y_1, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    
    global train_count, epoch_start_time

    for X, Y, delta_X, delta_Y, Y_1 in data.triplet_get_batches(X, Y, delta_X, delta_Y, Y_1, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad()

        output, delta_output, output_1 = model(X, delta_X)
        scale = data.scale.expand(output.size(0), data.m)
        loss_1 = criterion(output * scale, Y * scale) 
        loss_2 = criterion(delta_output * scale, delta_Y * scale) 
        loss_3 = criterion(output_1 * scale, Y_1 * scale) 
        loss = loss_1 + loss_2 + loss_3
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.m)
        torch.cuda.empty_cache()

        train_count=train_count+1

        if train_count%500==0:
        	endtime = datetime.datetime.now()	
        	print("count",train_count,(endtime - starttime).seconds)

    # print("count:",train_count)
    return total_loss / n_samples

parser = argparse.ArgumentParser(description='Multivariate Time series forecasting')
parser.add_argument('--data', type=str, default="./dataset/exchange_rate.txt",help='location of the data file')
# parser.add_argument('--data', type=str, default="./dataset/energydata_complete.txt",help='location of the data file')
# parser.add_argument('--data', type=str, default="./dataset/nasdaq100_padding.csv",help='location of the data file')
# parser.add_argument('--data', type=str, default="./dataset/solar_AL.txt",help='location of the data file')
# parser.add_argument('--data', type=str, default="./dataset/electricity.txt",help='location of the data file')

parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units each layer')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')

parser.add_argument('--hidCNN', type=int, default=50, help='number of CNN hidden units (channels)')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')

parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=10)

parser.add_argument('--n_e', type=int, default=8,help='The number of graph nodes')
parser.add_argument('--model', type=str, default='X_CNN',help='')
parser.add_argument('--k_size', type=list, default=[3,5,7],help='number of CNN kernel sizes')
parser.add_argument('--window', type=int, default=32,help='window size')
parser.add_argument('--decoder', type=str, default= 'GNN',help = 'type of decoder layer')
parser.add_argument('--horizon', type=int, default= 24)
parser.add_argument('--A', type=str, default="./TE/exte.txt",help='A')
parser.add_argument('--highway_window', type=int, default=8, help='The window size of the highway component')
parser.add_argument('--channel_size', type=int, default=12,help='the channel size of the CNN layers')
parser.add_argument('--hid1', type=int, default=30,help='the hidden size of the GNN layers')
parser.add_argument('--hid2', type=int, default=10,help='the hidden size of the GNN layers')
parser.add_argument('--clip', type=float, default=10,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',help='path to save the final model')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default=None)
parser.add_argument('--Triplet_loss', type=int, default=1)
# parser.add_argument('--Triplet_loss', type=int, default=1)

# parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='batch size')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',help='batch size')
parser.add_argument('--Test', type=int, default=1)
# parser.add_argument('--Test', type=int, default=1)

args = parser.parse_args()
print("args:",args)

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize, args.Triplet_loss)
print(Data.rse)

if args.Triplet_loss:
    # model_x_cnn = eval("X_CNN").Model(args,Data)
    # model_delta_x_cnn = eval("delta_X_CNN").Model(args,Data)
    # model_x_cnn = eval("X_LSTNet").Model(args,Data)
    # model_delta_x_cnn = eval("delta_X_LSTNet").Model(args,Data)
    model_x_cnn = eval("X_RNN").Model(args,Data)
    model_delta_x_cnn = eval("delta_X_RNN").Model(args,Data)
    model = eval("TripletNet").Model(model_x_cnn, model_delta_x_cnn)
    if args.cuda:
        model.cuda()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average = False).cpu()
    else:
        criterion = nn.MSELoss(size_average = False).cpu()
    evaluateL2 = nn.MSELoss(size_average = False).cpu()
    evaluateL1 = nn.L1Loss(size_average = False).cpu()
    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        
    # best_corr = 0
    best_rae = 99999

    optim = Optim.Optim(
        model.parameters(), args.optim, args.lr, args.clip,
    )
    print("??",args.Test)
    if args.Test == False:
        try:
            print('begin training')
            starttime = datetime.datetime.now()
            for epoch in range(1, args.epochs+1):
                
                # global epoch_start_time
                epoch_start_time = time.time()
               	
                train_loss = triplet_train(Data, Data.train[0], Data.train[1], Data.train[2], Data.train[3], Data.train[4], model, criterion, optim, args.batch_size)
                
                val_rmse, val_rse, val_mae, val_rae, val_corr = triplet_evaluate(Data, Data.valid[0], Data.valid[1], Data.valid[2], Data.valid[3], Data.valid[4], model, evaluateL2, evaluateL1, args.batch_size)

                print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | valid rmse {:5.5f} |valid rse {:5.5f} | valid mae {:5.5f} | valid rae {:5.5f} |valid corr  {:5.5f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse,val_rse, val_mae,val_rae, val_corr))

                # Save the model if the validation loss is the best we've seen so far.
                val = val_corr
                if val > best_corr:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_corr = val

                # if epoch % 5 == 0:
                #     test_rmse, test_acc, test_mae, test_rae, test_corr  = triplet_evaluate(Data, Data.test[0], Data.test[1], Data.test[2], Data.test[3], Data.test[4], model, evaluateL2, evaluateL1, args.batch_size)
                #     print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))
                
                print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | valid rmse {:5.5f} |valid rse {:5.5f} | valid mae {:5.5f} | valid rae {:5.5f} |valid corr  {:5.5f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse, val_acc, val_mae, val_rae, val_corr))

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_mse, test_acc, test_mae, test_rae, test_corr  = triplet_evaluate(Data, Data.test[0], Data.test[1], Data.test[2], Data.test[3], Data.test[4], model, evaluateL2, evaluateL1, args.batch_size)
    print ("\ntest rmse {:5.4f} |test rse {:5.4f} | test mae {:5.4f} | test rae {:5.4f} |test corr {:5.4f}".format(test_mse,test_acc, test_mae,test_rae, test_corr))

else:
    model = eval(args.model).Model(args,Data)

    if args.cuda:
        model.cuda()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average = False).cpu()
    else:
        criterion = nn.MSELoss(size_average = False).cpu()
    evaluateL2 = nn.MSELoss(size_average = False).cpu()
    evaluateL1 = nn.L1Loss(size_average = False).cpu()
    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        
        
    best_val = 111110
    optim = Optim.Optim(
        model.parameters(), args.optim, args.lr, args.clip,
    )

    if args.Test == False:
        try:
            print('begin training')
            starttime = datetime.datetime.now()
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

        #         val_rmse, val_rse, val_mae, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)

        #         print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | valid rmse {:5.5f} |valid rse {:5.5f} | valid mae {:5.5f} | valid rae {:5.5f} |valid corr  {:5.5f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse,val_rse, val_mae,val_rae, val_corr))

        #         # Save the model if the validation loss is the best we've seen so far.
        #         val = val_mae
        #         if val < best_val:
                      # with open(args.save, 'wb') as f:
                      #     torch.save(model, f)
        #                 best_val = val

        #         if epoch % 5 == 0:
        #             test_rmse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
        #             print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_mse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
    print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_mse,test_acc, test_mae,test_rae, test_corr))

