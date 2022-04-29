from train import train 
from models import NeuralNetwork
import pandas as pd
import argparse

metadata = args.metadata

data= pd.read_csv(metadata)

x=data.iloc[:,1:].values
ycrude=data.iloc[:,0].values

#One hot encoding
y = np.zeros((ycrude.size, ycrude.max()+1))
y[np.arange(ycrude.size),ycrude] = 1
print(x.shape)
print(y.shape)

optim=args.optim
batch_size=args.batch_size
epoch=args.epochs
lr=args.lr
gamma1=args.gamma1
gamma2=args.gamma2
epsillion= args.epsillon
alpha=args.alpha




# Model


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optim', help='This is name of the optimizer', required=True, type=str)
parser.add_argument('-n', '--num_epochs', help='This is the number of epochs', required=False, type=str)
parser.add_argument('-l', '--lr', help='This is the learning rate', required=False, type=str)
parser.add_argument('-g', '--gamma1', help='This is the momentum term', required=False, type=str)
parser.add_argument('-h', '--gamma2', help='This is the gammma 2', required=False, type=str)
parser.add_argument('-e', '--epsillon', help='This is the epsillon', required=False, type=str)
parser.add_argument('-a', '--alpha', help='This is the weight decay', required=False, type=str)

main_args = vars(parser.parse_args())
num_epochs = int(main_args['num_epochs'])
model_name = main_args['model_name'].lower()








neuralnet= NeuralNetwork(args,optim='')
epoch_loss, _=train(neuralnet)