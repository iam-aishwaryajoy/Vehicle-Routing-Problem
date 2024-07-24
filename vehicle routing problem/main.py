from source.libraries import *
from source.modelling import DeliverySystem



if __name__ == "__main__":
    train = 'data/input/train.csv'
    test = 'data/input/test.csv'
    delivery_sys = DeliverySystem(train, test)
    delivery_sys.train_the_model()
