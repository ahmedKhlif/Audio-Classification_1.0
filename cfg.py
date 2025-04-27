import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', f'{mode}.model')
        self.pickle_path = os.path.join('pickles', f'{mode}.p')
        self.min = None
        self.max = None
        self.classes = ['Swords', 'WildAnimals','Alarms']  # Update classes