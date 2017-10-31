import os

class NetworkParameters:
    def __init__(self, modelDirectory):
        self.modelDirectory = modelDirectory


        if os.path.exists(self.modelDirectory) is False:
            os.mkdir(self.modelDirectory)

        self.checkpointedModelDir = os.path.join(self.modelDirectory, 'savedModels')

        if os.path.exists(self.checkpointedModelDir) is False:
            os.mkdir(self.checkpointedModelDir)

        self.modelSaveName = os.path.join(self.checkpointedModelDir, 'model_{epoch:02d}.hdf5')
        self.bestModelSaveName = os.path.join(self.checkpointedModelDir, 'best_model.hdf5')
