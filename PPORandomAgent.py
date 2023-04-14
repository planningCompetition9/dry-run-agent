import tensorflow as tf
import numpy as np
import random
import gym

class PPORandomAgent():

    def __init__(self,stateSize,actionSpace):
        self.stateSize = stateSize
        self.actionSpace = actionSpace
        self.actionSize = len(self.actionSpace)
        self.isDiscreteAction = self.splitActionTypes()
        self.hiddenLayers = [64,64]
        self.actorNetwork = self.createActorNetwork()

    def splitActionTypes(self):
        discreteActions = []
        for key in self.actionSpace:
            if isinstance(self.actionSpace[key], gym.spaces.Box):
                discreteActions.append(False)
            elif isinstance(self.actionSpace[key],gym.spaces.Discrete):
                discreteActions.append(True)
        return discreteActions
    
    def createActorNetwork(self):
        outputLayers = []
        initializer = "orthogonal"
        inputLayer = tf.keras.Input(shape=(self.stateSize))
        lastLayer = inputLayer
        for size in self.hiddenLayers:
            nextLayer = tf.keras.layers.Dense(units=size, activation="relu",
                                                kernel_initializer=initializer, bias_initializer="glorot_uniform")(lastLayer)
            lastLayer = nextLayer
        for index, key in enumerate(self.actionSpace):
            if self.isDiscreteAction[index]:
                outputLayer = tf.keras.layers.Dense(units=self.actionSpace[key].n, activation="softmax",
                                                kernel_initializer=initializer, bias_initializer="glorot_uniform")(lastLayer)
            else:
                outputLayer = tf.keras.layers.Dense(units=1, activation="sigmoid",
                                                kernel_initializer=initializer, bias_initializer="glorot_uniform")(lastLayer)
            outputLayers.append(outputLayer)
        
        model = tf.keras.Model(inputs=inputLayer, outputs=outputLayers)
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4, epsilon=1e-5)
        model.compile(optimizer=opt,jit_compile=True)
        return model
    
    def act(self, state):
        state = np.array(list(state.values()),dtype=np.float32)
        tensorState =tf.reshape(tf.convert_to_tensor(state),shape=(1,self.stateSize))
        actionProbs = self.actorNetwork(tensorState)
        actions = {}
        for index, key in enumerate(self.actionSpace):
            prob = actionProbs[index].numpy()
            if self.isDiscreteAction[index]:
                action = random.choices(np.arange(self.actionSpace[key].n),weights = prob, k = 1)
            else:
                lowerBound = self.actionSpace[key].low
                upperBound = self.actionSpace[key].high
                #For now, box lower and upper bound
                if lowerBound == -np.inf and upperBound == np.inf:
                    lowerBound = -1
                    upperBound = 1
                elif lowerBound == -np.inf:
                    lowerBound = upperBound - 2
                elif upperBound ==np.inf:
                    upperBound = lowerBound +2
                action = ((upperBound - lowerBound) * prob + lowerBound)[0][0]
            actions[key] = action
        return actions


