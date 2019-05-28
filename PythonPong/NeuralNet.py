import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def mapFromTo(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y

class NeuralNet(object):
    def __init__(self, topology=[]):
        if len(topology) < 2:
            print("Error: topology should be a list with at least 2 values in the format:")
            print("[InputNumber, hidden_layer_1_neurons, hidden_layer_2_neurons, outputNumber...]")
            print('Example: [2,3,1]')
            exit(-1)
        
        self.W = []
        self.topology = list(topology)
        # Generate random arrays for each layer interface with the correct amount of values
        # W1 W2 W3....
        for i in range(len(topology)-1):
            self.W.append(
                np.random.randn(
                    topology[i], topology[i+1]
                ))
        self.image = False
    
    def forward_propagation(self, x):
        stage = x
        for w in self.W:
            weighted = np.dot(stage,w)
            stage = self.sigmoid(weighted)
        return stage
    
    def sigmoid(self, x, d = False):
        if d:
            return NeuralNet.sigmoid(x)*(1-NeuralNet.sigmoid(x))
        return 1/(1+np.exp(-x))
    
    def draw_circle(self, draw, x, y, r, color):
        upperLeftX = x-r
        upperLeftY = y-r
        lowerRightX = x+r
        lowerRightY = y+r
        draw.ellipse((upperLeftX, upperLeftY, lowerRightX, lowerRightY), 
                        fill = color)
        
    def draw_diagram(self, style='dark'):
        # Debug
        grid=False
        
        # get max and min values for W
        maxW = max([wn.max() for wn in self.W])
        minW = min([wn.min() for wn in self.W])
        abMW = max([abs(minW), maxW])
        # Space for each neuron
        blockSizeX = 1500
        blockSizeY = 1000
        rightPadding = blockSizeX - blockSizeY
        
        # Neuron radius
        nSize = 200
        borderSize = 20 # dont change radius
        lineClearance = 80
        horClearance = (blockSizeX-nSize*2)/2
        
        # Set image size based on neural network
        width = len(self.topology)*blockSizeX 
        height = max([n for n in self.topology])*blockSizeY
        layers = len(self.topology)
        maxLineWidth = 60
        
        # select color scheme
        if style == 'dark':
            cInput      = (  0,  16,  61, 255)
            cHidden     = (  0,  83,  86, 255)
            cOutput     = (  0,  16,  61, 255)
            cBorder     = (255, 255, 255, 255)
            cBackground = (0, 0, 0, 0)
            cText       = (200, 200, 200, 255)
        
        # Create image
        img = Image.new('RGBA', (width, height), cBackground)
        draw = ImageDraw.Draw(img)
        
        # Writings
        fontsize = 220
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', fontsize)
        wText = str('W: (%.2f, %.2f)' % (minW, maxW))
        draw.text((10, 10), wText, fill=cText, font=font,)
     ##################################### Code #####################################
        # Draw hidden layers
        for i in range(len(self.topology)):
            xStep = width/layers
            if i == 0:
                infill = cInput
            elif i == len(self.topology)-1:
                infill = cOutput
            else:
                infill = cHidden
            if(grid): draw.line((xStep * i,0,xStep * i,height), width=1)
            for j in range(self.topology[i]):
                verClearance = (height/self.topology[i] - 2*nSize)/2
                yStep = height/(self.topology[i])
                
                # Draw lines for weights
                if i < len(self.topology) - 1:
                    start = [xStep * i + (nSize + horClearance),
                                yStep * j + nSize + verClearance]
                    nextYStep = height/(self.topology[i+1])
                    nextVerClearance = (height/self.topology[i+1] - 2*nSize)/2
                    
                    for k in range(self.W[i][j].shape[0]):
                        end = [blockSizeX*(i+1) + nSize + horClearance, nextYStep * k + nSize + nextVerClearance]
                        # Calculate color based on weight
                        w = self.W[i][j][k]
                        if w < 0:
                            opacity   = int(mapFromTo(abs(w), 0, abMW, 0, 255))
                            lineWidth = int(mapFromTo(abs(w), 0, abMW, 0, maxLineWidth))
                            lineColor = (221, 93, 93, opacity)
                        else:
                            opacity   = int(mapFromTo(w, 0, abMW, 0, 255))
                            lineWidth = int(mapFromTo(w, 0, abMW, 0, maxLineWidth))
                            lineColor = (0, 216, 122, opacity)
                        draw.line(start+end, fill=lineColor, width=lineWidth)
                # Draw Circles
                self.draw_circle(draw, 
                                 xStep * i + nSize + horClearance, 
                                 yStep * j + nSize + verClearance, 
                                 nSize + lineClearance, cBackground)
                
                self.draw_circle(draw, 
                                 xStep * i + nSize + horClearance, 
                                 yStep * j + nSize + verClearance, 
                                 nSize, cBorder)
                self.draw_circle(draw, 
                                 xStep * i + nSize + horClearance, 
                                 yStep * j + nSize + verClearance, 
                                 nSize - borderSize, infill)
                if(grid): draw.line((xStep * i, j*height/self.topology[i],xStep * i + width/3 ,  j*height/self.topology[i]), width=1)
                
                        
        del draw
        self.image=img
        return img
    
    def save_image(self, path="NeuralNet.png"):
        if not self.image:
            self.image = self.draw_diagram()
        self.image.save('NeuralNet.png')

    def get_weights(self):
        # Unfold weights to a single vector and return them
        wVec = []
        for w in self.W:
            for node in w:
                for weight in node:
                    wVec.append(weight)
        return wVec
    
    def set_weights(self, wVec):
        for i in range(len(self.W)):
            for j in range(self.W[i].shape[0]):
                for k in range(self.W[i][j].shape[0]):
                    self.W[i][j][k] = wVec.pop(0)

if __name__ == "__main__":
    # Based on:
    # https://enlight.nyc/projects/neural-network/
    np.random.seed(5541)
    # X = (hours studying, hours sleeping), y = score on test
    xAll = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
    y = np.array(([92], [86], [89]), dtype=float) # output

    # scale units
    xAll = xAll/np.amax(xAll, axis=0) # scaling input data
    y = y/100 # scaling output data (max test score is 100)

    # split data
    X = np.split(xAll, [3])[0] # training data
    xPredicted = np.split(xAll, [3])[1] # testing data
    
    # print(X)
    # print(y)
    
    NN = NeuralNet([2, 3, 1])
    NN.save_image()
    # print(o)