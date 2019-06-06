# This code is not intended as a library (it doesnt even check for most of possible errors)
# the only purpose of this class is to train how to make a neuralNet from scratch
# if you need a ready to use lib... there are tons out there

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def mapFromTo(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y

class NeuralNet(object):
    def __init__(self, topology=[], seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        if len(topology) < 2:
            print("Error: topology should be a list with at least 2 values in the format:")
            print("[InputNumber, hidden_layer_1_neurons, hidden_layer_2_neurons, outputNumber...]")
            print('Example: [2,3,1]')
            exit(-1)
        
        self.W = []
        self.topology = list(topology)
        # Generate random arrays for each layer interface with the correct amount of values
        # Also creates biases for each layer (as if there was a input = 1)
        # W1 W2 W3....
        for i in range(len(topology)-1):
            self.W.append(
                np.random.randn(
                    1+topology[i], topology[i+1]
                ))
        self.image = False
    
    def forward_propagation(self, x):
        stage = x
        for w in self.W:
            # Add bias column to data
            B = np.ones((len(stage),1))
            stage = np.hstack((stage, B))
            # Get weighted sum for each destination neuron
            weighted = np.dot(stage,w)
            # Pass each value trough activation function
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
        
    def draw_diagram(self, style='dark', inputLabels=None, outputLabels=None):
        # Debug
        grid=False
        scale = 1/5 # change everything proportionaly for better or worse quality images
        
        # get max and min values for W
        maxW = max([wn.max() for wn in self.W])
        minW = min([wn.min() for wn in self.W])
        abMW = max([abs(minW), maxW])
        # Space for each neuron
        blockSizeX = scale*1500
        blockSizeY = scale*1000
        
        # Neuron radius
        nSize = scale*200
        # Bias radius
        bSize = scale*150
        # border line of circle
        borderSize = scale*20 # dont change radius
        # clear area around circle
        lineClearance = scale*80
        horClearance = (blockSizeX-nSize*2)/2
        
        # Set image size based on neural network
        width = int(len(self.topology)*blockSizeX)
        # heigth if there was no bias
        height = int(max([n for n in self.topology])*blockSizeY)
        # space to draw bias
        heightPadding = int(max([n+1 for n in self.topology])*(blockSizeY))
        layers = len(self.topology)
        maxLineWidth = scale*60
        
        # select color scheme
        if style == 'dark':
            cInput      = (  0,  16,  61, 255)
            cHidden     = (  0,  83,  86, 255)
            cOutput     = (  0,  16,  61, 255)
            cBorder     = (255, 255, 255, 255)
            cBackground = (  0,   0,   0,   0)
            cText       = (200, 200, 200, 255)
            cBias       = (  0,   0,   0, 255)
        
        # Create image
        if outputLabels is not None:    rightPadding = int(scale*400)
        else:                           rightPadding = 0
        img = Image.new('RGBA', (width+rightPadding, heightPadding), cBackground)
        draw = ImageDraw.Draw(img)
        
        # Writings
        fontsize = int(scale*200)
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', fontsize)
        wText = str('W: (%.2f, %.2f)' % (minW, maxW))
        draw.text((0, 0), wText, fill=cText, font=font,)
        labelTextSize = int(scale*150)
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', labelTextSize)
        
     ##################################### Code #####################################
        # For each layer
        for i in range(len(self.topology)):
            xStep = width/layers
            # Color acording to layer type
            if i == 0:
                infill = cInput
            elif i == len(self.topology)-1:
                infill = cOutput
            else:
                infill = cHidden
            
            # Draw vertical lines for each layer (in debug mode for reference)
            if(grid): draw.line((xStep * i,0,xStep * i,height), width=1)
            
            # Draw labels if they exist
            yStep = height/(self.topology[i])
            yOffset = yStep/2 - labelTextSize/2
            if inputLabels is not None and i == 0:
                for j in range(len(inputLabels)):
                    draw.text((0, yStep * j + yOffset
                               ), inputLabels[j], fill=cText, font=font)
            if outputLabels is not None and i == len(self.topology) - 1:
                for j in range(len(outputLabels)):
                    draw.text((width - (blockSizeX - blockSizeY) + lineClearance, 
                               yStep * j + yOffset
                               ), outputLabels[j], fill=cText, font=font)
            
            # For each node
            for j in range(self.topology[i] + 1):
                # X coordinates of node center
                verClearance = (height/self.topology[i] - 2*nSize)/2
                yStep = height/(self.topology[i])
                centerX = xStep * i + nSize + horClearance
                centerY = yStep * j + nSize + verClearance
                radius = nSize
                if j == self.topology[i]:
                    drawingBias = True
                    infill = cBias
                    radius = bSize
                    # All bias circles are alinged
                    centerY = height+(heightPadding-height)/2
                    if(grid):
                        draw.line((0, centerY, width, centerY))
                        draw.line((centerX, height, centerX, heightPadding))
                else:
                    drawingBias = False
                
                # Draw lines for weights
                if i < len(self.topology) - 1:
                    start = [centerX, centerY]
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
                # Excludes last circle (if not, it wil draw a gost bias next to output layer)
                if not(i == len(self.topology) - 1 and j == self.topology[i]):
                    ### Draw Circles ###
                    # First the bigger background to clear anything (clearance between circle and lines)
                    self.draw_circle(draw, centerX, centerY, radius + lineClearance, cBackground)
                    # Second the border circle
                    self.draw_circle(draw, centerX, centerY, radius, cBorder)
                    # Fill it with the correct botder size
                    self.draw_circle(draw, centerX, centerY, radius - borderSize, infill)
                
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
    
    NN = NeuralNet([2, 3, 1], 5541)
    NN.draw_diagram(inputLabels=['Study', 'Sleep'], outputLabels=['Score'])
    NN.save_image()
    
    o = NN.forward_propagation(X)
    print(o)