import pyglet
from PIL import Image, ImageDraw
from pyglet.window import key, FPSDisplay
import math
import numpy as np
from helperFunctions import *
from NeuralNet import *
from memory_profiler import profile
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import time

class GameObject:
    def __init__(self, window, position, visual=None, velocity=[0,0], batch=None):
        self.window     = window        # Pointer to a pyglet window object
        self.position   = position      # (x, y) list for inicial position
        self.visual     = visual        #
        self.velocity   = velocity      # (velx, vely) list for object velocity
        self.batch      = batch         # Pointer to a pyglet batch
        
        # Load image
        if type(visual) == str:
            img = pyglet.image.load(visual)
            self.sprite = pyglet.sprite.Sprite(img, position[0], position[1], batch=self.batch)
            self.boundingBox = (img.width, img.height)
        if type(visual) == Image.Image:
            raw_image = visual.tobytes()
            img = pyglet.image.ImageData(visual.width, visual.height, 'RGBA', raw_image, pitch=-visual.width * 4)
            self.sprite = pyglet.sprite.Sprite(img, position[0], position[1], batch=self.batch)
            self.boundingBox = [visual.width, visual.height]
        
        self.center = [self.position[0] + self.boundingBox[0]/2, self.position[1] + self.boundingBox[1]/2]
            
    def draw(self):
        self.sprite.draw()
        
    def update(self, dt):
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        
        # Avoid weird behaviour when player gous out of screen on low fps
        hTest = self.hit_border([0, 0, -1, 101])[1]
        if hTest == 1:
            self.position[1] = self.window.height - self.boundingBox[1] - 100
        elif hTest == -1:
            self.position[1] = 0
        
        self.sprite.x = self.position[0]
        self.sprite.y = self.position[1]
        self.center = [self.position[0] + self.boundingBox[0]/2, self.position[1] + self.boundingBox[1]/2]
    
    def hit_border(self, padding = [0, 0, 0, 0]):
        # padding: offset from xUpper, Xlower, yUpper yLower border of window
        eval = [0,0]
        # Check X
        if self.position[0] <= padding[0]:
            eval[0] = -1
        elif self.position[0] >= self.window.width - self.boundingBox[0] - padding[1]:
            eval[0] = 1
        # Check Y
        if self.position[1] <= padding[2]:
            eval[1] = -1
        elif self.position[1] >= self.window.height - self.boundingBox[1] - padding[3]:
            eval[1] = 1
        # Return if hit [x, y]
        # -1 for lower and 1 for upper value
        return eval
    
    def rectangle_collision(self, gObject):
        rect1 = {'x': self.position[0], 'y': self.position[1], 'width': self.boundingBox[0], 'height': self.boundingBox[1]}
        rect2 = {'x': gObject.position[0], 'y': gObject.position[1], 'width': gObject.boundingBox[0], 'height': gObject.boundingBox[1]}
        
        if (rect1["x"] < rect2["x"] + rect2["width"] and
            rect1["x"] + rect1["width"] > rect2["x"] and
            rect1["y"] < rect2["y"] + rect2["height"] and
            rect1["y"] + rect1["height"] > rect2["y"]):
                return True
        return False
    
    def distanceVec(self, gObject):
        return [gObject.center[0] - self.center[0],  gObject.center[1] - self.center[1]]
        
class PlayerAuto(GameObject):
    def __init__(self, window, position, visual=None, velocity=[0,0], batch=None):
        super().__init__(window, position, visual=visual, velocity=velocity, batch=batch)
        self.points = 0
        self.initial_velocity = list(velocity)
        self.tol = 100
        
    def score(self):
        self.points += 1
    
    def update(self, dt, ball):
        
        # Calculate next desired velocity
        # In this case, move to align center of ball with center of plate
        # It has a dead zone in the middle of + and - deadzone pixels
        deadzone = 5
        distVec = self.distanceVec(ball)[1]
        if distVec > deadzone:
            self.velocity[1] = abs(float(self.initial_velocity[1]))
        elif distVec < -deadzone:
            self.velocity[1] = -abs(float(self.initial_velocity[1]))
        else:
            self.velocity[1] = 0
        
        borderHit = self.hit_border([0, self.window.width - self.window.playArea[0], 0, self.window.height - self.window.playArea[1]])
        # if hit horizontal walls
        # and trying to move past it
        # Set velociti to zero and move to wall limit (this move is to avoid ugly looks due to variations on dt)
        if borderHit[1] == 1 and self.velocity[1] > 0:
            self.velocity[1] = 0
            self.position[1] = 400 - self.boundingBox[1]
        elif borderHit[1] == -1 and self.velocity[1] < 0:
            self.velocity[1] = 0
            self.position[1] = 0
        
        super().update(dt)

class PlayerManual(GameObject):
    def __init__(self, window, position, visual=None, velocity=[0,0], batch=None):
        super().__init__(window, position, visual=visual, velocity=velocity, batch=batch)
        self.points = 0
        self.maxSpeed = velocity[1]
    
    def score(self):
        self.points += 1
    
    def update(self, dt, ball):
        
        if self.window.keys[key.UP]:
            self.velocity[1] = self.maxSpeed
        elif self.window.keys[key.DOWN]:
            self.velocity[1] = -self.maxSpeed
        else:
            self.velocity[1] = 0
        borderHit = self.hit_border([0, self.window.width - self.window.playArea[0], 0, self.window.height - self.window.playArea[1]])
        # Hit horizontal walls
        if borderHit[1] == 1:
            self.velocity[1] = -abs(self.velocity[1])
        elif borderHit[1] == -1:
            self.velocity[1] = abs(self.velocity[1])
        super().update(dt)

class PlayerNN(GameObject):
    def __init__(self, window, position, neuralNet, ballSpeed, enemyPlayer, visual=None, velocity=[0,0], batch=None):
        super().__init__(window, position, visual=visual, velocity=velocity, batch=batch)
        self.points = 0
        self.maxSpeed = velocity[1]
        self.neuralNet = neuralNet
        self.ballSpeedModule = ballSpeed
        self.enemyPlayer = enemyPlayer
        
    def score(self):
        self.points += 1
    
    def update(self, dt, ball):
        # Build normalized state vector
        state = [[
            self.center[1]/self.window.playArea[1],
            self.enemyPlayer.center[1]/self.window.playArea[1],
            ball.center[0]/self.window.playArea[0],
            ball.center[1]/self.window.playArea[1],
            ball.velocity[0]/self.ballSpeedModule,
            ball.velocity[1]/self.ballSpeedModule
        ]]
        
        # Eval neuralnet
        action = self.neuralNet.forward_propagation(state)
        
        
        # For negative values go down, for positive go up. with treshold around 0 for stop
        if action <= 0.48:
            self.velocity[1] = -self.maxSpeed
        elif action >= 0.52:
            self.velocity[1] = self.maxSpeed
        else:
            self.velocity[1] = 0
        borderHit = self.hit_border([0, self.window.width - self.window.playArea[0], 0, self.window.height - self.window.playArea[1]])
        # Hit horizontal walls
        if borderHit[1] == 1:
            self.velocity[1] = -abs(self.velocity[1])
        elif borderHit[1] == -1:
            self.velocity[1] = abs(self.velocity[1])
        super().update(dt)

class Ball(GameObject):
    def __init__(self, window, game, position, visual=None, velModule=300, batch=None):
        self.module = velModule
        self.angle = 45
        self.velocity = [0, 0]
        super().__init__(window, position, visual=visual, velocity=self.velocity, batch=batch)
        self.startDelay = 0.5
        self.startCount = 0
        self.movingModule = self.module
        self.startPosition = [self.window.playArea[0]//2 - self.boundingBox[0]//2, self.window.height//2 - self.boundingBox[1]//2 - 50]
        self.position = list(self.startPosition)
        self.rolling = False
        self.game = game
    
    def set_cartesian_vel(self, mod, ang):
        self.velocity = [mod*math.cos(ang*math.pi/180), mod*math.sin(ang*math.pi/180)]
    
    def update(self, dt, p1, p2):
        borderHit = self.hit_border([0, self.window.width - self.window.playArea[0], 0, self.window.height - self.window.playArea[1]])
        if self.rolling:
            # Hit horizontal walls
            # Bounce with same angle
            if borderHit[1] == 1:
                self.velocity[1] = -abs(self.velocity[1])
            elif borderHit[1] == -1:
                self.velocity[1] = abs(self.velocity[1])
            
            # Hit vertical walls
            # Score for player and restart
            if borderHit[0]:
                if borderHit[0] == -1:
                    p2.score()
                else:
                    p1.score()
                #  update scoreboard
                self.game.updateScoreBoard()
                self.rolling = False
                self.velocity = [0,0]
                self.position = list(self.startPosition)
        
            # When hit player, deflect and change velocity direction depending on where it hit the plate
            N = 50
            if self.rectangle_collision(p1):
                # deflect from -plate height and plate height to -N and N
                D = self.distanceVec(p1)[1]
                PlateSize = p1.boundingBox[1]
                angle = mapVal(D, -PlateSize/2, PlateSize/2, -N, N)
                self.set_cartesian_vel(self.module, angle)
                
            elif self.rectangle_collision(p2):
                D = self.distanceVec(p2)[1]
                PlateSize = p1.boundingBox[1]
                # Same thing other direction
                angle = mapVal(D, -PlateSize/2, PlateSize/2, 180-N, 180+N)
                self.set_cartesian_vel(self.module, angle)
        # Start position and velocity
        else:
            self.startCount += dt
            if self.startCount >= self.startDelay:
                self.startCount = 0
                self.rolling = True
                
                # Random shooting angle between 80 and -80
                self.set_cartesian_vel(300, np.random.randint(-80, 80))
                # Random direction
                if np.random.random() >= 0.5:
                    self.velocity[0] *= -1
                
            
        super().update(dt)

class MatchHandler:
    def __init__(self, window, p1, p2, nn=None, color=(255,255,255,255)):
        self.window = window
        self.playersSpeed = 180
        self.ballSpeedModule = 400
        self.endScore = 11
        
        self.victory = 0
        self.neuralNet = nn
        self.color = color
        
        # Actors
        player_image = Image.new('RGBA', (8,80), color)
        
        ball_radius = 9
        ball_image = Image.new('RGBA', (ball_radius,ball_radius), (0,0,0,255))
        draw = ImageDraw.Draw(ball_image)
        draw.ellipse((0,0,ball_radius,ball_radius), color)
        del draw
        
        
        # Batch
        self.actors = pyglet.graphics.Batch()
        
        # Create players based on parameters
        # Player one can be auto or manual
        if p1 == 'Auto':
            self.playerOne = PlayerAuto(self.window, [15, self.window.playArea[1]//2], player_image, [0, self.playersSpeed], batch=self.actors)
        elif p1 == 'Manual':
            self.playerOne = PlayerManual(self.window, [15, self.window.playArea[1]//2], player_image, [0, self.playersSpeed], batch=self.actors)
        # Player 2 can be auto, manual or neuralnet
        if p2 == 'Auto':
            self.playerTwo = PlayerAuto(self.window, [self.window.playArea[0]-8-15, self.window.playArea[1]//2], player_image, [0, self.playersSpeed], batch=self.actors)
        elif p2 == 'Manual':
            self.playerTwo = PlayerManual(self.window, [self.window.playArea[0]-8-15, self.window.playArea[1]//2], player_image, [0, self.playersSpeed], batch=self.actors)
        elif p2 == 'NeuralNet':
            self.playerTwo = PlayerNN(self.window, [self.window.playArea[0]-8-15, self.window.playArea[1]//2], self.neuralNet, self.ballSpeedModule, self.playerOne, player_image, [0, self.playersSpeed], batch=self.actors)
        
        self.updateScoreBoard()
        # Create ball
        self.ball = Ball(self.window, self, (200,200), ball_image, self.ballSpeedModule, batch=self.actors)
        
        # If neuralnetwork, acomodate for drawing on window
        if nn is not None:
            # generate image
            netDiagram = nn.draw_diagram(inputLabels = ['Self_Y', 'P1_Y', 'Ball_X', 'Ball_Y', 'Ball_X\'', 'Ball_Y\'', 'Bias'], outputLabels=['Motion'])
            width = netDiagram.width
            height  = netDiagram.height
            aspectRatio = width/height
            leftClearance = 20
            
            # Scale to fit on rigth side of screen
            height = self.window.height
            width = int(aspectRatio*height)        
            netDiagram = netDiagram.resize((width, height), Image.ANTIALIAS)
            
            # Add to menu
            self.nnDiagram = GameObject(self,[self.window.playArea[0]+leftClearance, 0], netDiagram)
            
            # Resize window to fit
            self.window.set_size(self.window.playArea[0] + netDiagram.width + leftClearance, self.window.height)

    def update(self,dt):
        if not self.victory:
            self.ball.update(dt, self.playerOne, self.playerTwo)
            self.playerOne.update(dt, self.ball)
            self.playerTwo.update(dt, self.ball)
            
            # end game condition
            if self.playerOne.points >= self.endScore:
                self.victory = 1
            elif self.playerTwo.points >= self.endScore:
                self.victory = 2
    
    def updateScoreBoard(self):
        self.p1p = pyglet.text.Label(str(self.playerOne.points), x=self.window.playArea[0]/4, y = 450, align='center', font_size=26, bold=True, color=self.color)
        self.p1p.anchor_x = self.p1p.anchor_y = 'center'
        self.p2p = pyglet.text.Label(str(self.playerTwo.points), x=3*self.window.playArea[0]/4, y = 450, align='center', font_size=26, bold=True, color=self.color)
        self.p2p.anchor_x = self.p2p.anchor_y = 'center'
        
    def on_draw(self):
        if not self.victory:
            self.actors.draw()
            self.p1p.draw()
            self.p2p.draw()
            
class GameWindow(pyglet.window.Window):
    def __init__(self, x, y, title):
        super().__init__(x, y, title)
        self.frame_rate = 1/24
        self.fps_display = FPSDisplay(self)
        self.fps_display.label.font_size = 10
        self.playArea = [self.width, self.height-100]
        self.timeout = 0.5 # Time before closing window
        self.closeWindow = False # Flag to know when to close window
        
        # Key monitor
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
        # Menu
        self.interface = pyglet.graphics.Batch()
        self.interfaceObjects = []
        upperBar = Image.new('RGBA', (self.width, 4), (255,255,255,200))
        midLine = Image.new('RGBA', (2, 400), (255,255,255,60))
        self.interfaceObjects.append(GameObject(self,[0,400],upperBar,batch=self.interface))
        self.interfaceObjects.append(GameObject(self,[self.width//2 - 1, 0], midLine,batch=self.interface))        
        
        # Run games
        # List of type [winner, score p1, score p2, matcHandler]
        self.games = []
        self.neuralNet = False
        self.best = [-1,-1,-1,-1]
      
    def update(self,dt):
        # Check if there is any game running:
        self.closeWindow = all(g[0] != 0 for g in self.games)
        
        # if there are, update all running games
        if not self.closeWindow:
            for game in self.games:
                if game[0] == 0:
                    game[3].update(dt)
                    game[0] = game[3].victory
                    game[1] = game[3].playerOne.points
                    game[2] = game[3].playerTwo.points
        else:
            self.timeout -= dt
            if self.timeout <= 0:
                self.close()
    
    def on_draw(self):
        self.clear()
        
        # Draw highest score neuralnet
        if self.neuralNet and self.best[3] != -1:
            self.best[3].nnDiagram.draw()
            
        # Draw active games if exists
        if not self.closeWindow:
            self.interface.draw()
            for game in self.games:
                if game[0] == 0:
                    game[3].on_draw()
                    if game[2] > self.best[2]:
                        self.best = game 
        
        # Draw all game results
        else:
            text = 'Results:\n'
            for i, game in enumerate(self.games):
                text += "  Game %i: %ix%i\n" % (i, game[1], game[2])
            label = pyglet.text.Label(text,
                                       y = self.height - 30,
                                       x = 10,
                                       multiline=True,
                                       width = self.width,
                                       height = self.height)
            label.draw()
        self.fps_display.draw()
    
    def addGame(self, typeP1, typeP2, color=(255,255,255,255), nn=None):
        self.games.append([0, 0, 0, MatchHandler(self,
                                        str(typeP1),
                                        str(typeP2),
                                        nn=nn,
                                        color=tuple(color))])
    
        if typeP2 == 'NeuralNet':
            self.neuralNet = True
        
    def results(self):
        results = []
        # separa22te games from results
        for game in self.games:
            results.append(list(game[:-1]))
        return results

def plot(ax, development):
    plt.cla()
    ax.plot(list(zip(*development))[0], color='green')
    ax.plot(list(zip(*development))[1], color='yellow')
    ax.plot(list(zip(*development))[2], color='red')
    ax.grid()
    ax.set_title('Fitness over time')
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Epochs')
    plt.draw()
    plt.pause(0.001)

def diferential_selection(population, fitness, topology):
    populationSize = len(population)
    mutationFactor = 0.2
    crossoverFactor = 0.2
    populationCandidates = []
    candidatesFitness = np.array([])
    
    for i in range(populationSize):
        populationCandidates.append(NeuralNet(topology)) # Just to initialize neuralnet, weigths will be replaced later
    
    # Build new population of neuralnetworks
    for individualIndex in range(populationSize):
        # get weigths of individual
        individual = population[individualIndex].get_weights()
        
        # Create new individual based on others for DNA donation
        # Get 3 random individuals (excluding current) for diferential algoritm
        candidatesIndexList = list(range(populationSize))
        candidatesIndexList.remove(individualIndex)
        
        # When choosing, gives higher probabilities to use good individuals
        # Calculate propability based on fitness
        p = [mapVal(x, -11.0, 11.0, 0.2, 0.8) for x in fitness]
        chosen = np.random.choice(candidatesIndexList, 3, p)
        
        # Get diference of the two first one:
        diff = population[chosen[0]].get_weights() - population[chosen[1]].get_weights()
        
        # mutationFactor*diff + third random vector
        donor = mutationFactor*diff + population[chosen[2]].get_weights()
        
        # Crossover of donor and individual:
        # Replace the individual gene for the donor randomly
        offspring = np.array([])
        for gene in range(len(donor)):
            if np.random.random() <= crossoverFactor:
                offspring = np.append(offspring, donor[gene])
            else:
                offspring = np.append(offspring, individual[gene])
                
        # Add offspring gene code to candidates
        populationCandidates[individualIndex].set_weights(offspring)
        
    # Evaluate fitness of offspring and elders (to make sure it wasnt luck)
    candidatesFitness = get_fitness(populationCandidates + population)
    
    fitness = (np.array(candidatesFitness[populationSize:]) + np.array(fitness))/2 # mean of previous value and current (reduce random fluctuations whilst still valuing the mos resent result over the others)
    candidatesFitness = candidatesFitness[:populationSize]
    
    # Replace individuals that performed worse than theyr offsprring (greedy method)
    for candidateIndex in range(populationSize):
        if candidatesFitness[candidateIndex] > fitness[candidateIndex]:
            population[candidateIndex] = populationCandidates[candidateIndex]
            fitness[candidateIndex] = candidatesFitness[candidateIndex]
    return population, fitness

def simple_selection(population, fitness, topology, color):
    populationSize = len(population)
    killRate = 0.30 # percentage of the weakest to be killed
    ReproductionRate = 0.15 # Only the strongest one reproduce
    mutationFactor = 0.005
    multiFactorMin = -1.2
    multiFactorMax = 1.2
    intruderChance = 0.005
    
    # Order population by fitness
    sortedIndexes = np.flip(np.argsort(fitness))
    population[:] = [population[i] for i in sortedIndexes]
    
    # Kill weakest individuals
    population = population[:int(populationSize*killRate)]
    fitness = fitness[:int(populationSize*(1-killRate))]
    color = color[:int(populationSize*(1-killRate))]
    
    # Add random individual (for new genes in gene pool)
    # Inserted at the top of the list to make sure it reproduces at least one
    # This helps getting out of local minimums
    if np.random.random() <= intruderChance:
        population.insert(0, NeuralNet(topology))
        color.insert(0, [np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255), 150])
    
    # Breed
    livePopSize = len(population)
    while(len(population) < populationSize):
        # Parent selection
        idx = np.random.choice(list(range(int(livePopSize*ReproductionRate))), 2)
        
        p1 = population[idx[0]].get_weights()
        p2 = population[idx[1]].get_weights()
        
        # Crossover
        crossOverPoint = np.random.randint(0,len(p1)-1) # select crossover point
        child = np.concatenate((p1[:crossOverPoint], p2[crossOverPoint:]))
        
        # Merge parents colors
        childColor = [sum(hue)/2 for hue in zip(*[color[idx[0]], color[idx[1]]])]
        
        # Mutation
        for gene in range(len(child)):
            if np.random.random() <= mutationFactor:
                # Generate multiplication factor 
                scale = multiFactorMin + (np.random.random() * (multiFactorMax - multiFactorMin))
                # Scale gene by random factor
                child[gene] *= scale
                # Mutate color (sligtly change color based on same scale that changed gene)
                cIdx = np.random.randint(0,3)
                childColor[cIdx] += scale*childColor[cIdx]/2
                if(childColor[cIdx]) < 0:
                    childColor[cIdx] = 0
                elif(childColor[cIdx] > 255):
                    childColor[cIdx] = 255
                 

        # Add child to population pool
        population.append(NeuralNet(topology))
        population[-1].set_weights(child)
        color.append(childColor)
    
    # Evaluate population
    fitness = get_fitness(population)
    return population, fitness


def get_fitness(neuralnets, titlePostfix = '', color=None): # why doesnt it free any memory? the scope is closed
    # Uncomment for debugging loop purpuses
    # return np.array([abs(sum(n.get_weights())) for n in neuralnets])
    # return np.random.randint(-11,11,len(neuralnets))
    
    # Generate color randomly if not specified
    colorVec = []
    for i in range(len(neuralnets)):
        colorVec.append((np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255), 150))
    
    # Create game window
    window = GameWindow(500, 500, 'SmartPong' + titlePostfix)    
    
    # Populate game window
    for i in range(len(neuralnets)):
        # Add games to window
        window.addGame('Auto', 'NeuralNet', nn=neuralnets[i], color=colorVec[i])
    # run pyglet widow
    pyglet.clock.schedule_interval(window.update, float(window.frame_rate))
    pyglet.app.run()
    
    # get results
    results = window.results()
    
    # Close pyglet window
    pyglet.clock.unschedule(window.update)
    pyglet.app.exit()
    
    fitness = []
    # Calculate fitness
    for result in results:
        # Change here for different fitness evaluations
        fitness.append(result[2] - result[1]) # value who makes most points and take less hits
    return np.array(fitness)

def main():
    np.set_printoptions(10, linewidth = 92, sign=' ', floatmode='fixed')
    # NEAT algorithm (I think) to train(find) best neuralnet to beat pong
    topology = [6, 5, 1]
    populationSize = 140
    epochs = 400
    
    population = []
    fitness = np.array([])
    
    # For plotting purposes
    development = []
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('SmartPong') 
    
    # Random seed
    np.random.seed(1)
    
    # Time keeping variables
    t0 = time.time()
    tVec = []
    # if not, initialize random population
    print('Initializing neural networks: population of', populationSize)
    for i in range(populationSize):
        population.append(NeuralNet(topology))
    
    # Create color vector for neuralnetworks
    colorVec = []
    for i in range(len(population)):
        colorVec.append([np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255), 150])
    
    # Check if there are any dump files, if there are load from them (can delete each portion separatly)
    if os.path.isfile('./dump_history'):
        print('Loading population history')
        with open('dump_history', 'rb') as fp:
            development = pickle.load(fp)
    
    if os.path.isfile('./dump_color'):
        print('Loading population color')
        with open('dump_color', 'rb') as fp:
            colorVec = pickle.load(fp)
            
    if os.path.isfile('./dump_population'):
        print('Loading population memory')
        with open('dump_population', 'rb') as fp:
            popWeights = pickle.load(fp)
        for i in range(len(popWeights)):
            population[i].set_weights(popWeights[i])

    # Get fitness vector for first generation
    fitness = get_fitness(population, ': Initializing population')
    development.append([max(fitness), sum(fitness)/len(fitness), min(fitness)])
    
    # Plot development
    plot(ax, development)
    tVec.append(time.time()-t0)
    print('Population initialized in %.1f seconds' % tVec[-1])
    print('Training started:')
    # start genetic loop (diferential genetic algoritm)
    for epoch in range(epochs):
        t0 = time.time()
        ETAs = sum(tVec)/len(tVec)*(epochs-epoch)
        ETAm = int(ETAs/60.0)
        ETAs = ETAs - 60*ETAm
        print('Epoch %i/%i: ETA %im %is'% (epoch+1, epochs, ETAm, ETAs))
        
        # Genetic algoritm
        # population, fitness = diferential_selection(population, fitness, topology)
        population, fitness = simple_selection(population, fitness, topology, colorVec)

        # Record of fitness over time
        development.append([max(fitness), sum(fitness)/len(fitness), min(fitness)])
        
        # Plot development
        plot(ax, development)
        
        # Save current results so program can continue from where it started
        with open('dump_history', 'wb') as fp:
            pickle.dump(development, fp)
        with open('dump_population', 'wb') as fp:
            pickle.dump([weight.get_weights() for weight in population], fp)
        with open('dump_color', 'wb') as fp:
            pickle.dump(colorVec, fp)
            
        # Print statistics
        tVec.append(time.time()-t0)
        print('Elapsed time: %is' % tVec[-1])
        bestIndex = 0
        for i in range(populationSize):
            if fitness[i] >= fitness[bestIndex]:
                bestIndex = i
        print('Best player fitness: ', fitness[bestIndex])
        print(population[bestIndex].get_weights())
        print('~ '*46)
        
    print('Training done')
    plt.show()

if __name__ == "__main__":
    main()