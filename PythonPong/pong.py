import pyglet
from PIL import Image, ImageDraw
from pyglet.window import key, FPSDisplay
import math
import random
from helperFunctions import map

random.seed(1)

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
        self.sprite.x = self.position[0]
        self.sprite.y = self.position[1]
        self.center = [self.position[0] + self.boundingBox[0]/2, self.position[1] + self.boundingBox[1]/2]
    
    def hit_border(self, padding = [0, 0, 0, 100]):
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
        
        borderHit = self.hit_border()
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
    
    def score(self):
        self.points += 1
    
    def update(self, dt):
        borderHit = self.hit_border()
        # Hit horizontal walls
        if borderHit[1] == 1:
            self.velocity[1] = -abs(self.velocity[1])
        elif borderHit[1] == -1:
            self.velocity[1] = abs(self.velocity[1])
        super().update(dt)

class Ball(GameObject):
    def __init__(self, window, position, visual=None, velModule=300, batch=None):
        self.module = velModule
        self.angle = 45
        self.velocity = [0, 0]
        super().__init__(window, position, visual=visual, velocity=self.velocity, batch=batch)
        self.startDelay = 2
        self.startCount = 0
        self.movingModule = self.module
        self.startPosition = [window.width//2 - self.boundingBox[0]//2, window.height//2 - self.boundingBox[1]//2 - 50]
        self.position = list(self.startPosition)
        self.rolling = False
    
    def set_cartesian_vel(self, mod, ang):
        self.velocity = [mod*math.cos(ang*math.pi/180), mod*math.sin(ang*math.pi/180)]
    
    def update(self, dt, p1, p2):
        borderHit = self.hit_border()
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
                self.rolling = False
                self.velocity = [0,0]
                self.position = list(self.startPosition)
        
            # When hit player, deflect and change velocity direction depending on where it hit the plate
            N = 50
            if self.rectangle_collision(p1):
                # deflect from -plate height and plate height to -N and N
                D = self.distanceVec(p1)[1]
                PlateSize = p1.boundingBox[1]
                angle = map(D, -PlateSize/2, PlateSize/2, -N, N)
                self.set_cartesian_vel(self.module, angle)
                
            elif self.rectangle_collision(p2):
                D = self.distanceVec(p2)[1]
                PlateSize = p1.boundingBox[1]
                # Same thing other direction
                angle = map(D, -PlateSize/2, PlateSize/2, 180-N, 180+N)
                self.set_cartesian_vel(self.module, angle)
        # Start position and velocity
        else:
            self.startCount += dt
            if self.startCount >= self.startDelay:
                self.startCount = 0
                self.rolling = True
                
                # Random shooting angle between 80 and -80
                self.set_cartesian_vel(300, random.randint(-80, 80))
                # Random direction
                if random.random() >= 0.5:
                    self.velocity[0] *= -1
                
            
        super().update(dt)
class GameWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60
        self.playersSpeed = 180
        self.ballSpeedModule = 400 
        self.fps_display = FPSDisplay(self)
        self.fps_display.label.font_size = 10
        self.endScore = 11
        self.victory = 0
        # Actors
        player_image = Image.new('RGBA', (8,80), (255,255,255,255))
        
        ball_radius = 9
        ball_image = Image.new('RGBA', (ball_radius,ball_radius), (0,0,0,255))
        draw = ImageDraw.Draw(ball_image)
        draw.ellipse((0,0,ball_radius,ball_radius),(255,255,255,255))
        del draw
        
        # Batch
        self.actors = pyglet.graphics.Batch()
        
        self.playerOne = PlayerAuto(self, [15, self.height//2], player_image, [0, self.playersSpeed], batch=self.actors)
        self.playerTwo = PlayerManual(self, [self.width-8-15, self.height//2], player_image, [0, self.playersSpeed], batch=self.actors)
        self.ball = Ball(self, (200,200), ball_image, self.ballSpeedModule, batch=self.actors)
        
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
        
    def update(self,dt):
        if not self.victory:
            if self.keys[key.UP]:
                self.playerTwo.velocity[1] = self.playersSpeed
            elif self.keys[key.DOWN]:
                self.playerTwo.velocity[1] = -self.playersSpeed
            else:
                self.playerTwo.velocity[1] = 0
            
            self.ball.update(dt, self.playerOne, self.playerTwo)
            self.playerOne.update(dt, self.ball)
            self.playerTwo.update(dt)
            
            # end game condition
            if self.playerOne.points >= self.endScore:
                self.victory = 1
            elif self.playerTwo.points >= self.endScore:
                self.victory = 2

    def on_draw(self):
        self.clear()
        if not self.victory:
            self.interface.draw()
            self.actors.draw()
            self.fps_display.draw()
            p1p = pyglet.text.Label(str(self.playerOne.points), x=self.width/4, y = 450, align='center', font_size=26, bold=True)
            p1p.anchor_x = p1p.anchor_y = 'center'
            p1p.draw()
            p2p = pyglet.text.Label(str(self.playerTwo.points), x=3*self.width/4, y = 450, align='center', font_size=26, bold=True)
            p2p.anchor_x = p2p.anchor_y = 'center'
            p2p.draw()
        else:
            txt = pyglet.text.Label('Victory P' + str(self.victory) + ': ' + str(self.playerOne.points), 
                              x = self.width/2, 
                              y = self.height/2, 
                              align='center', 
                              font_size=26, 
                              bold=True
                              )
            txt.anchor_x = txt.anchor_y = 'center'
            txt.draw()
        
if __name__ == "__main__":
    window = GameWindow(500, 500, 'SmartPong')
    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()