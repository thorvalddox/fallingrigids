import sys
import random
import re
import glob

import pygame
from pygame.locals import *

import pymunk
import pymunk.pygame_util
import pymunk.constraint

import pybrain
from pybrain.structure.evolvables.evolvable import Evolvable
from pybrain.optimization import HillClimber, GA

import numpy as np
import pandas as pd
import cv2
import subprocess as sp

def getfilename():
    files = glob.glob("ai*.avi")
    #def extract(f):
    #    s = re.findall("\d+$",f)
    #    return (int(s[0]) if s else -1)
    #try:
    #    return "ai{:05}".format(extract(max(files,key=extract))+1)
    #except ValueError:
    #    return "ai00001"
    return "ai{:05}".format(len(files))

class Universe():
    max_idle = 100
    end_time = 5000
    aslope = 0

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Falling over simulator")
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -50.0)
        self.init_ground()
        self.faller = None
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.timecounter = 0
        self.max_movement = 0.0
        self.idlecounter = 0
        self.font = pygame.font.SysFont('Courier', 20)
        self.crnum = 0
        self.bestfaller = None
        self.maxfitness = -1000
        self.record = Camera(getfilename())

    def handle_close(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                print("exit")
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                print("exit")
                sys.exit(0)

    def run_step(self):
        self.faller.update(self.timecounter)
        self.handle_close()
        self.space.step(1 / 50.0)
        self.timecounter += 1
        if self.max_movement < self.current_movement():
            self.max_movement = self.current_movement()
            self.idlecounter = 0
        else:
            self.idlecounter += 1

    def draw_step(self):
        self.screen.fill((255, 255, 255))
        text = self.font.render(
            f'gen: {self.crnum} fitness: {self.current_movement():>6.2f};{self.max_movement:>6.2f}; {self.maxfitness:>6.2f}'
                , False, (0, 0, 0))
        self.screen.blit(text, (0, 0))
        # self.space.debug_draw(self.draw_options)
        self.draw()
        if self.timecounter % 3 == 0:
            self.record.capture_frame(self.screen)
        pygame.display.flip()
        self.clock.tick(50)

    def draw(self):
        offset = self.faller.body.position
        scaling = Faller.dsize / 150
        self.faller.draw(self.screen, offset, scaling) 

        ground = create_pixel_points([(0,0),(600,600*self.aslope)], offset, scaling)
        pygame.draw.line(self.screen, [0, 0, 0], *ground)
    def init_ground(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (300, -20/np.cos(np.arctan(self.aslope)))
        ground = pymunk.Segment(body, (-300.0, 0), 
                                      ( 300.0,  600.0*self.aslope), 20.0)
        ground.friction = 1
        #ground.elasticity = 0.5
        self.space.add(ground, body)

    def init_faller(self, faller):
        if self.faller is not None:
            self.space.remove(*self.space.shapes)
            self.space.remove(*self.space.bodies)
            self.space.remove(*self.space.constraints)
            self.init_ground()
        self.faller = faller
        self.faller.build_physics(self.space,self.aslope)

        self.max_movement = 0.0
        self.idlecounter = 0
        self.timecounter = 0

    def current_movement(self):
        #return -self.faller.body.angle/np.pi*180
        return self.faller.fitness()

    def do_simulation(self, faller, animate=True, noStop=False):
        self.crnum += 1
        if self.crnum % 100 == 0:
            print(f"number {self.crnum:05}")

        self.run_sim(faller,animate,noStop)
        if self.max_movement > self.maxfitness:
            self.maxfitness = self.max_movement 
            self.bestfaller = faller
            if not animate:
                self.run_sim(faller,True,noStop)
        return self.max_movement

    def run_sim(self,faller,animate,noStop):
        self.init_faller(faller)
        while (self.idlecounter < self.max_idle and self.timecounter < self.end_time) or noStop:
            if animate:
                self.draw_step()
            self.run_step()

class Faller(Evolvable):
    offset = 100
    dsize = 40
    mutrate = 0.01
    minsize = 1

    def __init__(self, spokes=..., angle=0):
        Evolvable.__init__(self)


        self.spokeProjFrac = np.cos(2*np.pi/60)
        #print(self.spokeProjFrac)
        if spokes is ...:
            self.randomize()
        else:
            self.spokes = spokes
            self.sangle = angle
        self.spokes = np.clip(self.spokes, 0, 1)

    @property
    def spokenum(self):
        return len(self.spokes)

    def fitness(self):
        return -self.body.angle/np.pi*180

    def fixspokes(self):
        leftProj = self.spokeProjFrac*np.roll(self.spokes,-1)
        rightProj = self.spokeProjFrac*np.roll(self.spokes,1)
        ConvProj = (leftProj + rightProj)/2
        self.spokes = np.maximum(ConvProj,self.spokes)

    def randomize(self):
        self.spokes = np.ones(60)#np.random.rand(60)
        self.sangle = 0#np.random.rand() * 2 * np.pi
        self.mutate()

    def mutate(self):
        c = np.random.normal(0, self.mutrate, self.spokenum)
        conv = c
        for i in range(1,12):
            conv += np.roll(c,i)
        self.spokes += conv/12  
        self.spokes = np.clip(self.spokes,self.minsize/self.dsize,1)

        if np.random.rand() < 0.05:
            index = np.random.randint(self.spokenum)
            np.insert(self.spokes, index, self.spokes[index])
        elif np.random.rand() < 0.05:
            index = np.random.randint(self.spokenum)
            np.delete(self.spokes, index)
        self.spokes = np.clip(self.spokes, 0, 1)

        self.sangle += np.random.normal(0, self.mutrate * 2 * np.pi)
        
        
        #self.fixspokes()

    def copy(self):
        return type(self)(self.spokes)

    def build_physics(self, space, slope=0):
        vertices = list(self.get_vertices())
        moment = pymunk.moment_for_poly(1, vertices)
        self.body = pymunk.Body(1, moment, pymunk.Body.DYNAMIC)
        self.body.position = (self.offset,
                              self.dsize*np.max(self.spokes)+slope*self.offset)
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.friction = 100
        space.add(self.body, self.shape)
        self.body.center_of_gravity = (self.shape.center_of_gravity.x,
                                       self.shape.center_of_gravity.y)
        


    def get_vertices(self):
        for i in range(self.spokenum):
            phi = 2 * np.pi * (i) / self.spokenum + self.sangle
            rho = self.spokes[i] * self.dsize
            yield rho * np.cos(phi), rho * np.sin(phi)

    def draw(self, screen, offset, scaling):
        pp = create_pixel_points(self.get_world_vertices(), offset, scaling)
        center = create_pixel_points([self.body.position],offset,scaling)[0]
        mcenter = self.create_pixel_point_local(self.body.center_of_gravity,
                                                offset,scaling
                                               )
        pygame.draw.polygon(screen, [255, 0, 0], pp, 2)
        pygame.draw.circle(screen,[255,0,0],mcenter,5)

                             
        
    def get_world_vertices(self):
        for v in self.get_vertices():
            yield(self.get_rot_coords(v))

    def get_rot_coords(self,v):
        return pymunk.Vec2d(v).rotated(self.body.angle) + self.body.position

    def create_pixel_point_local(self,v,offset,scaling):
        return create_pixel_points([self.get_rot_coords(v)],offset,scaling)[0]

    def update(self, timecounter):
        pass


def transl(p, offset, scale):
    x, y = p
    dx, dy = offset
    return (x - dx) / scale, (y - dy) / scale


def create_pixel_points(points, offset, scaling):
    rpoints = (transl(tuple(x), offset, scaling) for x in points)
    vpoints = (transl(x, (-400, -300), 1) for x in rpoints)
    return [(int(x), 600 - int(y)) for x, y in vpoints]


class Animal(Evolvable):
    offset = 100
    dsize = 40
    mutrate = 0.3

    def __init__(self):
        self.mass = 4
        self.width = 2
        self.limbs = []
        self.add_legs(self.dsize)
        self.add_legs(self.dsize * 0.5)

    def fitness(self):
        return self.body.position[0] - self.offset

    def add_leg(self, pos, segments=2):
        target = self
        for _ in range(segments):
            target = Limb(target, (pos, 0))
            target.randomize()
            self.limbs.append(target)

    def add_legs(self, pos, segments=2):
        self.add_leg(pos, segments)
        self.add_leg(-pos, segments)

    def update(self, time):
        part = np.sin(time / 20)
        for l in self.limbs:
            l.update(part)

    def mutate(self):
        for l in self.limbs:
            l.mutate(self.mutrate)

    def randomize():
        for l in self.limbs:
            l.randomize()

    def build_physics(self, space):
        moment = pymunk.moment_for_segment(
            self.mass, (-self.dsize, 0), (self.dsize, 0), self.width * 2)
        self.body = pymunk.Body(self.mass, moment, pymunk.Body.DYNAMIC)
        self.body.position = (self.offset, self.dsize)
        border = build_borders(-self.dsize, self.dsize, -
                               self.width, self.width)
        self.shape = pymunk.Poly(self.body, border)
        #self.shape.friction = 1
        for l in self.limbs:
            l.build_physics(space)
        space.add(self.body, self.shape)
        self.update(0)

    def draw(self, screen, offset, scaling):
        pp = create_pixel_points(self.get_world_vertices(), offset, scaling)
        pygame.draw.polygon(screen, [255, 0, 0], pp, 2)
        hords = [(x + self.body.position[0] // 10)
                 * 10 for x in range(-20, 21)]
        x0, y0 = create_pixel_points([(0, 0)], offset, scaling)[0]
        pygame.draw.line(screen, [0, 0, 0], (0, y0), (800, y0))
        rs = create_pixel_points([(x, -10) for x in hords], offset, scaling)
        for x, y in rs:
            pygame.draw.line(screen, [0, 0, 0], (x, y0), (x, y))
        for l in self.limbs:
            l.draw(screen, offset, scaling)

    def get_world_vertices(self):
        for v in self.shape.get_vertices():
            yield(v.rotated(self.body.angle) + self.body.position)


class Limb:
    density = 0.01
    widthpart = 0.1
    maxlen = 10
    minlen = 2

    def __init__(self, parent, attpos, length=5, minangle=0, maxangle=0):
        self.parent = parent
        self.attpos = attpos
        self.length = length
        self.minangle = minangle
        self.maxangle = maxangle
        self.verify()

    def verify(self):
        self.length = np.clip(self.length, self.minlen, self.maxlen)
        self.minangle = np.clip(self.minangle, 0, np.pi / 6)
        self.maxangle = np.clip(self.maxangle, 0, np.pi / 6)

    @property
    def mass(self):
        return self.length * self.density

    @property
    def width(self):
        return self.length * self.widthpart

    def mutate(self, rate):
        self.length += np.random.normal(0, rate * self.length)
        self.minagle += np.random.normal(0, rate / 20)
        self.maxangle += np.random.normal(0, rate / 20)
        self.verify()

    def randomize(self):
        self.length = np.random.normal(10, 4)
        self.minagle = np.random.normal(0, 0.2)
        self.maxangle = np.random.normal(0, 0.2)
        self.verify()

    def build_physics(self, space):
        moment = pymunk.moment_for_segment(
            self.mass, (0, 0), (0, -self.length), self.width)
        if isinstance(self.parent, Limb):
            self.attpos = (0, -self.parent.length)
        self.body = pymunk.Body(self.mass, moment, pymunk.Body.DYNAMIC)
        self.body.position = self.attpos[0] + self.parent.body.position[0], \
            self.attpos[1] + self.parent.body.position[1] + self.length / 2
        #border = build_borders(self.width,-self.width,0,-self.length)
        self.shape = pymunk.Segment(
            self.body, (0, 0), (0, -self.length), self.width)
        self.shape.friction = 20
        #self.shape.elasticity = 0.5
        self.joint = pymunk.constraint.PinJoint(
            self.parent.body, self.body, self.attpos, (0, 0))
        self.joint.distance = 0
        self.joint.collide_bodies = False
        self.mussle = pymunk.constraint.DampedRotarySpring(
            self.parent.body, self.body, self.minangle, 500, 50)
        self.shape.friction = 1
        space.add(self.body, self.shape, self.joint, self.mussle)
        self.update(0)

    def draw(self, screen, offset, scaling):
        pp = create_pixel_points(self.get_world_vertices(), offset, scaling)
        x0, y0 = create_pixel_points([(0, 0)], offset, scaling)[0]
        pygame.draw.polygon(screen, [255, 0, 0], pp, 2)

    def update(self, part):
        self.mussle.rest_angle = (
            self.maxangle + self.minangle - part * (self.maxangle - self.minangle)) / 2

    def get_world_vertices(self):
        for v in (self.shape.a, self.shape.b):
            yield(v.rotated(self.body.angle) + self.body.position)

    def get_con_vertices(self):
        for v in (self.shape.a, self.shape.b):
            yield(self.shape.b.rotated(self.body.angle) + self.body.position)

class Camera:
    def __init__(self,filename):
        self.proc = sp.Popen(['ffmpeg','-y','-f', 'rawvideo',
                                      '-vcodec','rawvideo',
                                      '-s', "800x600",
                                      '-pix_fmt','rgba',
                                      '-r','10',
                                      '-i','-',
                                      '-an',
                                      '-vcodec', 'qtrle',
                                          filename + '.mov'], stdin=sp.PIPE)
        self.filename = filename
    def capture_frame(self,screen):
        #get the rgb channels from the screen and make a dummy alpha
        r = pygame.surfarray.pixels_red(screen)
        g = pygame.surfarray.pixels_green(screen)
        b = pygame.surfarray.pixels_blue(screen)
        a = np.ones_like(r)*255

        #merge the rgba channels
        mergedImage = cv2.merge((r,g,b,a))
        rows,cols,draws = mergedImage.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        rotatedMergedImage = cv2.warpAffine(mergedImage,M,(rows,cols))
        
        rotatedMergedImage = cv2.flip(rotatedMergedImage,1)

        black = rotatedMergedImage[:,:,0] == 0

        rotatedMergedImage[black] = [0,0,0,0]

        self.proc.stdin.write(rotatedMergedImage)
        
        r = []
        g = []
        b = []
        a = []
    def finish(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.proc = sp.Popen(['ffmpeg',"-i",self.filename+".mov","-acodec",
                              "copy","-vcodec","copy",self.filename+".avi"],
                             stdin=sp.PIPE)
        self.proc.wait()


def build_borders(x1, x2, y1, y2):
    return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))


def main():
    start = Faller(...)
    #start = Animal()
    generate(start)

def generate(start):
    u = Universe()

    def fitness(f):
        return u.do_simulation(f, False)
    l = HillClimber(fitness, start, maxEvaluations=int(sys.argv[1]))
    best, fitn = l.learn()
    print(f"fitness: {fitn}")
    #u.do_simulation(u.bestfaller, noStop=False)
    u.record.finish()

if __name__ == '__main__':
    main()
