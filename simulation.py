# https://github.com/openai/gym/issues/626#issuecomment-310642853
import json
import pymunk
from pymunk.vec2d import Vec2d
import pygame
import scaledrenderer
import sys
import random
import math
import utils
import neat
import os
import pickle
import numpy as np

# mass is in grams
# measurement is in centimeters
# All measurements are based on either the official RoboCup Jr specification or from real-life measurements on an accurate playing field

LENGTH_SCALAR = 2
COLLISION_TYPES = {
    "ball": 1,
    "goal_blue": 2,
    "goal_yellow": 3,
    "wall": 4,
    "robot": 5
}
RESTITUTION = 0.45
robot_touched_ball = False
space = pymunk.Space()
reset_sim = False # if we need to reset on the next loop
fitness = 0
total_steps = 0
MAX_STEPS = 750
total_steps_touching_ball = 0

def create_robot(position):
    global space
    x, y = position
    robot_upscale = 21 # robot diameter = 210 mm (21 cm)
    with open("robot.json", "r") as f:
        robot_data = json.load(f)

    verts = robot_data["rigidBodies"][0]["shapes"][0]["vertices"]
    vertices = []
    for vert in verts:
        vertices.append((vert["x"] * robot_upscale, vert["y"] * robot_upscale))
    
    mass = 700 # robot weights around 1kg
    inertia = pymunk.moment_for_poly(mass, vertices)
    body = pymunk.Body(mass, inertia)
    body.position = x, y

    shapes = robot_data["rigidBodies"][0]["polygons"]
    for shape in shapes:
        verties = []
        for vert in shape:
            vert_x = vert["x"] * robot_upscale
            vert_y = vert["y"] * robot_upscale
            verties.append((vert_x, vert_y))
        poly_shape = pymunk.Poly(body, verties)
        poly_shape.color = pygame.color.THECOLORS["red"]
        poly_shape.elasticity = 0
        poly_shape.friction = 10
        poly_shape.collision_type = COLLISION_TYPES["robot"]
        space.add(poly_shape)
    space.add(body)

    return body

# https://www.robocupjunior.org.au/sites/default/files/RCJASoccer-Ball%20Specification%202017.pdf
def create_ball(x, y):
    global space
    # we increase the weight a bit so it's less glidey. 
    # TODO this is physically inaccurate, do with friction instead
    mass = 1500 # mass = 0.15kg max (150 grams)
    radius = 7.4 / 2 # ball diameter = 74mm
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
    body = pymunk.Body(mass, inertia)
    body.position = x, y
    shape = pymunk.Circle(body, radius, (0,0))
    shape.color = pygame.color.THECOLORS["black"]
    shape.collision_type = COLLISION_TYPES["ball"]
    space.add(body, shape)
    shape.elasticity = RESTITUTION
    return body

def create_goal(x, y):
    global space
    goal_width = 7.4
    goal_height = 45
    radius = 1
    static = [
        # back wall
        pymunk.Segment(
            space.static_body,
            (x - goal_width, y), (x - goal_width, y + goal_height), radius
        ),

        # top wall
        pymunk.Segment(
            space.static_body,
            (x - goal_width, y + goal_height), (x + goal_width, y + goal_height), radius
        ),

        # bottom wall
        pymunk.Segment(
            space.static_body,
            (x - goal_width, y), (x + goal_width, y), radius
        )
    ]

    for wall in static:
        wall.friction = 1
        wall.elasticity = RESTITUTION
        wall.color = pygame.color.THECOLORS["blue"]

    static[0].collision_type = COLLISION_TYPES["goal_blue"] # this is only temporary, we will have other goals in future
    space.add(static)
    return static[0]

def create_field(x, y):
    global space
    field_width = 243 # 2430 mm
    field_height = 182 # 1820 mm
    radius = 1

    static = [
        # left wall
        pymunk.Segment(
            space.static_body,
            (x, y), (x, y + field_height), radius
        ),
        # top wall
        pymunk.Segment(
            space.static_body,
            (x, y + field_height), (x + field_width, y + field_height), radius
        ),
        # bottom wall
        pymunk.Segment(
            space.static_body,
            (x, y), (x + field_width, y), radius
        ),
        # right wall
        pymunk.Segment(
            space.static_body,
            (x + field_width, y + field_height), (x + field_width, y), radius
        )
    ]

    for wall in static:
        wall.friction = 1
        wall.elasticity = RESTITUTION
        wall.color = pygame.color.THECOLORS["black"]
        wall.collision_type = COLLISION_TYPES["wall"]
    space.add(static)

def reset():
    global space, robot_touched_ball, fitness, total_steps, total_steps_touching_ball
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = 0.6
    space.collision_bias = 0 # add realism, don't allow shapes to overlap. may cause stability issues.
    space.iterations = 28
    robot = create_robot((140 + 5, 91 + 5))
    #robot.angle = math.radians(random.randint(0, 360))
    ball = create_ball(121.5 + 5, 91 + 5 + random.uniform(-20.0, 20.0))
    create_field(5, 5)
    goal = create_goal(5 + 25, 5 + 68.5)
    robot_touched_ball = False
    fitness = 0.0
    total_steps = 0
    total_steps_touching_ball = 0
    
    def ball_goal(arbiter, spacey, data):
        global reset_sim, fitness, total_steps_touching_ball
        fitness += 1000 # good robot
        fitness -= total_steps / 1.5
        fitness += total_steps_touching_ball
        reset_sim = True
        return True

    def robot_wall(arbiter, spacey, data):
        global reset_sim, fitness, total_steps_touching_ball
        fitness -= 1000 # bad robot
        fitness -= total_steps / 1.5
        fitness += total_steps_touching_ball
        reset_sim = True
        return True

    def robot_ball_enter(arbiter, spacey, data):
        global robot_touched_ball, total_steps_touching_ball
        robot_touched_ball = True
        total_steps_touching_ball += 1
        return True

    def robot_ball_leave(arbiter, spacey, data):
        global robot_touched_ball
        robot_touched_ball = False
        return True

    score_handler = space.add_collision_handler(COLLISION_TYPES["ball"], COLLISION_TYPES["goal_blue"])
    score_handler.begin = ball_goal

    wall_handler = space.add_collision_handler(COLLISION_TYPES["robot"], COLLISION_TYPES["wall"])
    wall_handler.begin = robot_wall

    ball_handler = space.add_collision_handler(COLLISION_TYPES["robot"], COLLISION_TYPES["ball"])
    ball_handler.begin = robot_ball_enter
    ball_handler.separate = robot_ball_leave

    return robot, ball, goal

def calculate_net_inputs(robot, ball, goal):
    global fitness, total_steps, MAX_STEPS, reset_sim
    # calculate new net inputs
    rotated_center = Vec2d(10.5, 10.5)
    rotated_center.rotate(robot.angle)
    rotated_center += robot.position
    goal_pos = utils.avg(goal.a, goal.b)

    ball_dist = utils.dist(rotated_center, ball.position) # robot -> ball dist
    goal_dist = utils.dist(rotated_center, goal_pos) # ball -> goal dist
    ball_dir, goal_dir = utils.get_angles(rotated_center, ball.position, goal_pos) # inputs for neural net
    ball_dir = (450 - ball_dir + math.degrees(robot.angle)) % 360
    goal_dir = (450 - goal_dir + math.degrees(robot.angle)) % 360
    fitness = utils.calculate_fitness(ball_dist, goal_dist, robot_touched_ball)

    ball_dist = np.interp(ball_dist, [0, 303.6], [0.0, 1.0])
    goal_dist = np.interp(goal_dist, [0, 303.6], [0.0, 1.0])

    # TODO need to scale this??? :thinking_face:
    ball_i = math.cos(math.radians(ball_dir)) * ball_dist
    ball_j = math.sin(math.radians(ball_dir)) * ball_dist
    goal_i = math.cos(math.radians(goal_dir)) * goal_dist
    goal_j = math.sin(math.radians(goal_dir)) * goal_dist

    return fitness, [ball_i, ball_j, goal_i, goal_j, ball_dist, goal_dist]

# non graphical simulation, used to evaluate neural net from genetic algorithm
def simulate(net, config):
    robot, ball, goal = reset()
    global fitness, total_steps, MAX_STEPS, reset_sim
    for step in range(MAX_STEPS):
        fitness, inputs = calculate_net_inputs(robot, ball, goal)

        # get input from neural net here, need to calculate balldir and goaldir though
        # NEW INPJTS SHOULD BE: fixed ball dir, fixed ball dist, fix goal direction, fixed goal distance
        # need to subtract curent heading from the ball dir
        # goal dir should be robot to goal not bloody ball to goal
        # remove int touched ball
        # something else here?

        # needs to be: [cos(ball_dir) * ball_dist, sin(ball_dir) * ball_dist, cos(goal_dir) * goal_dist, sin(goal_dir) * goal_dist]
        rotation, speed = net.activate(inputs)
        rotation = utils.clamp(rotation, -1.0, 1.0)
        speed = utils.clamp(speed, -1.0, 1.0)
        rotation *= 10 # rotation will be in degrees
        speed *= 50 # max speed = 60

        robot.angle += math.radians(rotation)
        robot.velocity = (speed * math.cos(robot.angle - 1.5708), speed * math.sin(robot.angle - 1.5708))

        # step sim based on input
        robot.angular_velocity = 0
        robot.center_of_gravity = (10.5, 10.5)
        space.step(1.0 / 60.0)

        total_steps += 1

        # session was ended from one of the callback listeners, so we know it's got the bonuses already
        if reset_sim:
            reset_sim = False
            return fitness

    # test failed to complete, still subtract total steps
    fitness -= 1000 # bad robot
    fitness -= total_steps / 1.5
    fitness += total_steps_touching_ball
    return fitness

# graphical simulation. this will show a single net, running in real time.
# TODO let neural net control this instead of player
if __name__ == "__main__":
    print("Loading winner neural net...")

    with open("winner.net", "rb") as f:
        net = pickle.load(f)

    robot, ball, goal = reset()
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Calibri", 24)

    screen = pygame.display.set_mode((1024, 600))
    draw_options = scaledrenderer.DrawOptions(screen)
    clock = pygame.time.Clock()

    running = True
    while running:
        for i in pygame.event.get():
            if i.type == pygame.QUIT or (i.type == pygame.KEYUP and i.key == pygame.K_ESCAPE):
                running = False
                print("Exiting on next frame")
            elif i.type == pygame.KEYUP and i.key == pygame.K_r:
                robot, ball, goal = reset()

        clock.tick(60)
        screen.fill((230, 230, 230))

        fitness, inputs = calculate_net_inputs(robot, ball, goal)

        rotation, speed = net.activate(inputs)
        rotation = utils.clamp(rotation, -1.0, 1.0)
        speed = utils.clamp(speed, -1.0, 1.0)
        rotation *= 10 # rotation will be in degrees
        speed *= 50 # max speed = 60

        robot.angle += math.radians(rotation)
        robot.velocity = (speed * math.cos(robot.angle - 1.5708), speed * math.sin(robot.angle - 1.5708))
        
        # key_pressed = False
        # keys = pygame.key.get_pressed()
        # robot_speed = 50
        # rotate_speed = math.radians(5)
        # if keys[pygame.K_d]:
        #     robot.angle -= rotate_speed
        #     key_pressed = True
        # if keys[pygame.K_a]:
        #     robot.angle += rotate_speed
        #     key_pressed = True
        # if keys[pygame.K_w]:
        #     robot.velocity = (robot_speed * math.cos(robot.angle - 1.5708), robot_speed * math.sin(robot.angle - 1.5708))
        #     key_pressed = True
        # if keys[pygame.K_s]:
        #     robot.velocity = (-robot_speed * math.cos(robot.angle - 1.5708), -robot_speed * math.sin(robot.angle - 1.5708))
        #     key_pressed = True

        # if not key_pressed:
        #     robot.velocity = (0, 0)

        # step sim based on input
        robot.angular_velocity = 0
        robot.center_of_gravity = (10.5, 10.5)
        space.step(1.0 / 60.0)
        space.debug_draw(draw_options)

        #debug = font.render(f"Balldir: {ball_dir} Goaldir: {goal_dir} Balldist: {ball_dist}", False, (0, 0, 0))
        #debug = font.render(f"Outputs: {int(rotation)}", False, (0, 0, 0))
        #screen.blit(debug, (0, 0))

        # session was ended from one of the callback listeners, so we know it's got the bonuses already
        if reset_sim:
            reset_sim = False
            print("Goodbye")
            robot, ball, goal = reset()
        pygame.display.update()
        pygame.display.set_caption("FPS: {}".format(clock.get_fps()))

    pygame.quit()
    pygame.font.quit()
    sys.exit()