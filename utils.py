import math

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

# TODO fitness should get much higher when you approach the robot not just linearly but a lot
# TODO just take goa and ball positions
def calculate_fitness(ball_dist, goal_dist, touching_ball):
    # max distance = 303.6 (that's the diagonal)
    if touching_ball:
        return 303.6 - goal_dist
    else:
        return 303.6 - ball_dist

# returns goal dir and ball dir
def get_angles(robot, ball, goal):
    #rX, rY = robot
    #bX, bY = ball
    #gX, gY = goal
    #ball_dir = math.degrees(math.atan2(bY - rY, bX - rX))
    #ball_dir = math.degrees(math.atan2(rY - bY, rX - bX))
    #goal_dir = math.degrees(math.atan2(gY - rY, gX - rX))
    #return (ball_dir + 360) % 360, (goal_dir + 360) % 360
    robot_ball_gradient = gradient(robot, ball)
    robot_goal_gradient = gradient(robot, goal)

    ball_dir = math.degrees(math.atan(robot_ball_gradient))
    goal_dir = math.degrees(math.atan(robot_goal_gradient))
    return (ball_dir + 360) % 360, (goal_dir + 360) % 360

def dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def gradient(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (y2 - y1) / (x2 - x1)

def avg(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (x1 + x2) / 2, (y1 + y2) / 2