import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions and colors
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game - Human Mode")

# Game Objects (Ball and Paddles)
ball = pygame.Rect(WIDTH//2 - 15, HEIGHT//2 - 15, 30, 30)
player = pygame.Rect(WIDTH - 20, HEIGHT//2 - 70, 10, 140)
opponent = pygame.Rect(10, HEIGHT//2 - 70, 10, 140)

# Speed control
ball_speed_x = 7
ball_speed_y = 7
player_speed = 0
opponent_speed = 7

def ball_animation():
    """ Handles ball movement, wall collisions, and scoring. """
    global ball_speed_x, ball_speed_y
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Wall collisions (Top and Bottom)
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
    
    # Scoring: Check if ball goes off screen (Left or Right)
    if ball.left <= 0 or ball.right >= WIDTH:
        print("Point scored! Resetting ball...")
        ball_restart()

    # Paddle collisions
    if ball.colliderect(player) or ball.colliderect(opponent):
        ball_speed_x *= -1

def ball_restart():
    """ Resets the ball to the center of the screen. """
    global ball_speed_x, ball_speed_y
    ball.center = (WIDTH//2, HEIGHT//2)
    ball_speed_x *= -1

# Main Game Loop
print("Game Started. Use UP and DOWN arrows to move.")
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Closing the game...")
            pygame.quit()
            sys.exit()
        
        # Human Input: Keyboard arrows (Player control)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed += 7
            if event.key == pygame.K_UP:
                player_speed -= 7
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 7
            if event.key == pygame.K_UP:
                player_speed += 7

    # Update logic
    ball_animation()
    player.y += player_speed
    
    # Simple Opponent logic (Moves towards the ball's Y position)
    if opponent.top < ball.y: 
        opponent.top += opponent_speed
    if opponent.bottom > ball.y: 
        opponent.bottom -= opponent_speed

    # Drawing the frames (High-dimensional sensory input)
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player)
    pygame.draw.rect(screen, WHITE, opponent)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT))

    # Refresh the display
    pygame.display.flip()
    pygame.time.Clock().tick(60)