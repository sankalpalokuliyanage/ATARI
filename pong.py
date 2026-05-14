import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions and colors
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game - ATARI GAMES")

# Fonts for Score and UI
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)

# Game Objects
ball = pygame.Rect(WIDTH//2 - 15, HEIGHT//2 - 15, 25, 25)
player = pygame.Rect(WIDTH - 20, HEIGHT//2 - 70, 10, 120)
opponent = pygame.Rect(10, HEIGHT//2 - 70, 10, 120)

# Game Variables
ball_speed_x = 7
ball_speed_y = 7
player_speed = 0
opponent_speed = 7
player_score = 0
opponent_score = 0
game_active = False

def ball_animation():
    """ Handles complex movement, scoring, and speed scaling. """
    global ball_speed_x, ball_speed_y, player_score, opponent_score
    
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Wall collisions
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1

    # Scoring logic
    if ball.left <= 0:
        player_score += 1
        print(f"Player Scores! Current Score - Player: {player_score} Opponent: {opponent_score}")
        ball_restart()
    
    if ball.right >= WIDTH:
        opponent_score += 1
        print(f"Opponent Scores! Current Score - Player: {player_score} Opponent: {opponent_score}")
        ball_restart()

    # Collision with Paddles (with slight speed increase to make it harder)
    if ball.colliderect(player) or ball.colliderect(opponent):
        ball_speed_x *= -1.05 
        ball_speed_y *= 1.05

def ball_restart():
    """ Resets ball and speed for the next round. """
    global ball_speed_x, ball_speed_y
    ball.center = (WIDTH//2, HEIGHT//2)
    ball_speed_x = 7 * (1 if ball_speed_x < 0 else -1)
    ball_speed_y = 7

def draw_ui():
    """ Draws the scores and the center line. """
    player_text = font.render(str(player_score), True, WHITE)
    screen.blit(player_text, (WIDTH*3//4, 20))
    
    opponent_text = font.render(str(opponent_score), True, WHITE)
    screen.blit(opponent_text, (WIDTH//4, 20))
    
    pygame.draw.aaline(screen, GRAY, (WIDTH//2, 0), (WIDTH//2, HEIGHT))

def show_menu():
    """ Displays the start menu. """
    title_text = font.render("PONG GAME", True, WHITE)
    start_text = small_font.render("Press SPACE to Start", True, WHITE)
    screen.blit(title_text, (WIDTH//2 - 150, HEIGHT//2 - 100))
    screen.blit(start_text, (WIDTH//2 - 120, HEIGHT//2 + 20))

# Main Game Loop
print("Application running. Waiting for user to start...")
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("System Exit Requested. Closing...")
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_active:
                game_active = True
                print("Game Session Started.")
            if event.key == pygame.K_DOWN:
                player_speed += 7
            if event.key == pygame.K_UP:
                player_speed -= 7
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 7
            if event.key == pygame.K_UP:
                player_speed += 7

    screen.fill(BLACK)

    if game_active:
        # Game Logic
        ball_animation()
        player.y += player_speed
        
        # Keep paddle on screen
        player.y = max(0, min(player.y, HEIGHT - player.height))
        
        # Intelligent Opponent
        if opponent.centery < ball.y: opponent.y += opponent_speed
        if opponent.centery > ball.y: opponent.y -= opponent_speed
        opponent.y = max(0, min(opponent.y, HEIGHT - opponent.height))

        # Rendering
        pygame.draw.rect(screen, WHITE, player)
        pygame.draw.rect(screen, WHITE, opponent)
        pygame.draw.ellipse(screen, WHITE, ball)
        draw_ui()
    else:
        show_menu()

    pygame.display.flip()
    clock.tick(60)