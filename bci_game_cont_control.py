import pygame
import random

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lane Runner - Continuous Control Mode")

# Color Palette
BACKGROUND_COLOR = (31, 41, 51)  # Dark Gray
PLAYER_COLOR = (59, 130, 246)    # Blue
OBSTACLE_COLOR = (249, 115, 22)   # Orange
TEXT_COLOR = (229, 231, 235)      # Light Gray

# Game settings
error_free_mode = True  # Toggle this for error-free gameplay
player_radius = 15  # Use a circular player shape
player_x = WIDTH // 2  # Start in the center
player_y = HEIGHT - player_radius * 2
player_speed = 5  # Speed for incremental left-right movement

# Obstacle settings
obstacle_width, obstacle_height = 50, 20  # Rounded rectangle style
obstacle_speed = 2  # Start slow for training
obstacle_interval = 150  # Interval between obstacles

# Game variables
clock = pygame.time.Clock()
score = 0
level = 1
error_count = 0  # Tracks collisions/errors when in error-free mode
font = pygame.font.Font(None, 36)
obstacle = None  # Initialize obstacle as None

# Main game loop
running = True
obstacle_timer = 0  # Tracks time since last obstacle

while running:
    screen.fill(BACKGROUND_COLOR)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Continuous player movement control
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x - player_speed - player_radius > 0:
        player_x -= player_speed  # Move left incrementally
    if keys[pygame.K_RIGHT] and player_x + player_speed + player_radius < WIDTH:
        player_x += player_speed  # Move right incrementally
    
    # Spawn obstacle above player after interval or when previous obstacle is off-screen
    if obstacle is None or obstacle.y > HEIGHT:
        obstacle = pygame.Rect(player_x - obstacle_width // 2, -obstacle_height, obstacle_width, obstacle_height)
        obstacle_timer = 0  # Reset timer
        score += 1  # Increase score as reward for dodging

        # Gradually increase difficulty
        if score % 5 == 0:
            level += 1
            obstacle_speed += 1  # Increase speed
            obstacle_interval = max(50, obstacle_interval - 10)  # Increase frequency
    
    # Move the obstacle if it exists
    if obstacle:
        obstacle.y += obstacle_speed

    # Check for collisions
    player_rect = pygame.Rect(player_x - player_radius, player_y - player_radius, player_radius * 2, player_radius * 2)
    if obstacle and player_rect.colliderect(obstacle):
        if error_free_mode:
            error_count += 1  # Track errors instead of ending the game
            obstacle = None  # Remove obstacle to continue gameplay
        else:
            print(f"Game Over! Your Score: {score}")
            running = False

    # Draw player as a circle with shadow effect
    pygame.draw.circle(screen, (20, 25, 35), (player_x, player_y + 5), player_radius)  # Shadow
    pygame.draw.circle(screen, PLAYER_COLOR, (player_x, player_y), player_radius)

    # Draw obstacle with rounded rectangle shape and shadow
    if obstacle:
        pygame.draw.rect(screen, (20, 25, 35), (obstacle.x, obstacle.y + 5, obstacle_width, obstacle_height), border_radius=10)  # Shadow
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle, border_radius=10)

    # Display score, level, and errors
    score_text = font.render(f"Score: {score}  Level: {level}", True, TEXT_COLOR)
    screen.blit(score_text, (10, 10))
    
    if error_free_mode:
        error_text = font.render(f"Errors: {error_count}", True, TEXT_COLOR)
        screen.blit(error_text, (10, 50))  # Display error count in error-free mode

    # Update display and timer
    pygame.display.flip()
    clock.tick(30)
    obstacle_timer += 1  # Increase timer for obstacle interval tracking

pygame.quit()
