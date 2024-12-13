# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:50:05 2024

@author: uceerjp
"""

import pygame
import random

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lane Runner - Aesthetic Mode")

# Color Palette
BACKGROUND_COLOR = (31, 41, 51)  # Dark Gray
PLAYER_COLOR = (59, 130, 246)    # Blue
OBSTACLE_COLOR = (249, 115, 22)   # Orange
TEXT_COLOR = (229, 231, 235)      # Light Gray

# Game settings
error_free_mode = True  # Toggle this for error-free gameplay
player_radius = 15  # Use a circular player shape
player_x = WIDTH // 4  # Start in the left lane
player_y = HEIGHT - player_radius * 2
lanes = [WIDTH // 4, WIDTH * 3 // 4]  # Left and right lane positions

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
    
    # Player movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x = lanes[0]  # Move to the left lane
    if keys[pygame.K_RIGHT]:
        player_x = lanes[1]  # Move to the right lane
    
    # Spawn a single obstacle after interval
    if obstacle is None and obstacle_timer > obstacle_interval:
        obstacle_lane = random.choice(lanes)
        obstacle = pygame.Rect(obstacle_lane, -obstacle_height, obstacle_width, obstacle_height)
        obstacle_timer = 0  # Reset timer

    # Move the obstacle if it exists
    if obstacle:
        obstacle.y += obstacle_speed
        if obstacle.y > HEIGHT:  # Remove obstacle if it goes off-screen
            obstacle = None
            score += 1  # Increase score as reward for dodging
            # Increase difficulty gradually
            if score % 5 == 0:
                level += 1
                obstacle_speed += 1  # Increase speed
                obstacle_interval = max(50, obstacle_interval - 10)  # Increase frequency

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
