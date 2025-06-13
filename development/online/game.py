import pygame
import random


def run_game(action_queue):
    """
    Launch the Lane Runner game with continuous BCI and keyboard control.
    """
    pygame.init()

    WIDTH, HEIGHT = 400, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Lane Runner - Continuous Control Mode")

    # Colors & font
    BACKGROUND_COLOR = (31, 41, 51)
    PLAYER_COLOR     = (59, 130, 246)
    OBSTACLE_COLOR   = (249, 115, 22)
    TEXT_COLOR       = (229, 231, 235)
    font = pygame.font.Font(None, 36)

    # Settings
    error_free_mode   = True
    player_radius     = 15
    player_speed      = 5
    base_obstacle_w   = 50
    obstacle_height   = 20
    obstacle_speed    = 2

    # Initial positions & counters
    player_x = WIDTH // 2
    player_y = HEIGHT - player_radius * 2
    score = 0
    level = 1
    error_count = 0

    # Spawn first obstacle
    obstacle_width = base_obstacle_w
    obstacle = pygame.Rect(
        player_x - obstacle_width // 2,
        -obstacle_height,
        obstacle_width,
        obstacle_height
    )

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(BACKGROUND_COLOR)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get BCI command if any
        cmd = None
        try:
            cmd = action_queue.get_nowait()
        except Exception:
            cmd = None

        # Movement control
        if cmd is not None:
            if cmd == 0 and player_x - player_speed - player_radius > 0:
                player_x -= player_speed
            elif cmd == 1 and player_x + player_speed + player_radius < WIDTH:
                player_x += player_speed
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and player_x - player_speed - player_radius > 0:
                player_x -= player_speed
            if keys[pygame.K_RIGHT] and player_x + player_speed + player_radius < WIDTH:
                player_x += player_speed

        # Move obstacle
        obstacle.y += obstacle_speed

        # Check for passing
        if obstacle.y > HEIGHT:
            score += 1
            # increase difficulty on milestones
            if score % 5 == 0:
                level += 1
                obstacle_speed += 1
            # respawn
            obstacle_width = base_obstacle_w + (level - 1) * 30
            obstacle = pygame.Rect(
                player_x - obstacle_width // 2,
                -obstacle_height,
                obstacle_width,
                obstacle_height
            )

        # Collision detection
        player_rect = pygame.Rect(
            player_x - player_radius,
            player_y - player_radius,
            player_radius * 2,
            player_radius * 2
        )

        if obstacle.colliderect(player_rect):
            if error_free_mode:
                error_count += 1
                # immediately respawn without score increment
                obstacle_width = base_obstacle_w + (level - 1) * 30
                obstacle = pygame.Rect(
                    player_x - obstacle_width // 2,
                    -obstacle_height,
                    obstacle_width,
                    obstacle_height
                )
            else:
                print(f"Game Over! Your Score: {score}")
                running = False

        # Draw player shadow + circle
        pygame.draw.circle(screen, (20, 25, 35), (player_x, player_y + 5), player_radius)
        pygame.draw.circle(screen, PLAYER_COLOR, (player_x, player_y), player_radius)

        # Draw obstacle shadow + rect
        pygame.draw.rect(screen, (20, 25, 35), (obstacle.x, obstacle.y + 5, obstacle.width, obstacle.height), border_radius=10)
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle, border_radius=10)

        # HUD
        screen.blit(font.render(f"Score: {score}  Level: {level}", True, TEXT_COLOR), (10, 10))
        if error_free_mode:
            screen.blit(font.render(f"Errors: {error_count}", True, TEXT_COLOR), (10, 50))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
