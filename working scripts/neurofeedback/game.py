import os, pickle, random, pygame, queue
from config import SESSION_DIR

WIDTH, HEIGHT = 1280, 720
PLAYER_RADIUS = 50
GOAL_WIDTH, GOAL_HEIGHT = 150, 30
PLAYER_SPEED, GOAL_SPEED = 3, 1
FPS = 60

def run_game(action_queue, label_queue, raw_eeg_log):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("EEG Manual Game")
    font = pygame.font.Font(None, 48)

    px = WIDTH // 2
    py = HEIGHT // 2  # halfway up screen
    goal = pygame.Rect(random.randint(0, WIDTH - GOAL_WIDTH), -GOAL_HEIGHT, GOAL_WIDTH, GOAL_HEIGHT)
    clock = pygame.time.Clock()
    run = True
    hits = misses = 0
    data = []

    trial_wins = []
    last_win = None
    spawn_ts = pygame.time.get_ticks()
    spawn_px, spawn_gx = px, goal.x
    side = random.choice([0,1])  # left/right pseudo-label

    while run:
        ts = pygame.time.get_ticks()
        screen.fill((30,30,30))

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] and px - PLAYER_RADIUS > 0:
            px -= PLAYER_SPEED
        if keys[pygame.K_l] and px + PLAYER_RADIUS < WIDTH:
            px += PLAYER_SPEED

        goal.y += GOAL_SPEED

        if raw_eeg_log:
            w = raw_eeg_log[0]
            if id(w) != last_win:
                trial_wins.append(w.copy())
                last_win = id(w)

        player_rect = pygame.Rect(px - PLAYER_RADIUS, py - PLAYER_RADIUS, PLAYER_RADIUS*2, PLAYER_RADIUS*2)
        outcome = None
        if player_rect.colliderect(goal):
            outcome = 'hit'
            hits += 1
        elif goal.y > py + PLAYER_RADIUS:  # GOAL has dropped *past* the player
            outcome = 'miss'
            misses += 1

        if outcome:
            label_queue.put((side, trial_wins.copy()))
            data.append({
                'spawn_ts': spawn_ts, 'spawn_px': spawn_px, 'spawn_gx': spawn_gx,
                'label': side, 'outcome_ts': ts, 'end_px': px, 'end_gx': goal.x,
                'outcome': outcome, 'eeg_wins': trial_wins.copy()
            })
            trial_wins = []
            last_win = None
            goal = pygame.Rect(random.randint(0, WIDTH - GOAL_WIDTH), -GOAL_HEIGHT, GOAL_WIDTH, GOAL_HEIGHT)
            spawn_ts = ts
            spawn_px, spawn_gx = px, goal.x
            side = random.choice([0,1])

        pygame.draw.circle(screen, (59,130,246), (px, py), PLAYER_RADIUS)
        pygame.draw.rect(screen, (255,100,100), goal, border_radius=10)
        screen.blit(font.render(f"Hits: {hits}  Misses: {misses}", True, (255,255,255)), (20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(os.path.join(SESSION_DIR, "manual_game_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    pygame.quit()
