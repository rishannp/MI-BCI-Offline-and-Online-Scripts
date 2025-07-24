# game.py

import pygame
import pickle
import os
import queue
import random

from config import SUBJECT_ID, RESULTS_DIR, NUM_LEVELS, TRIALS_PER_LEVEL

# ─── CONSTANTS ─────────────────────────────────────────────────────────
GOAL_SPEED         = 1     # pixels/frame
PLAYER_SPEED       = 1     # pixels/frame
PLAYER_RADIUS      = 15
GOAL_WIDTH         = 50
GOAL_HEIGHT        = 20
FPS                = 60
MIN_SPAWN_DISTANCE = 100   # px away from player
ADAPT_MSG_DURATION_MS = 1000
GREY               = (128,128,128)
GREEN              = (0,255,0)

def run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log):
    """
    action_queue → commands
    adapt_queue  → adaptation events
    game_states  → frame logs
    label_queue  → true labels
    raw_eeg_log  → latest EEG window
    """
    pygame.init()
    WIDTH, HEIGHT = 400, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Lane Runner")
    font = pygame.font.Font(None, 36)

    BG, PC, GC, TC = (31,41,51), (59,130,246), (249,115,22), (229,231,235)
    mid_spawn = (WIDTH - GOAL_WIDTH)//2

    level          = 1
    trial_in_level = 0
    hits           = 0
    misses         = 0
    level_data     = []

    px = WIDTH//2
    py = HEIGHT - PLAYER_RADIUS*2
    last_cmd = None
    last_adapt_ts = -ADAPT_MSG_DURATION_MS

    def make_spawn_sides():
        half = TRIALS_PER_LEVEL//2
        sides = [0]*half + [1]*half
        if TRIALS_PER_LEVEL % 2:
            sides.append(random.choice([0,1]))
        random.shuffle(sides)
        return sides

    def spawn_goal(side):
        if side == 0:
            xmin, xmax = 0, mid_spawn
            dm = px - MIN_SPAWN_DISTANCE - GOAL_WIDTH//2
            if dm >= xmin: xmax = min(xmax, dm)
        else:
            xmin, xmax = mid_spawn+1, WIDTH-GOAL_WIDTH
            dm = px + MIN_SPAWN_DISTANCE - GOAL_WIDTH//2
            if dm <= xmax: xmin = max(xmin, dm)
        if xmin > xmax:
            xmin, xmax = (0,mid_spawn) if side==0 else (mid_spawn+1,WIDTH-GOAL_WIDTH)
        return pygame.Rect(random.randint(xmin,xmax), -GOAL_HEIGHT, GOAL_WIDTH, GOAL_HEIGHT)

    spawn_sides = make_spawn_sides()
    side = spawn_sides[0]
    goal = spawn_goal(side)
    spawn_ts = pygame.time.get_ticks()
    spawn_px = px
    spawn_gx = goal.x
    spawn_label = side
    trial_windows = []
    last_window_id = None

    clock = pygame.time.Clock()
    running = True

    while running and level <= NUM_LEVELS:
        ts = pygame.time.get_ticks()
        screen.fill(BG)

        # — Handle adaptation event —
        try:
            adapt_queue.get_nowait()
            last_adapt_ts = pygame.time.get_ticks()
        except queue.Empty:
            pass

        # — Input —
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        moved = False
        if keys[pygame.K_LEFT] and px-PLAYER_RADIUS>0:
            px -= PLAYER_SPEED; moved = True
        if keys[pygame.K_RIGHT] and px+PLAYER_RADIUS<WIDTH:
            px += PLAYER_SPEED; moved = True

        try:
            cmd = action_queue.get_nowait()
            last_cmd = cmd
        except queue.Empty:
            cmd = last_cmd

        if not moved and cmd is not None:
            if cmd==0 and px-PLAYER_RADIUS>0:
                px -= PLAYER_SPEED
            elif cmd==1 and px+PLAYER_RADIUS<WIDTH:
                px += PLAYER_SPEED

        # — Move goal & capture EEG —
        goal.y += GOAL_SPEED
        if raw_eeg_log:
            window = raw_eeg_log[0]
            if id(window) != last_window_id:
                trial_windows.append(window.copy())
                last_window_id = id(window)

        # — Check end of trial —
        player_rect = pygame.Rect(px-PLAYER_RADIUS, py-PLAYER_RADIUS,
                                  PLAYER_RADIUS*2, PLAYER_RADIUS*2)
        outcome = None
        if player_rect.colliderect(goal):
            outcome = 'hit'; hits += 1
        elif goal.y > HEIGHT:
            outcome = 'miss'; misses += 1

        if outcome:
            trial = {
                'spawn_timestamp':   spawn_ts,
                'spawn_player_x':    spawn_px,
                'spawn_goal_x':      spawn_gx,
                'label':             spawn_label,
                'outcome_timestamp': ts,
                'end_player_x':      px,
                'end_goal_x':        goal.x,
                'outcome':           outcome,
                'eeg_windows':       trial_windows
            }
            level_data.append(trial)
            label_queue.put(spawn_label)

            trial_in_level += 1

            if trial_in_level >= TRIALS_PER_LEVEL:
                level += 1
                trial_in_level = 0
                spawn_sides = make_spawn_sides()

            side = spawn_sides[trial_in_level]
            goal = spawn_goal(side)
            spawn_ts    = pygame.time.get_ticks()
            spawn_px    = px
            spawn_gx    = goal.x
            spawn_label = side
            trial_windows = []
            last_window_id = None

        # — Frame log —
        game_states.append({
            'timestamp': ts,
            'player_pos': (px, py),
            'goal_pos':   (goal.x, goal.y),
            'level':      level,
            'hit':        (outcome=='hit')
        })

        # — Draw —
        pygame.draw.circle(screen, PC, (px, py), PLAYER_RADIUS)
        pygame.draw.rect(screen, GC, goal, border_radius=10)
        screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TC), (10, 10))
        screen.blit(font.render(f"Hits {hits}  Misses {misses}", True, TC), (10, 50))

        # — Adapt indicator —
        now = pygame.time.get_ticks()
        color = GREEN if (now - last_adapt_ts) < ADAPT_MSG_DURATION_MS else GREY
        screen.blit(font.render("Adapting", True, color), (WIDTH-140, 10))

        pygame.display.flip()
        clock.tick(FPS)

    # — Save at end —
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"Subject{SUBJECT_ID}_Session.pkl"), 'wb') as f:
        pickle.dump(level_data, f)

    pygame.quit()
