# game.py

import os, pickle, random, pygame, queue
from config import SUBJECT_DIR, SESSION_DIR, NUM_LEVELS, TRIALS_PER_LEVEL

# CONSTANTS
GOAL_SPEED, PLAYER_SPEED = 1,1
PLAYER_RADIUS, GOAL_WIDTH, GOAL_HEIGHT = 15,50,20
FPS, MIN_SPAWN_DISTANCE = 60, 100
GREY, GREEN = (128,128,128), (0,255,0)

def run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log):
    pygame.init()
    screen = pygame.display.set_mode((400,600))
    pygame.display.set_caption("Lane Runner")
    font = pygame.font.Font(None,36)

    BG, PC, GC, TC = (31,41,51),(59,130,246),(249,115,22),(229,231,235)
    mid = (400 - GOAL_WIDTH)//2

    level=1; trial_in=0; hits=0; misses=0; data=[]
    px,py=200,600-PLAYER_RADIUS*2
    last_cmd=None; last_adapt=0; adapt_dur=0

    def spawn_list(): 
        s=[0]*(TRIALS_PER_LEVEL//2)+[1]*(TRIALS_PER_LEVEL//2)
        if TRIALS_PER_LEVEL%2: s.append(random.choice([0,1]))
        random.shuffle(s); return s

    spawns=spawn_list(); side=spawns[0]
    goal=pygame.Rect(* (random.randint(0,mid) if side==0 else random.randint(mid+1,400-GOAL_WIDTH),
                        -GOAL_HEIGHT), GOAL_WIDTH, GOAL_HEIGHT)
    spawn_ts, spawn_px, spawn_gx = 0,px,goal.x
    trial_wins=[]; last_win=None
    clock=pygame.time.Clock(); run=True

    while run and level<=NUM_LEVELS:
        ts=pygame.time.get_ticks(); screen.fill(BG)

        try:
            d=adapt_queue.get_nowait(); last_adapt=ts; adapt_dur=d
        except queue.Empty: pass

        for e in pygame.event.get():
            if e.type==pygame.QUIT: run=False

        keys=pygame.key.get_pressed(); moved=False
        if keys[pygame.K_LEFT] and px-PLAYER_RADIUS>0: px-=PLAYER_SPEED; moved=True
        if keys[pygame.K_RIGHT] and px+PLAYER_RADIUS<400: px+=PLAYER_SPEED; moved=True

        try: cmd=action_queue.get_nowait(); last_cmd=cmd
        except queue.Empty: cmd=last_cmd

        if not moved and cmd is not None:
            if cmd==0 and px-PLAYER_RADIUS>0: px-=PLAYER_SPEED
            if cmd==1 and px+PLAYER_RADIUS<400: px+=PLAYER_SPEED

        goal.y+=GOAL_SPEED
        if raw_eeg_log:
            w=raw_eeg_log[0]
            if id(w)!=last_win: trial_wins.append(w.copy()); last_win=id(w)

        player_rect=pygame.Rect(px-PLAYER_RADIUS,py-PLAYER_RADIUS,PLAYER_RADIUS*2,PLAYER_RADIUS*2)
        outcome=None
        if player_rect.colliderect(goal): outcome='hit'; hits+=1
        elif goal.y>600: outcome='miss'; misses+=1

        if outcome:
            label_queue.put((side, trial_wins.copy()))
            data.append({
                'spawn_ts':spawn_ts,'spawn_px':spawn_px,'spawn_gx':spawn_gx,
                'label':side,'outcome_ts':ts,'end_px':px,'end_gx':goal.x,
                'outcome':outcome,'eeg_wins':trial_wins.copy()
            })
            trial_in+=1
            if trial_in>=TRIALS_PER_LEVEL:
                trial_in=0; level+=1; spawns=spawn_list()
            side=spawns[trial_in]
            goal=pygame.Rect(*(random.randint(0,mid) if side==0 else random.randint(mid+1,400-GOAL_WIDTH),
                              -GOAL_HEIGHT),GOAL_WIDTH,GOAL_HEIGHT)
            spawn_ts=ts; spawn_px,spawn_gx=px,goal.x
            trial_wins=[]; last_win=None

        game_states.append({'ts':ts,'pp':(px,py),'gp':(goal.x,goal.y),'lvl':level,'hit':outcome=='hit'})

        pygame.draw.circle(screen,PC,(px,py),PLAYER_RADIUS)
        pygame.draw.rect(screen,GC,goal,border_radius=10)
        screen.blit(font.render(f"Level {level}/{NUM_LEVELS}",True,TC),(10,10))
        screen.blit(font.render(f"Hits {hits}  Misses {misses}",True,TC),(10,50))

        color=GREEN if (ts-last_adapt)<adapt_dur else GREY
        screen.blit(font.render("Adapting",True,color),(260,10))

        pygame.display.flip(); clock.tick(FPS)

    # save everything
    os.makedirs(SESSION_DIR,exist_ok=True)
    with open(os.path.join(SESSION_DIR,"session_data.pkl"),"wb") as f:
        pickle.dump(data,f)

    pygame.quit()
