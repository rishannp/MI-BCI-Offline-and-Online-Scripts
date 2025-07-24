# game.py

import pygame, time
from config import ADAPTATION

def run_game(action_queue, pipeline):
    pygame.init()
    WIDTH, HEIGHT = 400, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.Font(None,36)

    # game vars
    player_x = WIDTH//2; player_y = HEIGHT-30
    pr = 15; spd=5
    bw=50; h=20; os=2
    score=0; level=1; err=0
    obstacle = pygame.Rect(player_x-bw//2, -h, bw, h)
    clock=pygame.time.Clock()
    running=True

    while running:
        screen.fill((31,41,51))
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False

        # get BCI
        try: cmd=action_queue.get_nowait()
        except: cmd=None

        # move
        if cmd==0 and player_x-pr>0:        player_x-=spd
        elif cmd==1 and player_x+pr<WIDTH: player_x+=spd
        else:
            keys=pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and player_x-pr>0:   player_x-=spd
            if keys[pygame.K_RIGHT] and player_x+pr<WIDTH: player_x+=spd

        # obstacle
        obstacle.y+=os
        passed=False
        if obstacle.y>HEIGHT:
            score+=1; passed=True
            if score%5==0:
                level+=1; os+=1
            # respawn
            bw2 = bw+(level-1)*30
            obstacle=pygame.Rect(player_x-bw2//2,-h,bw2,h)

        # collision
        prc = pygame.Rect(player_x-pr,player_y-pr,pr*2,pr*2)
        if obstacle.colliderect(prc):
            err+=1
            bw2=bw+(level-1)*30
            obstacle=pygame.Rect(player_x-bw2//2,-h,bw2,h)

        # draw
        pygame.draw.circle(screen,(20,25,35),(player_x,player_y+5),pr)
        pygame.draw.circle(screen,(59,130,246),(player_x,player_y),pr)
        pygame.draw.rect(screen,(20,25,35),(obstacle.x,obstacle.y+5,obstacle.width,obstacle.height),border_radius=10)
        pygame.draw.rect(screen,(249,115,22),obstacle,border_radius=10)
        screen.blit(font.render(f"Score:{score} Level:{level}",True,(229,231,235)),(10,10))
        screen.blit(font.render(f"Err:{err}",True,(229,231,235)),(10,50))
        pygame.display.flip()
        clock.tick(30)

        # between-level pause + adapt
        if ADAPTATION and passed:
            t0=time.time()
            while time.time()-t0<3:
                screen.fill((0,0,0))
                screen.blit(font.render("Recalibrating...",True,(255,255,255)),(100,HEIGHT//2))
                pygame.display.flip()
            pipeline.adapt()

    pygame.quit()
