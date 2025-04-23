import pygame
import random
import copy
import cv2
import numpy as np
import ModulKlasifikasiCitraCNN as mCNN

pygame.init()

#variable game
rank = ["2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "A",
            "J",
            "K",
            "Q",]

        
#OpenCV
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4,720)

class card_data:
    def __init__(self):
        self.contour = []
        self.width, self.height =  0,0
        self.corner_pts = []
        self.center = []
        self.warp = []
        self.rank = []
        self.rank_predict = "Unkown"
        self.rank_diff = 0
        

kernel_dilasi = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]],dtype = np.uint8)

kernel_erosi = np.ones((5, 5), np.uint8)

#warna biru
lower_blue = np.array([90, 50, 0])
upper_blue = np.array([130, 255, 255])

#load model
modelRank = mCNN.LoadModel("bobotRank.h5")

def closing(image, kernel_dilate, kernel_erosion):
    img_dilation = cv2.dilate(image, kernel_dilate, iterations=1) 
    img_erosion = cv2.erode(img_dilation, kernel_erosion, iterations=2)
    return img_erosion



def flatten(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    s = np.sum(pts, axis = 2)
    
    #nyari ujung paling kiri dan paling kanan corner dari kartu
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts,axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    #listing points corner, yang atas apabila vertical, yang bawah horizontal
    if w <= 0.8*h: 
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl
    if w >= 1.2*h: 
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8*h and w < 1.2*h:
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left
        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
    
    maxWidth = 200
    maxHeight = 300
    
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp

def process_card(contour, image):
    card = card_data()
    card.contour = contour

    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    card.corner_pts = pts
    
    x,y,w,h = cv2.boundingRect(contour)
    card.width, card.height = w,h
    
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    card.center = [cent_x, cent_y]
    
    card.warp = flatten(image, pts, w, h)
    
    corner = card.warp[0:84, 0:32]
    corner_zoom = cv2.resize(corner,(0,0), fx=4, fy=4)
    
    white_level = corner_zoom[15,int((32*4)/2)]
    thresh_level = white_level - 30
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(corner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    rank = query_thresh[20:185, 0:128]
    
    rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rank_cnts = sorted(rank_cnts, key=cv2.contourArea,reverse=True)
    
    if len(rank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(rank_cnts[0])
        rank_roi = rank[y1:y1+h1, x1:x1+w1]
        rank_sized = cv2.resize(rank_roi, (70,125), 0, 0) #ukurannya width sama height
        card.rank = rank_sized

    return card


def match_card(card):
    bestname_rank = "Unknown"

    
    if (len(card.rank) != 0) :
        
        card_rank = cv2.cvtColor(card.rank,cv2.COLOR_GRAY2BGR)
        card_rank = cv2.resize(card_rank,(70,125))
        card_rank_normalized = card_rank / 255.0
        card_rank_input = np.expand_dims(card_rank_normalized, axis=0)
        card_rank_input = card_rank_input.astype('float32')
        
        
        possible_rank = modelRank.predict(card_rank_input,verbose=0)
        nR = np.max(np.where(possible_rank== possible_rank.max()))
        bestname_rank = rank[nR]
        best_rank_diff = nR

    # Return the identiy of the card and the quality of the suit and rank match
    return bestname_rank


#Pygame Stuff
one_deck = 4*rank
decks = 4

def calculate_score(hand):
    current_score = 0
    jumlah_ace = hand.count('A')
    for i in range(len(hand)):
        # Ngecek angka di hand apakah ada di rank?
        if hand[i] in rank[0:8]:
            current_score += int(hand[i])
        if hand[i] in rank[9:12]:
            current_score += 10
            
        elif hand[i]  == 'A':
            current_score += 11
    
    if current_score > 21 and jumlah_ace > 0 :
        for i in range(jumlah_ace):
            if current_score > 21 :
                current_score -= 10
    return current_score
            

def deal_card(kartu, deck):
    card = random.randint(0,len(deck))
    kartu.append(deck[card-1])
    # deck.pop(card-1)
    print(kartu)
    return kartu,deck

def draw_card(player, dealer, reveal):
    for i in range(len(player)):
        pygame.draw.rect(screen, (255,255,255) , [200 + 70*i , 100+(5*i), 120,220] , 0)
        screen.blit(font.render(player[i], True, (0,0,0)), (200 + 70*i, 100 + 5*i))
        pygame.draw.rect(screen,(255,0,0),[200 + 70*i, 100+(5*i), 120,220], 5)
    
    for i in range(len(dealer)):
        pygame.draw.rect(screen, (255,255,255) , [200 + 70*i , 350+(5*i), 120,220] , 0)
        if i != 0 or reveal:
            screen.blit(font.render(dealer[i], True, (0,0,0)), (200 + 70*i, 350 + 5*i))
            # screen.blit(font.render(dealer[i], True, (0,0,0)), (75 + 70*i, 635 + 5*i))
        else:
            screen.blit(font.render('???', True, (0,0,0)), (200 + 70*i, 350 + 5*i))
        pygame.draw.rect(screen,(0,0,255),[200 + 70*i, 350+(5*i), 120,220], 5)
        
def draw_scores(player,dealer):
    screen.blit(font.render(f'Player : {player}', True, (0,0,0)), (900, 120))
    if reveal_dealer:
        screen.blit(font.render(f'Dealer : {dealer}', True, (0,0,0)), (900, 400))

def check_gameover(hand_active, dealer, player, result, total, add):
    if not hand_active and dealer >= 17:
        if player > 21:
            result = 1
        elif dealer < player <=21 or dealer > 21:
            result = 2
        elif player < dealer <=21:
            result = 3
        else:
            result = 4
        
        if add == True:
            if result == 1 or result == 3:
                total[1] += 1
            elif result == 2:
                total[0] += 1
            else:
                total[2] +=1
            add = False
#add di false in biar tidak nge add terus
    return result, total, add

def draw_game(act,record,result):
    button_list = []
    score_text = small_font.render(f'Wins : {record[0]}   Loss : {record[1]}   Draw: {record[2]}', True, (255,255,255))
    screen.blit(score_text,(0,0))
    if (len(cards) == 2 and act == False):
        deal = pygame.draw.rect(screen,(255,255,255),[525, 20, 300, 100], 0)
        pygame.draw.rect(screen,(0,0,0),[525, 20, 300, 100], 3)
        deal_text = font.render('Deal', True, (0,0,0))
        screen.blit(deal_text,(625,50))
        button_list.append(deal)
        
    elif (act == True): 
        if (len(cards) == 1) and result == 0:
            hit = pygame.draw.rect(screen,(255,255,255),[300, 600, 300, 100], 0)
            pygame.draw.rect(screen,(0,0,0),[300, 600, 300, 100], 3)
            hit_text = font.render('HIT', True, (0,0,0))
            screen.blit(hit_text,(410,625))
            button_list.append(hit)
        
        stand = pygame.draw.rect(screen,(255,255,255),[700, 600, 300, 100], 0)
        pygame.draw.rect(screen,(0,0,0),[700, 600, 300, 100], 3)
        stand_text = font.render('STAND', True, (0,0,0))
        screen.blit(stand_text,(770,625))
        button_list.append(stand)
        
    elif (len(cards) < 2 or len(cards) > 2 and act == False ):
        alert = small_font.render('Tolong hanya taruh 2 kartu pada camera', True, (255,255,255))
        screen.blit(alert,(640,360))
    
    if result != 0 :
        screen.blit(font.render(results[result], True, (255,255,255)), (640, 25))
        if(len(cards) == 2):
            deal = pygame.draw.rect(screen,(255,255,255),[600, 325, 200, 75], 0)
            pygame.draw.rect(screen,(0,0,0),[600, 325, 200, 75], 3)
            deal_text = small_font.render('New Hand', True, (0,0,0))
            screen.blit(deal_text,(625,350))
            button_list.append(deal)
        elif (len(cards) < 2 or len(cards) > 2):
            announcement = small_font.render('Putto two Karto to restarto', True, (0,0,0))
            screen.blit(announcement,(625,200))
    
    return button_list


WIDTH, HEIGHT= 1280,720

screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('Belake jackeru')
fps = 30
clock = pygame.time.Clock()
font = pygame.font.Font('freesansbold.ttf',44)
small_font =pygame.font.Font('freesansbold.ttf',30)
active = False

records = [0,0,0]
player_score = 0
dealer_score = 0

deal_awal = False
reveal_dealer = False
hand_active = False

player_hand = []
dealer_hand = []
outcome = 0
add_score = False

results = ['', 'You BUSSSSTTTTT', 'You won! ;D', 'Dealer Win :(', 'TIE']

#gameloop
start = True
while start:
    clock.tick(fps)
    
    #OpenCV
    ret, frame = cap.read()
    
    process = copy.deepcopy(frame)
    
    hsv = cv2.cvtColor(process,cv2.COLOR_BGR2HSV)
    
    maskblue = cv2.inRange(hsv,lower_blue,upper_blue)
    foreground_mask = cv2.bitwise_not(maskblue)
    process = cv2.bitwise_and(process,process,mask=foreground_mask)
    
    foreground = closing(process,kernel_dilasi, kernel_dilasi)
    
    #Ngeblur biar threshold bagus
    frameGrayscaled = cv2.cvtColor(process, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frameGrayscaled, (3, 3), 0)
    
    #threholding shape (contour kartu)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    #nyari kartu dengan contour
    cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    if len(cnts) == 0:
        cnts_sort =  []
        cnt_is_card = []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])
        
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < 120000) and (size > 25000)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1
            
    
    if len(cnts_sort) != 0:
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(process_card(cnts_sort[i],frame))
                
                k = k + 1
                
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(frame,temp_cnts, -1, (255,0,0), 2)
    
    imgRGB  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    background = np.rot90(imgRGB)
    background = pygame.surfarray.make_surface(background).convert()
    background = pygame.transform.flip(background, True, False)
    screen.blit( background,(0,0))
    

    
    if deal_awal == True:
        for i in range(2):
            dealer_hand, game_deck = deal_card(dealer_hand,rank)
            cards[i].rank_predict = match_card(cards[i])
            player_hand.append(cards[i].rank_predict)
        print(dealer_hand, player_hand)
        
        deal_awal = False
    
    if active == True:
        player_score = calculate_score(player_hand)
        draw_card(player_hand,dealer_hand,reveal_dealer)
        if reveal_dealer == True:
            dealer_score = calculate_score(dealer_hand)
            if dealer_score < 17:
                dealer_hand, game_deck = deal_card(dealer_hand, game_deck)
        draw_scores(player_score,dealer_score)
    

    #calculate scores
    buttons = draw_game(active,records, outcome)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONUP:
            if not active:
                if buttons[0].collidepoint(event.pos):
                    #Ngeinit semua variable ke default dan game active 
                    game_deck = copy.deepcopy(decks * one_deck)
                    active = True
                    deal_awal = True
                    player_hand = []
                    dealer_hand = []
                    outcome = 0
                    reveal_dealer = False
                    hand_active = True
                    add_score = True
            else:
                if buttons[0].collidepoint(event.pos) and (len (cards) > 1 or len(cards) < 1) and not reveal_dealer:
                    #Check Stand button
                    reveal_dealer = True
                    hand_active = False
                elif buttons[0].collidepoint(event.pos) and player_score < 21 and hand_active and len(cards) == 1:
                    #Check HIT button
                    for i in range(1):
                        cards[i].rank_predict = match_card(cards[i])
                        player_hand.append(cards[i].rank_predict)
                elif buttons[1].collidepoint(event.pos) and not reveal_dealer:
                    #Check Stand button
                    reveal_dealer = True
                    hand_active = False
                elif outcome !=0:
                    if buttons[1].collidepoint(event.pos):
                    #New Hand
                        game_deck = copy.deepcopy(decks * one_deck)
                        active = True
                        deal_awal = True
                        player_hand = []
                        dealer_hand = []
                        outcome = 0
                        reveal_dealer = False
                        hand_active = True
                        add_score = True
                        dealer_score = 0
                        player_score = 0
                        
            
    
    if hand_active and player_score >= 21:
        hand_active = False
        reveal_dealer = True
    
    outcome, records, add_score = check_gameover(hand_active, dealer_score, player_score, outcome, records, add_score)
            
    pygame.display.flip()
    

    
    
    
