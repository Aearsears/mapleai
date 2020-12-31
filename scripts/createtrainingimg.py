# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:25:42 2020


"""

from PIL import Image
import os
import random

def main():
    
    #imgarray = [f for f in os.listdir('../imagesv2') if os.path.isfile(os.path.join('../imagesv2', f))]
    SAVE_DIR='path'
    IMG_DIR='path'
    BG_DIR='path'
    
    NUMBEROFIMG=5000
    
    f = open("path",'a+')
    
    imgarray = [f for f in os.listdir(IMG_DIR) if os.path.isfile(os.path.join(IMG_DIR, f))]
    for i in range(0,7):
        for file in imgarray:
            if "ss" in file:
                del imgarray[imgarray.index(file)]
        #TODO: REMOVE ALL OCCURENCES OF SS, THE NUMBER SHOULD BE 3410
        
    bgarray = [f for f in os.listdir(BG_DIR) if os.path.isfile(os.path.join(BG_DIR, f))]
    
    SPACING = 10

    #program for now goes row by row, so it calculates the X and whether it has reached the limit for the next row
    
    # #paste first mob
    # NEXT_Y += mob_y+SPACING
    # #loop
    # bgmod.paste(mob,(CURR_X,CURR_Y),mob)
    # CURR_X +=mob_x+SPACING
    
    # #next row
    # CURR_Y= NEXT_Y
    # NEXT_Y += mob_y+SPACING
    # #loop
    # bgmod.paste(mob,(CURR_X,CURR_Y),mob)
    # CURR_X +=mob_x+SPACING
    
    #go row by row and paste the images in the photo
    for i in range(0,NUMBEROFIMG+1):
        mobname = random.choice(imgarray)
        bgname = random.choice(bgarray)
        
        mob = Image.open(IMG_DIR+mobname)
        mob = mob.convert("RGBA")
        
        bg = Image.open(BG_DIR+bgname)
        bg = bg.convert("RGBA")
        bgmod = bg.copy()
        
        BG_X_MAX = bg.size[0]
        BG_Y_MAX = bg.size[1]
        
        mob_x = mob.size[0]
        mob_y = mob.size[1]
        #leave 10 pixels between pixels
        
        #start to paste the pictures at (10,10)
        CURR_X = 10
        CURR_Y = 10
        NEXT_Y =10
        filename='bgmod'+str(i)+'.png'
        #LOOP IS FOR THE SAME BACKGROUND PICTURE
        while(CURR_Y+mob_y<BG_Y_MAX):
            #THIS CHECKS THE Y AXIS
            NEXT_Y += mob_y+SPACING 
            #THIS IS FOR THE ROWS
            while(CURR_X+mob_x<BG_X_MAX):
                if(CURR_Y+mob_y>BG_Y_MAX):
                    CURR_X +=mob_x+SPACING
                    mobname = random.choice(imgarray)
                    mob = Image.open(IMG_DIR+mobname)
                    mob = mob.convert("RGBA")
                    mob_x = mob.size[0]
                    mob_y = mob.size[1]
                    continue
                bgmod.paste(mob,(CURR_X,CURR_Y),mob)
                printstring=(SAVE_DIR[3:]+filename+","+getCoord(mob,mobname,CURR_X,CURR_Y)+getType(mobname))
                f.write(printstring+'\n')
                CURR_X +=mob_x+SPACING
                mobname = random.choice(imgarray)
                mob = Image.open(IMG_DIR+mobname)
                mob = mob.convert("RGBA")
                mob_x = mob.size[0]
                mob_y = mob.size[1]
            CURR_X=10
            CURR_Y= NEXT_Y
        bgmod.save(SAVE_DIR+filename)
        #once this img is done, have to start all over again
    
    f.close()
    mob.close()
    bg.close()
    bgmod.close()


def getCoord(image,mobname,x,y):
    if "mob" in mobname:
        xmin=x
        ymin=y
        xmax=x+image.size[0]
        ymax=y+image.size[1]
        return str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","
    elif "player" in mobname:
        xmin=x+27
        ymin=y+20
        xmax=x+68
        ymax=y+87
        return str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","


def getType(imagename):
    if "mob" in imagename:
        return "mob"
    elif "player" in imagename:
        return "player"
    

if __name__=='__main__':
    main()
