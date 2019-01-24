int boardX;
int boardY;
int roadW;
int pieceX;
int pieceY;
int[][] map;
int dirX[] = {1,0,-1,0};
int dirY[] = {0,1,0,-1};
int pieceDir;
boolean mode3D;
boolean isSearchLeft;
float pieceSize;

void makeBoard(int x, int y,int w){
   boardX = x+4;
   boardY = y+4;
   roadW = w;
   
   map = new int[boardX][boardY];
}

void initMaze(){
   for(int x=0; x < boardX; x++){
     for(int y=0; y < boardY; y++){
        map[x][y] = 1;
     }
   }
   
   for(int x=3; x < boardX-3; x++){
     for(int y=3; y < boardY-3; y++){
       map[x][y] = 0;
     }
   }
   map[2][3] = 2;
   map[boardX - 3][boardY -4] = 3;
   
   pieceX = 2;
   pieceY = 3;
   pieceSize = 0.7*roadW;
   
   isSearchLeft = false;
   pieceDir = 0;
}

void drawMaze(){
   background(100);
   noStroke();
   
   for(int x = 2; x <= boardX - 3; x++){
     for(int y = 2; y <= boardY -3; y++){
       if(map[x][y] == 0){
          fill(255,183,76);
       }else if(map[x][y] == 1){
          fill(73,6,248);
       }else if(map[x][y] == 2){
          fill(255,241,15);
       }else if(map[x][y] == 3){
          fill(200,0,200);
       }
       rect(roadW*x,roadW*y,roadW,roadW);
        
     }
   }
}

void drawMaze3D(){
   background(100);
   stroke(0);
   
   camera(pieceX*roadW,pieceY*roadW,0 ,(pieceX + dirX[pieceDir])*roadW, 
     (pieceY + dirY[pieceDir])*roadW,0
          ,0,0,-1);
   perspective(radians(100),(float)width/(float)height,1,800);
   
   for(int x = 2; x <= boardX-3; x++){
     for(int y = 2; y <= boardY -3; y++){
      if(map[x][y] == 0){
          fill(255,183,76);
       }else if(map[x][y] == 1){
          fill(73,6,248);
       }else if(map[x][y] == 2){
          fill(255,241,15);
       }else if(map[x][y] == 3){
          fill(200,0,200);
       }
       
       pushMatrix();
       if(map[x][y] == 1){
         translate(x*roadW,y*roadW,0);
         box(roadW);
       }else{
         translate(x*roadW,y*roadW,-roadW/2);
         box(roadW,roadW,1);
       }      
       popMatrix();
     }
   }
}
       
void generateMazeRandom(){
   for(int x=2; x < boardX - 3; x+=2){
     for(int y = 2; y < boardY - 3; y+=2){
       if(map[x][y] == 1){ 
          int r = (int)random(4);
          int dx = dirX[r];
          int dy = dirY[r];
          
          if(map[x+dx*2][y+dy*2] == 0){
            map[x+dx][y+dy] = 1;
            map[x+dx*2][y+dy*2] = 1;
          }
       }
     }
   }
}

void SearchLeft(){
  int dir = 0;
  int x = 0;
  int y = 0;
  
  if(frameCount%10 == 0){
    for(int i = 0; i < 4; i++){
      dir = (pieceDir+3+i)%4;
      x = pieceX + dirX[dir];
      y = pieceY + dirY[dir];
      if(map[x][y] == 0 || map[x][y] == 3){
         break;
      }
    }         
    pieceDir = dir;
    pieceX = x;
    pieceY = y;
  }
}
    
   
            
void keyPressed(){
     if(key == 'r') generateMazeRandom();
     if(key == 's') isSearchLeft = true;
     if(key == 'm') {
       if(mode3D){
          mode3D = false;
       }else{
         mode3D = true;
       }
     }
}

void checkFinish(){
  if(map[pieceX][pieceY]==3){
    isSearchLeft = false;
  }
}


void drawPiece(){
 fill(0,200,0);
 ellipse((pieceX+0.5)*roadW,(pieceY+0.5)*roadW,pieceSize,pieceSize);
}

void setup(){
  size(800,600,P3D);
  makeBoard(13,9,46);
  initMaze();
}

void draw(){
  if(isSearchLeft){
   SearchLeft();
  }
  drawMaze();  
  drawPiece();
  checkFinish();
  if(mode3D){
    drawMaze3D();
  }
  else{
    camera(width/2,height/2,(height/2)/tan(PI/6),width/2,height/2,0,0,1,0);
    perspective(PI/3,(float)width/(float)height,(height/2)/tan(PI/6)/10,(height/2)/tan(PI/6)*10);
    drawMaze();
    drawPiece();
  }
}