int boardX;
int boardY;
int roadW;
int pieceX;
int pieceY;
int[][] map;
int dirX[] = {1,0,-1,0};
int dirY[] = {0,1,0,-1};
int pieceDir;
boolean mode3D = true;
boolean start = true;
boolean isSearchLeft;
boolean isPlaying;
boolean isGoal;
float pieceSize;

//盤の作成
void makeBoard(int x, int y,int w){
   boardX = x+10;
   boardY = y+10;
   roadW = w;
   
   map = new int[boardX][boardY];
}

//盤の初期化
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
   map[2][3] = 2; //start position setting
   map[boardX - 3][boardY -4] = 3;//goal position setting
   
   pieceX = 2;
   pieceY = 3;
   pieceSize = 0.7*roadW;
   
   isSearchLeft = false;
   pieceDir = 0;
   isPlaying = true;
   isGoal = false;
}

    

//盤の表示を行う
void drawMaze(){
   background(100);
   noStroke();
   randomSeed(1);
   
   
       for(int x = 2; x <= boardX-3; x++){
     for(int y = 2; y <= boardY -3; y++){
      if(map[x][y] == 0){
          fill(70,70,70);
       }else if(map[x][y] == 1 ){
         float w = random(255);
         fill(w,w,w);
       }else if(map[x][y] == 2){
          fill(255,255,255);
       }else if(map[x][y] == 3){
          fill(0,0,0);
       }
       rect(roadW*x,roadW*y,roadW,roadW);
        
     }
   }
}

//3D迷路の表示
void drawMaze3D(){
   background(100);
   stroke(0);
   camera(pieceX*roadW,pieceY*roadW,0,(pieceX + dirX[pieceDir])*roadW, 
     (pieceY + dirY[pieceDir])*roadW,0
          ,0,0,-1);
   perspective(radians(100),(float)width/(float)height,1,800);
   
   for(int x = 2; x <= boardX-3; x++){
     for(int y = 2; y <= boardY -3; y++){
      if(map[x][y] == 0){
          fill(70,70,70);
       }else if(map[x][y] == 1 ){
         float w = random(255);
         fill(w,w,w);
       }else if(map[x][y] == 2){
          fill(255,255,255);
       }else if(map[x][y] == 3){
          fill(0,0,0);
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
  
/*   translate(300,300,0);
   box(roadW);
   
   translate(0,200,0);
   box(roadW);
   
   translate(200,200,0);
   box(roadW);
   
   translate(0,50,0);
  box(roadW);
*/
}


/*
void generateMazeL(){
  for(int x=4; x < boardX - 3; x++){
     for(int y = 3; y < boardY - 4; y++){
       map[x][y] = 1;
     }
  }
  
}*/
       
//ランダムに迷路を生成
void generateMazeRandom(){
  randomSeed(1);
  int count;
  do{
    count = 0;
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
       }else{
         count += 1;
       }
     }
    }
  }while(count != 0);
}
/*         if(map[x+dx*2][y+dy*2] == 0){
            map[x+dx][y+dy] = 1;
            map[x+dx*2][y+dy*2] = 1;
*/
//左手法の探索
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

//キー操作
void keyPressed(){
     
     if(key == 't'){
          mode3D = true;
          start = false;
     }
     /*if(key == 'p'){
          generateMazeL();
     }  */        
     
     if(key == 'g'){
        save("./image.png");
     }
     if(key == 'i') {
         mode3D = true;
         isPlaying = true;
         isGoal = false;
         //initMaze();
         pieceX = 2;
         pieceY = 3;
         
     }
     if(key == 'r') generateMazeRandom();
     if(key == 'o') isSearchLeft = true;
     if(key == 'm') {
       if(mode3D){
          mode3D = false;
       }else{
         mode3D = true;
       }
     }
     if(mode3D){
       if(isPlaying){
             if(keyCode == 'W'){ 
                 if(pieceDir == 0 && map [pieceX+1][pieceY] != 1){
                      pieceX++;
                 }else if(pieceDir == 1 && map[pieceX][pieceY+1] != 1){
                      pieceY++;
                 }else if(pieceDir == 2 && map[pieceX-1][pieceY] != 1){
                      pieceX--;
                 }else if(pieceDir == 3 && map[pieceX][pieceY-1] !=1 ){
                      pieceY--;
                 }
             }    
             else if(keyCode == 'D'){
               pieceDir = (pieceDir+1)%4;
             }
             else if(keyCode == 'S'){
                 if(pieceDir == 0 && map [pieceX-1][pieceY] != 1){
                      pieceX--;
                 }else if(pieceDir == 1 && map[pieceX][pieceY-1] != 1){
                      pieceY--;
                 }else if(pieceDir == 2 && map[pieceX+1][pieceY] != 1){
                      pieceX++;
                 }else if(pieceDir == 3 && map[pieceX][pieceY+1] !=1 ){
                      pieceY++;
                 }
             }    
              else if(keyCode == 'A'){
                 pieceDir = (pieceDir+3)%4;
              }
                 
       }
   }
   else if(isPlaying){
     if(keyCode == 'W'){ 
        if(map[pieceX][pieceY-1] != 1){
           pieceY--;
        }
     }
     if(keyCode == 'D'){
       if(map[pieceX+1][pieceY] != 1){
         pieceX++;
       }
     }
     if(keyCode == 'A'){
        if(map[pieceX-1][pieceY] != 1){
          pieceX--;
        }
     }
     if(keyCode == 'S'){
        if(map[pieceX][pieceY+1] != 1){
           pieceY++;
        }
     }
   }
     
}
//終了判定
void checkFinish(){
  if(map[pieceX][pieceY]==3){
    isSearchLeft = false;
    isPlaying = false;
    isGoal = true;
      
      
  }
}

//コマを表示する関数
void drawPiece(){
 fill(0,200,0);
 ellipse((pieceX+0.5)*roadW,(pieceY+0.5)*roadW,pieceSize,pieceSize);
}

//3D表示にするためのセットアップ
void setup(){
  size(800,600,P3D);
  makeBoard(9,5,60);
  initMaze();
  translate(300,300,0);
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
  if(isGoal == true){
          mode3D = false;
          fill(255);
          textAlign(CENTER);
          textSize(64);
          background(0);
          text("C l e a r",width/2,height/2);          
  }
  
  if(start == true){
             mode3D = false;
             fill(255);
             textAlign(CENTER);
             textSize(64);
             background(0);
             text("S t a r t",width/2,height/2);
  }
}
