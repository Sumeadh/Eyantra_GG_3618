
int mr_f=27;
int mr_b=26;
int ml_f=25;
int ml_b=33;
int ir_l=32;
int ir_r=34;
int ir_c=35;
int buzz=13;
int red_led=23;
int green_led=22;
int cnt=0;
int s=1;
int node=0;
int c;
int l;
int r;
int z= 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(mr_f,OUTPUT);
  pinMode(mr_b,OUTPUT);
  pinMode(ml_f,OUTPUT);
  pinMode(ml_b,OUTPUT);
  pinMode(ir_l,INPUT);
  pinMode(ir_r,INPUT);
  pinMode(ir_c,INPUT);
  pinMode(buzz,OUTPUT);
  digitalWrite(buzz,HIGH);
  //digitalWrite(buzz,LOW);
  pinMode(red_led,OUTPUT);
  pinMode(green_led,OUTPUT);
  digitalWrite(red_led,HIGH);
  digitalWrite(buzz,LOW);
  delay(1000);
  digitalWrite(buzz,HIGH);
  digitalWrite(red_led,LOW);
}
void forward(){

  analogWrite(ml_f,110);
  analogWrite(ml_b,0);
  analogWrite(mr_f,110);
  analogWrite(mr_b,0);
  Serial.print("Forward\n");
}
void buzzer(){
  digitalWrite(buzz,LOW);
  delay(1000);
  digitalWrite(buzz,HIGH);
  }

void mid_right(){
  analogWrite(mr_f,25);
  analogWrite(ml_b,0);
  analogWrite(ml_f,80);
  analogWrite(mr_b,0);
  Serial.print("mid_right\n\n");
  if (z>=1){z+=1;}
}

void mid_left(){
  analogWrite(mr_f,90);
  analogWrite(ml_b,0);
  analogWrite(ml_f,25);
  analogWrite(mr_b,0);
  Serial.print("mid_left\n\n");
}

void left(){
  analogWrite(mr_f,80);
  analogWrite(ml_b,0);
  analogWrite(ml_f,05);
  analogWrite(mr_b,0);
  Serial.print("Left\n");
}

void right(){
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,50);
  analogWrite(mr_b,0);
  Serial.print("Right\n");
}

void back(){
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  delay(1000);
  Serial.print("stop");
}
void take_right(){
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,110);
  analogWrite(mr_b,0);
  Serial.print("Right\n");
}
void take_left(){
  analogWrite(mr_f,130);
  analogWrite(ml_b,0);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  Serial.print("Right\n");
}  

void nodes(){
  analogWrite(ml_f,0);
  analogWrite(ml_b,0);
  analogWrite(mr_f,0);
  analogWrite(mr_b,0);
  Serial.println("\nHalt\n");

  buzzer();

  if (node==1){
    Serial.print("N1\n");

  analogWrite(mr_f,130);
  analogWrite(ml_b,0);
  analogWrite(ml_f,40);
  analogWrite(mr_b,0);
    delay (300);
  }
  if (node==2){
    Serial.print("N2\n");
    forward();
    delay (400);
  }  
  if (node==3){
    Serial.print("N3\n");
    take_right();
    delay (800);
  }
  if (node==4){
    Serial.print("N4\n");
    take_left();
    delay (1300);
  }
  if (node==5){
    Serial.print("N5\n");
    take_right();
    delay (800);
  }
  if (node==6){
    Serial.print("N6\n");
    take_right();
    delay (750);
  }
  if (node==7){
    Serial.print("N7\n");
    forward();
    delay (200);
  }
  if (node==8){
    Serial.print("N8\n");
    take_right();
    delay (820);
  }
  if (node==9){
    Serial.print("N9\n");
    forward();
    delay (400);
  }

  if (node==10){
    Serial.print("N10\n");
    take_left();
    delay (1400);
  }
  if (node==11){
    Serial.print("N11\n");
    forward();
    z=1;
    delay(200);}}


void end(){
  analogWrite(ml_f,0);
  analogWrite(ml_b,0);
  analogWrite(mr_f,0);
  analogWrite(mr_b,0);
  digitalWrite(red_led,HIGH);
  digitalWrite(buzz,LOW);
  delay(5000);
  digitalWrite(buzz,HIGH);
  digitalWrite(red_led,LOW);
  analogWrite(ml_f,0);
  analogWrite(ml_b,0);
  analogWrite(mr_f,0);
  analogWrite(mr_b,0);
  delay(2000);}

void loop() {

  // put your main code here, to run repeatedly:

  if (z==1150){end();}

  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r);

//else{
 if (c==0){
  if(r==0 && l==0){
      if (cnt==-1){
          left();
          cnt=-1;
      }
      else if (cnt==1){
         right();
         cnt=1; 
      }   
      else if (cnt==0){
         mid_right();
         cnt=0;
      }   
  }
  if(r==1 && l==1){
    if (cnt==-1){
          left();
      }
      else if (cnt==1){
         right();
      } 
  }
}  
else if(c==1){
  
  if(r==0 && l==0){
    mid_left();  
    cnt=0;
  }
  if(r==0  && l==1){
    node+=1;
     nodes();
  }
  
  if (r==1  && l==0){
  node+=1;
  nodes();
  } 
  if(r==1 && l==1){
    node+=1;
     nodes();
  }
}
 if (c==0){
  if(r==1 && l==0){
    left();
    cnt=-1;
  } 
  if(r==0 && l==1){
    right();
    cnt=1;
  } 
 }
//}
}

