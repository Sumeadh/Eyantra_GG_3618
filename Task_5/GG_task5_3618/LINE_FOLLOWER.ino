#include<BluetoothSerial.h>
//#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
//#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
//#endif

BluetoothSerial SerialBT;
const char *pin = "0769";
bool connect;
int value;
String paths[5]= {} ;
String inData;
char dolly;
int pots=0;
int dasa=0;
int event=0;
int curvetop=0;
int uu=0;

int mr_f=27;
int mr_b=26;
int ml_f=25;
int ml_b=33;
int ir_l=32;
int ir_r=34;
int ir_c=35;
int ir_le=4;
int ir_re=18;
int buzz=13;
int red_led=23;
int green_led=22;
int cnt=0;
int s=1;
int node=0;

int c;
int l;
int r;
int re;
int le;
int z= 0;
int q =0;
int mission=0;

void setup() {

  Serial.begin(115200);
  SerialBT.begin("GG_3618");
  SerialBT.setPin(pin);

  pinMode(mr_f,OUTPUT);
  pinMode(mr_b,OUTPUT);
  pinMode(ml_f,OUTPUT);
  pinMode(ml_b,OUTPUT);
  pinMode(ir_l,INPUT);
  pinMode(ir_r,INPUT);
  pinMode(ir_le,INPUT);
  pinMode(ir_re,INPUT);
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

  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
}
void forward(){

  analogWrite(ml_f,80);
  analogWrite(ml_b,0);
  analogWrite(mr_f,75);
  analogWrite(mr_b,0);
  Serial.print("Forward\n");
}
void buzzer(){
  digitalWrite(buzz,LOW);
  delay(1000);
  digitalWrite(buzz,HIGH);
  }

void mid_right(){
  analogWrite(mr_f,25+event);
  analogWrite(ml_b,0);
  analogWrite(ml_f,70+event);
  analogWrite(mr_b,0);
  Serial.print("mid_right\n\n");
}

void mid_left(){
  analogWrite(mr_f,100+event);
  analogWrite(ml_b,0);
  analogWrite(ml_f,20+event);
  analogWrite(mr_b,0);
  Serial.print("mid_left\n\n");
}

void left(){
  analogWrite(mr_f,70+event);
  analogWrite(ml_b,0);
  analogWrite(ml_f,20-curvetop+event);
  analogWrite(mr_b,0);
  Serial.print("Left\n");
}

void right(){
  analogWrite(mr_f,30-curvetop+event);
  analogWrite(ml_b,0);
  analogWrite(ml_f,70+event);
  analogWrite(mr_b,0);
  Serial.print("Right\n");
}

void take_right(){ 
   forward();delay(200);
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,100+event);
  analogWrite(mr_b,130+event);
  delay(400);
  while(1){
    Serial.print("tatkerighhttt");
  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r); 
   le=digitalRead(ir_le);
  re=digitalRead(ir_re); 
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,80+event);
  analogWrite(mr_b,30+event);
  //Serial.print("Righttt\n");
  if (c==1 && r==0 && l==0 && le==0 && re==0){Serial.println("pfinished");break;}
}}
void take_left(){
    forward();delay(200);
  analogWrite(mr_f,120+event);
  analogWrite(ml_b,90+event);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  delay(400);
  while(1){
    Serial.print("tatkeleftttt");
  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r);  
  le=digitalRead(ir_le);
  re=digitalRead(ir_re);   
  analogWrite(mr_f,110+event);
  analogWrite(ml_b,70+event);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  //Serial.print("leftttt\n");
  if (c==1 && r==0 && l==0 && le==0 && re==0){ Serial.println("pfinished");break;}
}  }

void u_turn(){
  //Serial.print("UTURN\n");
  analogWrite(mr_f,80);
  analogWrite(ml_b,75);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  delay(1000);
  //Serial.print("delay_END");
  while(1){
    //Serial.print("suthituuu\n");
  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r);  
  le=digitalRead(ir_le);
  re=digitalRead(ir_re);   
  analogWrite(mr_f,80);
  analogWrite(ml_b,60);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);

  if(l==1){
    uu=1;
    Serial.print("1st over\n");}

  if (uu==1 && c==0 && r==1){ uu=0;Serial.println("pfinished");break;}
  //Serial.print("leftttt\n");
  // if (c==0 && re==1){ Serial.println("pfinished");break;}
} 
}   

void stop()
{
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  Serial.print("STOP\n");
}

void nodes(){
  
  stop();delay(10);
  node+=1;

  if ((4<=node)|| (node<=11)|| node==14 || node==19 || node==22){
      Serial.print("CURVETOP");curvetop=20;
  }
  else{curvetop=0;}
  Serial.println(node);
  //Serial.println(inData[node]);
  Serial.print("\n");
  if(inData[node]=='$'){
    node+=1;}
  if(inData[node]=='F'){
    forward();delay(550);}
  //   if(node==8){
  //     analogWrite(mr_f,50);
  // analogWrite(ml_b,0);
  // analogWrite(ml_f,65);
  // analogWrite(mr_b,0);
  // delay(350);
    
  
  
  else if(inData[node]=='L'){take_left();}
  else if(inData[node]=='R'){take_right();}
  else if(inData[node]=='U'){u_turn();}
}

void loop() {

  // put your main code here, to run repeatedly:
  while(SerialBT.available()>0){
    //Serial.write(SerialBT.read());
    char recieved = SerialBT.read();
    //Serial.print(recieved);
    dolly=recieved;
    if (dolly== 'S'){
      recieved='Q';pots+=1;event+=3;stop();digitalWrite(buzz,LOW);
      if(pots==1){event+=3;}
      if(pots==3){event+=3;}
      if(pots==5){event+=3;}
      // if(pots>=2){event+=3;}
      if(pots==1){dasa=2;}
      if(pots<3){delay(1000);}
      else {delay(5000);}
      digitalWrite(buzz,HIGH);
      // fury=1;
      nodes();
    }
    if (mission==0){
    inData += recieved; 
    delay(10);}
    if (recieved == '\n')
    {
      mission=1;
      Serial.println("Arduino Received: ");
      Serial.println(inData);
      if(inData == "+++\n"){ // DON'T forget to add "\n" at the end of the string.
        Serial.println("OK. Press h for help.");
      }
      // forward();
      //SerialBT.end();
      //break; 
    }}

  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r);
  
  re=digitalRead(ir_re);
  le=digitalRead(ir_le);

  // Serial.print("CENTRE=");
  // Serial.println(c);
  // Serial.print("LEFTMID=");
  // Serial.println(l);
  // Serial.print("RIGHTMID=");
  // Serial.println(r);
  // Serial.print("LEFTEND=");
  // Serial.println(L);
  // Serial.print("RIGHTEND=");
  // Serial.println(R);

  // Serial.println("\n");
  // delay(1000);
if (mission==1){
  if (c==0)
  {
    if ((re==1 && le==1)){ //parallel end straight
      forward();
    }
    else if ((r==0 && l==1) ||(re==0 && le==1) ){ 
      right();
      cnt = 1;
    }
    else if ((l==0 && r == 1)||(le==0 && re == 1)){
        left();
        cnt = -1;
    }
    if(re==0 && le == 0 )
    {
      if (cnt == 1)
        right();
      else if (cnt == -1)
        left();
      else if (cnt==0)
        mid_right();  
    } 
  }
  else if(c==1){
    if(r==0 && l==0){
      mid_left();
      cnt=0;
    }
    if((r==1 && re==1)||(l==1 && le==1) || (r==1 && re==1 & l==1 && le==1)){
      nodes();
    }
    else if ((r==0 && l==1) ||(re==0 && le==1) ){ 
      right();
      cnt = 1;
    }
    else if ((l==0 && r == 1)||(le==0 && re == 1)){
        left();
        cnt = -1;
    }

  }}
}


