/*
 * Team Id: GG_3618
 * Author List: 
 * Filename: task-6_1.ino
 * Theme: Geoguide
 * Functions: forward(), left(),right(), mid_left(), mid_right(), take_left, take_right(), stop(), nodes(), buzzer()
 */ 
 
#include<BluetoothSerial.h>

BluetoothSerial SerialBT;
const char *pin = "0769";
bool connect;
int value;
String paths[5]= {} ;
String inData;
char next_action;

int node = 0;


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

int c;
int l;
int r;
int re;
int le;

int received_path=0;
char action;

void forward(){

  analogWrite(ml_f,150);
  analogWrite(ml_b,0);
  analogWrite(mr_f,150);
  analogWrite(mr_b,0);
  Serial.println("Forward");
}

char left(){
  analogWrite(mr_f,155);
  analogWrite(ml_b,0);
  analogWrite(ml_f,15);
  analogWrite(mr_b,0);
  return 'L';
}

char right(){
  analogWrite(mr_f,15);
  analogWrite(ml_b,0);
  analogWrite(ml_f,155);
  analogWrite(mr_b,0);
  return 'R';
}

char mid_left(){
  analogWrite(mr_f,150);
  analogWrite(ml_b,0);
  analogWrite(ml_f,20);
  analogWrite(mr_b,0);
  return 'Z';
}

char mid_right(){
  analogWrite(mr_f,20);
  analogWrite(ml_b,0);
  analogWrite(ml_f,150);
  analogWrite(mr_b,0);
  return 'C';
}

void take_right(){ 
   forward();delay(150);
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,170);
  analogWrite(mr_b,40);
  delay(270);
  while(1){
    Serial.print("takeright");
  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r); 
   le=digitalRead(ir_le);
  re=digitalRead(ir_re); 
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,150);
  analogWrite(mr_b,40);
  //Serial.print("Righttt\n");
  if (c==1 && r==0 && l==0 && le==0 && re==0){
    Serial.println("Turn finished");
    break;}
}}
void take_left(){
    forward();delay(150);
  analogWrite(mr_f,170);
  analogWrite(ml_b,45);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  delay(250);
  while(1){
    Serial.print("takeleft");
  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r);  
  le=digitalRead(ir_le);
  re=digitalRead(ir_re);   
  analogWrite(mr_f,160);
  analogWrite(ml_b,40);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  //Serial.print("leftttt\n");
  if (c==1 && r==0 && l==0 && le==0 && re==0){ 
    Serial.println("Turn finished");
    break;}
}  }
void stop()
{
  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
  Serial.print("STOP\n");
}
void nodes(){
  stop();delay(200);
  l=digitalRead(ir_l);
  Serial.print(inData[node]);
  Serial.print("\n");
  if(inData[node]=='$'){
    node+=1;}
  if(inData[node]=='F'){
    forward();delay(300);}

  if(inData[node]=='T')
  { 
    while(1)
    {
      c=digitalRead(ir_c);
      l=digitalRead(ir_l);
      r=digitalRead(ir_r);
      re=digitalRead(ir_re);
      le=digitalRead(ir_le);
      action = left();
      Serial.println("LeftExtreme");
      if (r==1 || le == 1)
      {
        break;
      }
    }
  }
  else if(inData[node]=='L'){take_left();}
  else if(inData[node]=='R'){take_right();}
  node =node+1;
}
void buzzer(){
  digitalWrite(buzz,LOW);
  delay(1000);
  digitalWrite(buzz,HIGH);
  }

void setup() {
  // put your setup code here, to run once:
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
  digitalWrite(buzz,HIGH);
  digitalWrite(red_led,LOW);

  analogWrite(mr_f,0);
  analogWrite(ml_b,0);
  analogWrite(ml_f,0);
  analogWrite(mr_b,0);
}


void loop() {
  // put your main code here, to run repeatedly:
    while(SerialBT.available()>0){
    char recieved = SerialBT.read();
    next_action=recieved;
    if (next_action== 'S'){
      recieved='Z';
      stop();
      digitalWrite(buzz,LOW);
      delay(1000);
      digitalWrite(buzz,HIGH);
      nodes();
    }
    else if (next_action == 'Q')
    {
      recieved = 'X';
      stop();
      digitalWrite(buzz,LOW);
      delay(5000);
      digitalWrite(buzz,HIGH);
      delay(30000);
    }
    if (received_path==0){
    inData += recieved; 
    delay(10);}
    if (recieved == '\n')
    {
      received_path=1;
      delay(3000);
    }
    }
  c=digitalRead(ir_c);
  l=digitalRead(ir_l);
  r=digitalRead(ir_r);
  re=digitalRead(ir_re);
  le=digitalRead(ir_le);
  
  if (received_path == 1){
  if((r==1 && re==1)||(l==1 && le==1)){
      nodes();
  }
  if (l == 1)
  // { while( r == 0 || le == 0)
   {while(1) 
    {
      c=digitalRead(ir_c);
      l=digitalRead(ir_l);
      r=digitalRead(ir_r);
      re=digitalRead(ir_re);
      le=digitalRead(ir_le);
      action = mid_left();
      Serial.println("Left");
      if (r==1 || le == 1)
      {
        break;
      }
    }
  }
  if (r == 1)
  { //while( l == 0 || re == 0)
    while(1)
    {
      c=digitalRead(ir_c);
      l=digitalRead(ir_l);
      r=digitalRead(ir_r);
      re=digitalRead(ir_re);
      le=digitalRead(ir_le);
      action = mid_right();
      Serial.println("Right");
      if (l==1 || re == 1)
      {
        break;
      }
    }
  }
  if (re == 1)
  // { while( r == 0 || le == 0)
  {while(1)
    {
      c=digitalRead(ir_c);
      l=digitalRead(ir_l);
      r=digitalRead(ir_r);
      re=digitalRead(ir_re);
      le=digitalRead(ir_le);
      action = left();
      Serial.println("leftExtreme");
      if (r==1 || le == 1)
      {
        break;
      }
    }
  }
  if (le == 1)
  // { while( re == 0 || l == 0)
  {while(1)
    {
      c=digitalRead(ir_c);
      l=digitalRead(ir_l);
      r=digitalRead(ir_r);
      re=digitalRead(ir_re);
      le=digitalRead(ir_le);
      action = right();
      Serial.println("righttExtreme");
      if (l==1 || re == 1)
      {
        break;
      }
    }
  }
  if ((r== 0 && l ==0 && c ==1)||(re == 1 && le == 1))
  {
    forward();
  }
  
  }
}



