

#include<SoftwareSerial.h>
SoftwareSerial Serial1(3,2);

int dist;
int strength;
int check; 
int uart[9];
int i;
const int HEADER=0x59;
int dist_old=0;

void setup(){
  Serial.begin(115200);
  Serial1.begin(115200);
}

void loop(){
  if (Serial1.available()){
    if(Serial1.read()){
      uart[0]=HEADER;
      if(Serial1.read()==HEADER){
        uart[1]=HEADER;
        for(i=2;i<9;i++){
          uart[i]=Serial1.read();
        }
        check=uart[0]+uart[1]+uart[2]+uart[3]+uart[4]+uart[5]+uart[6]+uart[7];
        if(uart[8]==(check&0xff)){
          dist=uart[2]+uart[3]*256;
          strength=uart[4]+uart[5]*256;
          Serial.print("d");
          Serial.print(dist);
          //Serial.print('\t');
          /*Serial.print("strength");
          Serial.print(strength);*/
          int t=100;
          /*if(dist - dist_old>-100 & dist - dist_old<100)
            Serial.print((dist-dist_old)*360/t);
          dist_old = dist;*/
          Serial.print('\n');
          
          delay(t);
        }
      }
    }
  }
}
