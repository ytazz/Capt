/*
   NOTE:
     positive numbers start with 0
     negative numbers start with n-1

   ex)
     +10 (10) -> 01010 (2), 0a (16)
     -10 (10) -> 10110 (2), f6 (16)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// symbol[key] = *
char symbol[] = "0123456789abcdef";

// prepend number
// ex) 12a4 + b -> b12a4
void prepend(char* s, const char s_){
  const int size = (int)strlen(s);
  for(int i = size; i > 0; i--) {
    s[i] = s[i - 1];
  }
  s[0] = s_;
}

// get the number of a symbol
int key(char s){
  return strchr(symbol, s) - symbol;
}

// calculate complement
// (complement) = (bit inversion of original number) + 1
void complement(char* s, int n){
  const int size = (int)strlen(s);

  // inversion
  for(int i = 0; i < size; i++) {
    int id = key(s[i]);
    s[i] = symbol[n - id - 1];
  }

  // add 1
  for(int i = size - 1; i >= 0; i--) {
    int id = key(s[i]);
    if(id == n - 1) {
      s[i] = symbol[0];
    }else{
      s[i] = symbol[id + 1];
      break;
    }
  }
}

// base n number => integer
int a2i(const char* s, int n){
  const int size = (int)strlen(s);
  // const cast
  char c[size];
  strcpy(c, s);

  int coef = 1;
  if(c[0] == symbol[n - 1]) {
    coef = -1;
    complement(c, n);
  }

  int val = 0;
  for(int i = size - 1; i >= 0; i--) {
    int id = key(c[i]);
    val  += coef * id;
    coef *= n;
  }

  return val;
}

// integer => base n number
void i2a(int x, int n, char* s){
  for(int i = 0; i < 1024; i++) {
    s[i] = '\0';
  }

  if(x >= 0) {
    while(x > 0) {
      prepend(s, symbol[( x % n )]);
      x = x / n;
    }
    prepend(s, symbol[0]);
  }else{
    i2a(-x, n, s);
    complement(s, n);
  }
}

int main(int argc, char** argv){
  char s[1024];
  for(int i = 0; i < 1000; i++) {
    int x1 = rand() - RAND_MAX / 2;
    for(int n = 2; n <= 16; n++) {
      i2a(x1, n, s);
      int x2 = a2i(s, n);
      printf("%d %d %s\n", x1, x2, s);
    }
  }
}