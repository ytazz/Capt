#include <stdio.h>

// letter types
//  - NONE     : not a letter. ex) " '
//  - LETTER   : letter.       ex) A B a b
//  - DELIMITER: delimiter.    ex) , . ! tab
enum Type { NONE, LETTER, DELIMITER };

int main(int argc, char **argv)
{
  // initialize variables
  enum Type type  = NONE; // current letter type
  enum Type type_ = NONE; // previous letter type
  int       words = 0;    // the number of word
  char      buf   = '\0'; // current letter

  // open file
  FILE *fp = fopen(argv[1], "r");
  if ( fp == NULL) {
    printf("Error: Couldn't open the file \"%s\"\n", argv[1] );
    return -1;
  }

  // read file & count words
  while ( ( buf = fgetc(fp) ) != EOF ) { // store one letter in the variable "buf"
    if( buf == ' ' || buf == '\n' || buf == '\t' ||
        buf == ',' || buf == '.' || buf == '!' || buf == '?' ) {
      type = DELIMITER;
    }else if (buf == '\"' || buf == '\'') {
      type = NONE;
    }else{
      type = LETTER;
    }

    // words always end with a delimiter
    // therefore, count the number of delimiters following a word
    if(type_ == LETTER && type == DELIMITER) {
      words++;
    }

    // store current type to use next processing
    type_ = type;
  }

  // close file
  fclose(fp);

  // print the number of words
  printf("%d words\n", words);

  return 0;
}