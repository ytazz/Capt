#include <expat.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFSIZE 102400

static void XMLCALL elementStart(void *user_data, const XML_Char *el,
                                 const XML_Char *attr[]) {
  printf("[ELEMENT] %s Start!\n", el);
}

static void XMLCALL elementEnd(void *user_data, const XML_Char *el) {
  printf("[ELEMENT] %s End!\n", el);
}

int main(int argc, char *argv[]) {
  char buf[BUFSIZE];
  int done;
  XML_Parser parser;

  if ((parser = XML_ParserCreate(NULL)) == NULL) {
    fprintf(stderr, "Parser Creation Error.\n");
    exit(1);
  }

  XML_SetElementHandler(parser, elementStart, elementEnd);

  FILE *fp = fopen("../data/data.xml", "r");

  do {
    size_t len = fread(buf, sizeof(char), BUFSIZE, fp);
    if (ferror(fp)) {
      fprintf(stderr, "File Error.\n");
      exit(1);
    }

    done = len < sizeof(buf);
    if (XML_Parse(parser, buf, (int)len, done) == XML_STATUS_ERROR) {
      fprintf(stderr, "Parse Error.\n");
      exit(1);
    }
  } while (!done);

  XML_ParserFree(parser);
  return (0);
}
