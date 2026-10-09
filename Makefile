CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDLIBS = -lm
OBJECTS = symnmf.o matrix.o input.o

.PHONY: all clean

all: symnmf

symnmf: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

symnmf.o: symnmf.c symnmf.h input.h
matrix.o: matrix.c symnmf.h
input.o: input.c input.h symnmf.h

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) symnmf
