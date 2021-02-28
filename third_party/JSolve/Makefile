all:	jsolve jsolve32

jsolve:	JSolveMain.o JSolve.o Makefile
	gcc -m64 JSolveMain.o JSolve.o -o jsolve

jsolve32:	JSolveMain32.o JSolve32.o Makefile
	gcc -m32 JSolveMain32.o JSolve32.o -o jsolve32

clean:
	rm jsolve *.o

dist:	JSolve12.zip

JSolve12.zip:	JSolveMain.c JSolve.h JSolve.c Makefile README.txt
	rm -f JSolve12.zip
	zip JSolve12.zip JSolveMain.c JSolve.h JSolve.c Makefile README.txt

JSolveMain.o:	JSolveMain.c JSolve.h
	gcc -O3 -Wall -g -m64 -c JSolveMain.c -o JSolveMain.o

JSolve.o:	JSolve.c JSolve.h
	gcc -O3 -Wall -g -m64 -c JSolve.c -o JSolve.o

JSolveMain32.o:	JSolveMain.c JSolve.h
	gcc -O3 -Wall -g -m32 -c JSolveMain.c -o JSolveMain32.o

JSolve32.o:	JSolve.c JSolve.h
	gcc -O3 -Wall -g -m32 -c JSolve.c -o JSolve32.o
