README.txt
----------
JSolve - a very fast Sudoku solver.
	int JSolve(const char *clues,char *result,int max_solutions);

clues - a null terminated 81 character string with '1' through '9' for
	digits and either '.' or '0' for empty cells
result - If you want the soltuion, pass in a pointer to 82 characters
	of storage where you want the solution written. Pass in 0 to
	ignore the solution.
max_solutions - the largest number of solutions to be searched for
	Typically 2 to check for soluble, not soluble, or multipule solutions.
	Pass in a very large number to count all solutions.
<returns> - the number of solutions found

The solver (JSolve.c and JSolve.h) should be quite portable.

The main routine (JSolveMain.c) works well under MacOS and should work under Linux
and Windows. I would welcome it if someone could improve the Windows support for the
main routine. Accurate timing under Windows appears to be tricky.

This code was inspired by BB Sudoku (by Brian Turner),
<http://www.setbb.com/phpbb/viewtopic.php?t=1824>. Some of the algorithms
are taken from BB Sudoku, but all of the code was re-written from scratch.

Enjoy
Jason Linhart
<jason@enjoysudoku.com>
