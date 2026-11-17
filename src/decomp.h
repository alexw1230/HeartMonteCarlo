#ifndef DECOMP_H
#define DECOMP_H

//Sampling methods
//M1  = every rank samples the whole box
//M2V = p vertical slices (cut along x)
//M2H = p horizontal slices (cut along y)
//M3  = sqrt(p) x sqrt(p) cells
typedef enum {
    M1  = 1,
    M2V = 2,
    M2H = 3,
    M3  = 4
} method_t;

#define N_METHODS 4

//Axis aligned box [x0, x1) x [y0, y1)
typedef struct {
    double x0, x1;
    double y0, y1;
} box_t;

//Method name <-> enum, -1 / "?" if unknown
int method_from_str(const char *s);
const char *method_name(int method);

//sqrt(p) if p is a perfect square, else -1
int isqrt_exact(int p);

//This rank's sub box for a method, 0 on success, -1 on bad input
//(unknown method, rank out of range, p not a perfect square for M3)
int subbox(int method, int rank, int p, box_t *b);

double box_area(const box_t *b);

//1 if (x, y) in b, upper edges count only on the outer edge of the full box
int box_contains(const box_t *b, double x, double y);

#endif