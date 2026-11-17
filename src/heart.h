#ifndef HEART_H
#define HEART_H

//Heart curve: (x^2 + y^2 - 1)^3 - x^2 y^3 = 0

//Bounding box from the assignment
//Rounded up so the heart fits (true max |x| = 1.1390282, max y = 1.2366592, min y = -1)
#define BOX_X0 (-1.13903)
#define BOX_X1 ( 1.13903)
#define BOX_Y0 (-1.0)
#define BOX_Y1 ( 1.23666)
#define BOX_W  (BOX_X1 - BOX_X0)
#define BOX_H  (BOX_Y1 - BOX_Y0)
#define BOX_B  (BOX_W * BOX_H) //~5.0952457

//Reference area, more digits than the given 3.66197
//From polar quadrature, r(theta) has one root per ray, good to ~1e-12
#define A_REF  3.66197272582

//1 if (x, y) inside or on the heart, 0 if outside
//9 flops, no branches, no pow()
static inline int in_heart(double x, double y) {
    double x2 = x * x;                    //1 mul
    double y2 = y * y;                    //1 mul
    double t  = x2 + y2 - 1.0;            //2 add
    return (t * t * t - x2 * y2 * y) <= 0.0; //4 mul, 1 sub
}

#define FLOPS_IN_HEART 9

#endif