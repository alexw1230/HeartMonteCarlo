#include "decomp.h"
#include "heart.h"

#include <string.h>

static const char *NAMES[N_METHODS + 1] = {"?", "m1", "m2v", "m2h", "m3"};

int method_from_str(const char *s) {
    for (int m = 1; m <= N_METHODS; m++) {
        if (!strcmp(s, NAMES[m])) {
            return m;
        }
    }
    return -1;
}

const char *method_name(int method) {
    if (method < 1 || method > N_METHODS) {
        return NAMES[0];
    }
    return NAMES[method];
}

int isqrt_exact(int p) {
    if (p < 1) {
        return -1;
    }
    for (int q = 1; q * q <= p; q++) {
        if (q * q == p) {
            return q;
        }
    }
    return -1;
}

//k-th cut of [lo, hi] into q equal pieces
//Neighbors compute their shared edge with the exact same expression -> bit identical, no gaps
//Last edge forced to hi so rounding can't shave off the end of the box
static double cut(double lo, double hi, int k, int q) {
    if (k == q) {
        return hi;
    }
    return lo + (hi - lo) * k / q;
}

int subbox(int method, int rank, int p, box_t *b) {
    if (p < 1 || rank < 0 || rank >= p) {
        return -1;
    }

    //Start from the full box
    b->x0 = BOX_X0;
    b->x1 = BOX_X1;
    b->y0 = BOX_Y0;
    b->y1 = BOX_Y1;

    switch (method) {
    case M1:
        break; //Whole box
    case M2V:
        b->x0 = cut(BOX_X0, BOX_X1, rank, p);
        b->x1 = cut(BOX_X0, BOX_X1, rank + 1, p);
        break;
    case M2H:
        b->y0 = cut(BOX_Y0, BOX_Y1, rank, p);
        b->y1 = cut(BOX_Y0, BOX_Y1, rank + 1, p);
        break;
    case M3: {
        int q = isqrt_exact(p);
        if (q < 0) {
            return -1;
        }
        int i = rank % q; //Column (x)
        int j = rank / q; //Row (y)
        b->x0 = cut(BOX_X0, BOX_X1, i, q);
        b->x1 = cut(BOX_X0, BOX_X1, i + 1, q);
        b->y0 = cut(BOX_Y0, BOX_Y1, j, q);
        b->y1 = cut(BOX_Y0, BOX_Y1, j + 1, q);
        break;
    }
    default:
        return -1;
    }
    return 0;
}

double box_area(const box_t *b) {
    return (b->x1 - b->x0) * (b->y1 - b->y0);
}

int box_contains(const box_t *b, double x, double y) {
    int in_x = (x >= b->x0) && (x < b->x1 || (x == b->x1 && b->x1 == BOX_X1));
    int in_y = (y >= b->y0) && (y < b->y1 || (y == b->y1 && b->y1 == BOX_Y1));
    return in_x && in_y;
}