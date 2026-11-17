#Run from repo root: make -f Make/Makefile [heart|test|clean]

CC      = mpicc
TESTCC  = gcc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS  = -lm

SRC   = src
TESTS = tests
BUILD = build
OBJS  = $(BUILD)/main.o $(BUILD)/decomp.o

#MPI driver (main.c comes in steps 5-6)
heart: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

$(BUILD)/main.o:   $(SRC)/main.c $(SRC)/heart.h $(SRC)/rng.h $(SRC)/decomp.h
$(BUILD)/decomp.o: $(SRC)/decomp.c $(SRC)/decomp.h $(SRC)/heart.h

#Serial tests, no MPI needed
test: $(BUILD)/tests $(BUILD)/decomp_tests
	./$(BUILD)/tests
	./$(BUILD)/decomp_tests

$(BUILD)/tests: $(TESTS)/tests.c $(SRC)/heart.h $(SRC)/rng.h | $(BUILD)
	$(TESTCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDLIBS)

#decomp.c compiled straight in with gcc, separate from the mpicc objects
$(BUILD)/decomp_tests: $(TESTS)/decomp_tests.c $(SRC)/decomp.c $(SRC)/decomp.h $(SRC)/heart.h $(SRC)/rng.h | $(BUILD)
	$(TESTCC) $(CFLAGS) -I$(SRC) -o $@ $(TESTS)/decomp_tests.c $(SRC)/decomp.c $(LDLIBS)

clean:
	rm -rf heart $(BUILD)

.PHONY: test clean