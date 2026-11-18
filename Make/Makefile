#Run from repo root: make -f Make/Makefile [heart|test|clean]

CC      = mpicc
TESTCC  = gcc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS  = -lm

SRC   = src
TESTS = tests
BUILD = build
OBJS  = $(BUILD)/main.o $(BUILD)/decomp.o $(BUILD)/kernel.o

#MPI driver
heart: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

$(BUILD)/main.o:   $(SRC)/main.c $(SRC)/heart.h $(SRC)/rng.h $(SRC)/decomp.h $(SRC)/kernel.h
$(BUILD)/decomp.o: $(SRC)/decomp.c $(SRC)/decomp.h $(SRC)/heart.h
$(BUILD)/kernel.o: $(SRC)/kernel.c $(SRC)/kernel.h $(SRC)/decomp.h $(SRC)/heart.h $(SRC)/rng.h

#Serial tests, no MPI needed
test: $(BUILD)/tests $(BUILD)/decomp_tests $(BUILD)/kernel_tests
	./$(BUILD)/tests
	./$(BUILD)/decomp_tests
	./$(BUILD)/kernel_tests

$(BUILD)/tests: $(TESTS)/tests.c $(SRC)/heart.h $(SRC)/rng.h | $(BUILD)
	$(TESTCC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDLIBS)

#decomp.c / kernel.c compiled straight in with gcc, separate from the mpicc objects
$(BUILD)/decomp_tests: $(TESTS)/decomp_tests.c $(SRC)/decomp.c $(SRC)/decomp.h $(SRC)/heart.h $(SRC)/rng.h | $(BUILD)
	$(TESTCC) $(CFLAGS) -I$(SRC) -o $@ $(TESTS)/decomp_tests.c $(SRC)/decomp.c $(LDLIBS)

$(BUILD)/kernel_tests: $(TESTS)/kernel_tests.c $(SRC)/kernel.c $(SRC)/decomp.c $(SRC)/kernel.h $(SRC)/decomp.h $(SRC)/heart.h $(SRC)/rng.h | $(BUILD)
	$(TESTCC) $(CFLAGS) -I$(SRC) -o $@ $(TESTS)/kernel_tests.c $(SRC)/kernel.c $(SRC)/decomp.c $(LDLIBS)

clean:
	rm -rf heart $(BUILD)

.PHONY: test clean