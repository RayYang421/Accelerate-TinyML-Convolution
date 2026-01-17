CC := gcc
CFLAGS := -O2 -Wall 

TARGET := Accelerate-TinyML-Convolution
SRCS := Accelerate-TinyML-Convolution.c

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

valid:
	python Valid.py


clean:
	rm -f $(TARGET) *.o output.txt
