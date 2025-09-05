CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -shared -ldl

TARGET = libcuda_ipc_trace.so
SRC = cuda_ipc_trace.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

