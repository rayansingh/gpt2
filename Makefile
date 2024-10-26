# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_60 -std=c++11 -rdc=true -g

# Source files
OUTPUT_VERIFICATION_SRC = output_verification.cu
GPT2_SRC = gpt2.cu

# Include and library paths for cuBLAS
CUBLAS_INCLUDES = -I/usr/local/cuda/include
CUBLAS_LIBS = -L/usr/local/cuda/lib64 -lcublas -lcublasLt

# Default target
all: output_verification

# Individual targets
.output_verification: output_verification
output_verification: $(OUTPUT_VERIFICATION_SRC)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c $(OUTPUT_VERIFICATION_SRC) -o output_verification.o
	$(NVCC) $(CFLAGS) -o output_verification output_verification.o $(CUBLAS_LIBS)

.gpt2: gpt2
gpt2: $(GPT2_SRC)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c $(GPT2_SRC) -o gpt2.o
	$(NVCC) $(CFLAGS) -o gpt2 gpt2.o $(CUBLAS_LIBS)

# Clean target
clean:
	rm -f output_verification output_verification.o gpt2 gpt2.o

