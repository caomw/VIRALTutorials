CC		:=	clang++
LINKER		:=	clang++
FLAGS		:=	-std=c++11 -O3
INCS		:=	-I./include/
LIBS		:=	-lprotobuf -llmdb
APP_NAME	:=	"NNSearch"

all: caffe.pb.o main.o
	$(LINKER) main.o caffe.pb.o -o $(APP_NAME) $(LIBS)
caffe.pb.o: src/caffe.pb.cpp
	$(CC) -c src/caffe.pb.cpp $(INCS) $(FLAGS)
main.o: main.cpp
	$(CC) -c main.cpp $(INCS) $(FLAGS)
clean:
	rm -rf *.o $(APP_NAME)
