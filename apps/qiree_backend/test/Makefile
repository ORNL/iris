
all: my_library test_quiree

my_library:
	gcc -fPIC -shared -o libexample.so my_library.c

test_quiree:
	gcc -o test_quiree test_quiree.c -ldl

clean:
	rm libexample.so test_quiree
