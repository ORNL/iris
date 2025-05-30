include ../makefile_defs.mk

all: qiree_task

qiree_task: qiree_task.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f qiree_task
