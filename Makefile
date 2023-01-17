all: train

train:
	python main.py

clean:
	rm -f *.pth
	rm -f *.log
