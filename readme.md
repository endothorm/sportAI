# Sport AI

This project is a resnet-based neural network designed to recognize 100 different types of sports with about 60% accuracy. This was inspired by the fact that the Olympics are currently occuring. The main python file has 92 lines of code. Most of the information the neural net has about the sports comes from Wikipedia. 

## The Algorithm

The algorithm is a retrained resnet18 model, using a dataset of 100 sports downloaded from kaggle. It was trained for 15 epochs. It takes in a test image of the sport and outputs the class index, class name (the sport name) and a short description of the sport based on the class index.

[View a video explanation here](video link)

## Reproducing the Model
1. Code a basic Python image recognition script based on the my_recognition project.
2. Download the dataset from Kaggle:  https://www.kaggle.com/datasets/gpiosenka/sports-classification/data
3. Retrain the Resnet-18 Imagenet model based on this dataset for 15 epochs, with a batch size of 4 and 1 worker.
4. Add short descriptions based on the class indices for each sport in a list; print out the index of the list.
5. Export the model and hardcode it into the python file.
6. Run on a test image!
This project requires the jetson-inference and jetson-utils libraries, as well as pytorch.
