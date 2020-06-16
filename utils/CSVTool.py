# -*-coding:utf-8 -*-
import csv


class CSVReader(object):
    def __init__(self, filename):
        self.__reader = csv.reader(open(filename, 'r', encoding='utf-8'))

    def getData(self):
        data = []
        for row in self.__reader:
            data.append(row)
        return data


class CSVWriter(object):
    def __init__(self, filename):
        self.__writer = csv.writer(open(filename, 'w', encoding='utf-8', newline=''), dialect='excel')

    def writeData(self, results):
        for row in results:
            if len(row) != 0:
                self.__writer.writerow(row)


if __name__ == '__main__':
    data = CSVReader('').getData()
    print(data[1])
