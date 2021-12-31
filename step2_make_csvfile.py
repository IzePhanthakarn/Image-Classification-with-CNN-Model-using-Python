import csv

nclass = 4
nimgperclass = 400
ntrain = 300
ntest = 100
csv_train_filename = 'train_dataset.csv'
csv_test_filename = 'test_dataset.csv'
csvData = [['filename', 'class']]

# write train csv
with open(csv_train_filename, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
    for i in range(nclass):
        for j in range(ntrain):
            csvdataimage = [[str((j+1)+(i*nimgperclass))+'.jpg', str(i)]]
            writer.writerows(csvdataimage)
csvFile.close()

# write test csv
with open(csv_test_filename, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
    for i in range(nclass):
        for j in range(ntrain, nimgperclass):
            csvdataimage = [[str((j+1)+(i*nimgperclass))+'.jpg', str(i)]]
            writer.writerows(csvdataimage)
csvFile.close()

print(csv_train_filename)
print(csv_test_filename)
print("Make CSV Files Done")
