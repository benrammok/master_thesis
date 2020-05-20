
training_lines = []
with open('training_log_yolov3.txt', mode='r') as f:
    for line in f.readlines():
        if "avg loss" in line:
            training_lines.append(line)

# Remove Lines that does not have a normal itteration number
extracted_data = []
for lines in training_lines:
    split_line = lines.split(':')
    # This removes bad lines, that is lines which does not contain an itteration number
    if len(split_line) == 2 and split_line[0] != '':
        itteration = int(lines.split(':')[0])
        total_loss = float(split_line[1].split(',')[0])
        average_loss = float(split_line[1].split(',')[1].strip().split(' ')[0])
        extracted_data.append((itteration, total_loss, average_loss))

with open('training_results_yolov3.csv', mode='w') as f:
    f.write('itteration, total_loss, average_loss\n')
    for data in extracted_data:
        f.write(f'{data[0]}, {data[1]}, {data[2]}\n')
    print("Finished converting training log to csv file")
#print(f"Found {len(training_lines)} lines containing 'average loss'")
