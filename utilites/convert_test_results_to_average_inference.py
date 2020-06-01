
import sys
import re
# This is not the most elgant way of doing this
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <result.txt>")

timing_lines = []
with open(sys.argv[1], mode='r') as f:
    for line in f.readlines():
        if "Predicted in" in line:
            timing_lines.append(line)

# Remove Lines that does not have a normal itteration number
extracted_data = []
print(timing_lines[0])
print(f"Number of lines extracted {len(timing_lines)}")
for lines in timing_lines:
    found_sentence = re.findall(r'\s+\d+.\d+', lines)
    if len(found_sentence) == 1:
       extracted_data.append(float(found_sentence[0])) 
average_inf = sum(extracted_data) /  len(extracted_data)
print(f"Average Inference Time {average_inf} and Average FPS {1 / (average_inf / 1000)}")
#with open(sys.argv[1].split('.')[0] + 'csv', mode='w') as f:
#    f.write('itteration, total_loss, average_loss\n')
#    for data in extracted_data:
#        f.write(f'{data[0]}, {data[1]}, {data[2]}\n')
#    print("Finished converting training log to csv file")
#print(f"Found {len(training_lines)} lines containing 'average loss'")
