# the PeMS data comes in txt 
# for me to do pandas with them, I need csv
# therefore, this script from convert it from txt to csv
import pandas as pd
import sys

if len(sys.argv) != 3:
    print("Usage: python script_name.py input_file_path output_file_path")
    sys.exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

columns = [
    'Timestamp', 'Station', 'District', 'Freeway #', 'Direction of Travel',
    'Lane Type', 'Station Length', 'Samples', '% Observed', 'Total Flow', 'Avg Occupancy',
    'Avg Speed'
]

with open(input_file_path, 'r') as file:
    data = file.readlines()


all_rows = []

for line in data:
    line = line.strip()
    if not line:
        continue
    parts = line.split(',')

    row = {
        'Timestamp': parts[0],
        'Station': parts[1],
        'District': parts[2],
        'Freeway #': parts[3],
        'Direction of Travel': parts[4],
        'Lane Type': parts[5],
        'Station Length': parts[6],
        'Samples': parts[7],
        '% Observed': parts[8],
        'Total Flow': parts[9],
        'Avg Occupancy': parts[10],
        'Avg Speed': parts[11]
    }

    lane_count = 1
    index = 12
    while index < len(parts):
        lane_prefix = f"Lane {lane_count}"
        row[f'{lane_prefix} Samples'] = parts[index]
        row[f'{lane_prefix} Flow'] = parts[index + 1]
        row[f'{lane_prefix} Avg Occ'] = parts[index + 2]
        row[f'{lane_prefix} Avg Speed'] = parts[index + 3]
        row[f'{lane_prefix} Observed'] = parts[index + 4]
        index += 5 
        lane_count += 1
        if f'{lane_prefix} Samples' not in columns:
            columns.extend([f'{lane_prefix} Samples', f'{lane_prefix} Flow', f'{lane_prefix} Avg Occ', f'{lane_prefix} Avg Speed', f'{lane_prefix} Observed'])

    all_rows.append(row)

df = pd.DataFrame(all_rows, columns=columns)

df.to_csv(output_file_path, index=False)

print(f'Data successfully written to {output_file_path}')
