#!/usr/bin/env python3

import os
import sys
import gzip
import pandas as pd

def convert_txt_gz_to_df(input_file_path):
    # base columns for PEMS
    columns = [
        'Timestamp', 'Station', 'District', 'Freeway #', 'Direction of Travel',
        'Lane Type', 'Station Length', 'Samples', '% Observed',
        'Total Flow', 'Avg Occupancy', 'Avg Speed'
    ]
    all_rows = []

    with gzip.open(input_file_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # first 12 fields
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

            # any extra lane‐by‐lane fields
            lane_count = 1
            idx = 12
            while idx + 4 < len(parts):
                prefix = f"Lane {lane_count}"
                extras = [
                    (' Samples', parts[idx]),
                    (' Flow',    parts[idx+1]),
                    (' Avg Occ', parts[idx+2]),
                    (' Avg Speed', parts[idx+3]),
                    (' Observed', parts[idx+4])
                ]
                for suf, val in extras:
                    col = prefix + suf
                    row[col] = val
                    if col not in columns:
                        columns.append(col)
                idx += 5
                lane_count += 1

            all_rows.append(row)

    return pd.DataFrame(all_rows, columns=columns)


def preprocess_df(df):
    # timestamp → datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M:%S')

    # categorical conversions
    for cat in ['District', 'Freeway #', 'Direction of Travel', 'Lane Type']:
        if cat in df.columns:
            df[cat] = df[cat].astype('category')

    # numeric conversions
    numeric_fields = ['Station Length', 'Samples', '% Observed', 'Total Flow', 'Avg Occupancy', 'Avg Speed']
    for fld in numeric_fields:
        if fld in df.columns:
            df[fld] = pd.to_numeric(df[fld], errors='coerce')

    # fill missing numerics with mean
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # fill missing categoricals with mode
    cat_cols = df.select_dtypes(include='category').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # normalize Avg Speed into [0,1]
    if 'Avg Speed' in df.columns:
        mn, mx = df['Avg Speed'].min(), df['Avg Speed'].max()
        df['Avg Speed'] = (df['Avg Speed'] - mn) / (mx - mn)

    return df


def process_file(in_path, out_path):
    print(f"→ Processing {os.path.basename(in_path)} …", end=' ')
    df = convert_txt_gz_to_df(in_path)
    df = preprocess_df(df)
    df.to_csv(out_path, index=False)
    print("done")


def main():
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python process_pems.py <input_dir> [<output_dir>]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else input_dir

    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.txt.gz'):
            in_path  = os.path.join(input_dir, fname)
            base     = fname[:-7]           # strip .txt.gz
            out_name = f"{base}.csv"
            out_path = os.path.join(output_dir, out_name)
            process_file(in_path, out_path)

    print("All files processed.")

if __name__ == '__main__':
    main()
