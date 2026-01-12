import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict
import datetime

DATA_PATH = 'data/'

def prepare_dataset(path, name, target_name):
    """
    path: Path to the original source data folder
    name: Name of the original dataset
    target_name: Name of the new dataset folder to be generated
    """
    
    # 1. Read all raw data
    files = ['train', 'valid', 'test']
    all_raw_data = []
    entities, relations, timestamps = set(), set(), set()
    
    print(f"Loading raw data from {path}...")
    for f in files:
        file_path = os.path.join(path, f)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as to_read:
                for line in to_read.readlines():
                    try:
                        line_split = line.strip().split('\t')
                        if len(line_split) >= 4:
                            lhs, rel, rhs, timestamp = line_split[:4]
                            all_raw_data.append((lhs, rel, rhs, timestamp))
                            entities.add(lhs)
                            entities.add(rhs)
                            relations.add(rel)
                            timestamps.add(timestamp)
                    except ValueError:
                        continue
        else:
            print(f"Warning: {f} file not found in {path}, skipping.")

    if not all_raw_data:
        raise ValueError("No data found!")

    # 2. Build mappings
    sorted_timestamps = sorted(list(timestamps)) 
    sorted_entities = sorted(list(entities))
    sorted_relations = sorted(list(relations))

    entities_to_id = {x: i for (i, x) in enumerate(sorted_entities)}
    relations_to_id = {x: i for (i, x) in enumerate(sorted_relations)}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted_timestamps)}
    
    # Reverse mapping for timestamp display (ID -> Timestamp String)
    id_to_timestamps = {i: x for (i, x) in enumerate(sorted_timestamps)}

    n_relations = len(relations)
    n_entities = len(entities)
    n_timestamps = len(timestamps)

    # --- PRINT DATASET STATS ---
    print("\n" + "="*40)
    print(f"Dataset Statistics for: {target_name}")
    print(f"{'Entities:':<15} {n_entities}")
    print(f"{'Relations:':<15} {n_relations}")
    print(f"{'Timestamps:':<15} {n_timestamps}")
    print("="*40 + "\n")

    # Create output directory
    output_dir = os.path.join(DATA_PATH, target_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Save mapping files
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        with open(os.path.join(output_dir, f), 'w+', encoding='utf-8') as ff:
            for (x, i) in dic.items():
                ff.write("{}\t{}\n".format(x, i))

    # 4. Convert data to ID format
    all_examples = []
    for lhs, rel, rhs, ts in all_raw_data:
        try:
            all_examples.append([
                entities_to_id[lhs], 
                relations_to_id[rel], 
                entities_to_id[rhs], 
                timestamps_to_id[ts]
            ])
        except KeyError:
            continue
    
    all_examples = np.array(all_examples).astype('uint64')

    # 5. Sort by time and split 8:1:1
    print("Sorting by timestamp and splitting 8:1:1...")
    ind = np.argsort(all_examples[:, 3])
    all_examples = all_examples[ind]

    n_total = len(all_examples)
    n_train = int(n_total * 0.80)
    n_valid = int(n_total * 0.10)

    train_data = all_examples[:n_train]
    valid_data_raw = all_examples[n_train : n_train + n_valid]
    test_data_raw  = all_examples[n_train + n_valid :]

    # 6. Filter unseen entities in Valid/Test sets
    print("Filtering unseen entities in Valid/Test sets...")
    
    train_entities = set(train_data[:, 0]) | set(train_data[:, 2])
    print(f" -> Unique entities in Train set: {len(train_entities)}")

    def filter_unseen(data, split_name):
        initial_len = len(data)
        filtered_list = []
        for ex in data:
            s, r, o, t = ex
            if s in train_entities and o in train_entities:
                filtered_list.append(ex)
        
        filtered_data = np.array(filtered_list).astype('uint64')
        removed_count = initial_len - len(filtered_data)
        print(f" -> {split_name}: Removed {removed_count} examples containing unseen entities. (Remaining: {len(filtered_data)})")
        return filtered_data

    valid_data = filter_unseen(valid_data_raw, "Valid")
    test_data = filter_unseen(test_data_raw, "Test")

    split_data = {
        'train': train_data,
        'valid': valid_data,
        'test':  test_data
    }

    # --- PRINT SPLIT DETAILS AND TIME RANGES ---
    print("\n" + "="*60)
    print(f"{'Split':<10} | {'Quadruples':<12} | {'Time Range (Start -> End)'}")
    print("-" * 60)

    for split_name, data in split_data.items():
        if len(data) > 0:
            ts_ids = data[:, 3]
            min_ts_id = np.min(ts_ids)
            max_ts_id = np.max(ts_ids)
            
            # Map IDs back to original timestamp strings for display
            start_time_str = id_to_timestamps[min_ts_id]
            end_time_str = id_to_timestamps[max_ts_id]
            
            print(f"{split_name.capitalize():<10} | {len(data):<12} | {start_time_str} -> {end_time_str}")
        else:
             print(f"{split_name.capitalize():<10} | {0:<12} | N/A")
    print("="*60 + "\n")


    # 7. Save data
    for split_name, data in split_data.items():
        with open(os.path.join(output_dir, split_name + '.pickle'), 'wb') as out:
            pickle.dump(data, out)

    # 8. Create filtering lists (Filtering Lists) 
    print("Creating filtering lists and probas...")
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in ['train', 'valid', 'test']:
        examples = split_data[f]
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    with open(os.path.join(output_dir, 'to_skip.pickle'), 'wb') as out:
        pickle.dump(to_skip_final, out)

    # Compute probabilities
    examples = split_data['train']
    
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    
    for k, v in counters.items():
        sum_v = np.sum(v)
        if sum_v > 0:
            counters[k] = v / sum_v
        else:
            counters[k] = v 

    with open(os.path.join(output_dir, 'probas.pickle'), 'wb') as out:
        pickle.dump(counters, out)
    
    print(f"Dataset {target_name} processed successfully.\n")

if __name__ == "__main__":
    datasets = ['ICEWS14', 'ICEWS05-15']
    for d in datasets:
        new_dataset_name = f"{d}_time"
        print(f"Processing {d} -> {new_dataset_name}")
        src_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'src_data', d
        )
        try:
            prepare_dataset(src_path, d, new_dataset_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
            else:
                raise