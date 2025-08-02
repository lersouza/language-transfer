import argparse
import tensorflow as tf


def show_dataset_content(dataset_path, start: int = 0, max_records: int = 3):
    dataset = tf.data.TFRecordDataset(dataset_path)
    current_position = 0
    records_displayed = 0

    for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        example_content = example.features.feature["text"].bytes_list.value[0].decode("utf-8")

        if current_position + len(example_content) >= start:
            print(example_content)
            print("=" * 100)
            records_displayed += 1
        
        current_position += len(example_content)

        if records_displayed >= max_records:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show content of a TensorFlow dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the TensorFlow dataset file.")
    parser.add_argument("--start", type=int, default=0, help="Start position (chars) for displaying records.")
    parser.add_argument("--max_records", type=int, default=3, help="Maximum number of records to display.")

    args = parser.parse_args()

    show_dataset_content(args.dataset_path, args.start)